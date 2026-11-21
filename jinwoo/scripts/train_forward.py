"""Train the force-curve and stable-angle surrogates after joining jobs.csv."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from preprocess_forward import FEATURES, GRID, stability

FORCE_COLUMNS = [f"force_N_at_{u:05.2f}_mm" for u in GRID]
ANGLE_COLUMN = "stable_angle_upper_deg"
STATE_CODE = {"partial": 0, "full": 1}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return row["geom_id"], row["state"]


def design_matrix(rows):
    values = np.array([[*(float(row[name]) for name in FEATURES), STATE_CODE[row["state"]]]
                       for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite geometry input.")
    return values


def arrays(rows, target):
    x = design_matrix(rows)
    y = np.array([[float(row[name]) for name in target] for row in rows])
    if not np.isfinite(y).all():
        raise ValueError("Non-finite training target.")
    return x, y


def fit_pca(y):
    mean = y.mean(axis=0)
    _, _, basis = np.linalg.svd(y - mean, full_matrices=False)
    return mean, basis


def rms(predicted, actual):
    return float(np.sqrt(np.mean(np.square(predicted - actual))))


def evaluate_stability(curves, rows, labels, tolerance):
    counts = Counter()
    predictions = []
    errors = []
    for curve, row in zip(curves, rows):
        predicted, crossing = stability(curve, True, tolerance)
        actual = labels[key(row)]
        counts[actual["stability_label"], predicted] += 1
        u2 = None
        if crossing:
            left, right, ratio = crossing
            u2 = float(GRID[left] + ratio * (GRID[right] - GRID[left]))
            if actual["u2_mm"]:
                errors.append(abs(u2 - float(actual["u2_mm"])))
        predictions.append((predicted, u2))
    classes = ("bistable", "monostable", "unknown")
    tp = counts["bistable", "bistable"]
    fp = counts["monostable", "bistable"]
    actual_positive = sum(counts["bistable", x] for x in classes)
    return {
        "confusion": {a: {p: counts[a, p] for p in classes} for a in classes},
        "bistable_precision": tp / (tp + fp) if tp + fp else None,
        "bistable_recall": tp / actual_positive if actual_positive else None,
        "u2_mae_mm": float(np.mean(errors)) if errors else None,
        "u2_count": len(errors),
    }, predictions


def train(args):
    labels = {key(row): row for row in read_csv(args.data / "case_index.csv")}
    force = read_csv(args.data / "force_curves.csv")
    angle = read_csv(args.data / "stable_angles.csv")
    if not force or not angle:
        raise ValueError("Both force and stable-angle targets are required.")
    missing = {row["geom_id"] for row in force + angle if any(not row.get(name) for name in FEATURES)}
    if missing:
        raise ValueError(
            f"{len(missing)} geometry IDs have missing features. "
            "Rerun preprocess_forward.py with the original --jobs jobs.csv."
        )
    groups = sorted({row["geom_id"] for row in force if row["role"] == "train"})
    if len(groups) < 3:
        raise ValueError("At least three TR geometries with force targets are required.")
    rng = np.random.default_rng(args.seed)
    count = max(1, round(0.2 * len(groups)))
    validation_ids = set(rng.permutation(groups)[:count])

    def split(row):
        if row["role"] == "test":
            return "test"
        return "validation" if row["geom_id"] in validation_ids else "train"

    force_split = {part: [row for row in force if split(row) == part]
                   for part in ("train", "validation", "test")}
    angle_tr = [row for row in angle if row["role"] == "train"]
    if not all(force_split.values()) or not angle_tr:
        raise ValueError("The force train/validation/test split and angle train split must be nonempty.")
    import xgboost as xgb

    def fit(x, target):
        params = dict(objective="reg:squarederror", max_depth=3, eta=0.05,
                      subsample=0.85, colsample_bytree=0.9, seed=args.seed,
                      nthread=2, tree_method="hist")
        return xgb.train(params, xgb.DMatrix(x, label=target, nthread=2), num_boost_round=300)

    x_tr, y_tr = arrays(force_split["train"], FORCE_COLUMNS)
    x_val, y_val = arrays(force_split["validation"], FORCE_COLUMNS)
    mean, basis = fit_pca(y_tr)
    options = [k for k in (1, 2, 4, 8, 12, 16) if k <= min(len(x_tr) - 1, len(GRID))]
    models = [fit(x_tr, (y_tr - mean) @ component) for component in basis[:max(options)]]
    scores = np.column_stack([model.inplace_predict(x_val) for model in models])
    selection = []
    for k in options:
        predicted = mean + scores[:, :k] @ basis[:k]
        oracle = mean + ((y_val - mean) @ basis[:k].T) @ basis[:k]
        stats, _ = evaluate_stability(predicted, force_split["validation"], labels, args.force_tol)
        p, r = stats["bistable_precision"], stats["bistable_recall"]
        f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r else 0.0
        selection.append(dict(k=k, rmse_n=rms(predicted, y_val), pca_only_rmse_n=rms(oracle, y_val),
                              bistable_f1=f1, **stats))
    best = min(selection, key=lambda row: (-row["bistable_f1"], row["rmse_n"], row["k"]))
    k = best["k"]

    force_tr = [row for row in force if row["role"] == "train"]
    x, y = arrays(force_tr, FORCE_COLUMNS)
    mean, basis = fit_pca(y)
    basis = basis[:k]
    scores = (y - mean) @ basis.T
    models = [fit(x, column) for column in scores.T]
    x_angle, y_angle = arrays(angle_tr, [ANGLE_COLUMN])
    angle_model = fit(x_angle, y_angle[:, 0])

    # Predict both surrogates for all TE cases, including incomplete simulations.
    test_cases = [row for row in labels.values() if row["role"] == "test"
                  and all(row.get(name) for name in FEATURES)]
    x_test = design_matrix(test_cases)
    force_predictions = mean + np.column_stack([model.inplace_predict(x_test) for model in models]) @ basis
    angle_predictions = angle_model.inplace_predict(x_test)
    by_key = {key(row): i for i, row in enumerate(test_cases)}
    force_rows = force_split["test"]
    y_test = arrays(force_rows, FORCE_COLUMNS)[1]
    completed_predictions = np.array([force_predictions[by_key[key(row)]] for row in force_rows])
    stability_stats, predictions = evaluate_stability(force_predictions, test_cases, labels, args.force_tol)
    angle_rows = [row for row in angle if row["role"] == "test"]
    angle_errors = [abs(float(angle_predictions[by_key[key(row)]]) - float(row[ANGLE_COLUMN]))
                    for row in angle_rows]
    prediction_rows = []
    for i, (row, (label, u2)) in enumerate(zip(test_cases, predictions)):
        prediction_rows.append(dict(
            geom_id=row["geom_id"], state=row["state"], actual_stability=row["stability_label"],
            predicted_stability=label, actual_u2_mm=row["u2_mm"], predicted_u2_mm=u2,
            actual_angle_deg=row[ANGLE_COLUMN],
            predicted_angle_deg=float(angle_predictions[i]) if label == "bistable" else None,
            **dict(zip(FORCE_COLUMNS, force_predictions[i])),
        ))
    test_by_geom = {}
    for row in prediction_rows:
        test_by_geom.setdefault(row["geom_id"], {})[row["state"]] = row
    pair_counts = Counter()
    for states in test_by_geom.values():
        if set(states) != set(STATE_CODE):
            continue
        partial, full = states["partial"], states["full"]
        actual = (partial["actual_stability"], full["actual_stability"])
        predicted = (partial["predicted_stability"], full["predicted_stability"])
        if "unknown" in actual:
            continue
        target = ("bistable", "monostable")
        pair_counts[f"actual_{int(actual == target)}_predicted_{int(predicted == target)}"] += 1

    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output / "force_pca.npz", mean=mean, components=basis, displacement_mm=GRID)
    for i, model in enumerate(models, 1):
        model.save_model(args.output / f"force_pc_{i:02d}.json")
    angle_model.save_model(args.output / "angle_upper.json")
    write_csv(args.output / "test_predictions.csv", prediction_rows)
    metrics = dict(
        features=[*FEATURES, "state_code"], state_code=STATE_CODE, force_tolerance_n=args.force_tol,
        seed=args.seed, xgboost_version=xgb.__version__, pca_components=k, validation_geometry_ids=sorted(validation_ids),
        validation=selection, test_force_count=len(force_rows), test_force_rmse_n=rms(completed_predictions, y_test),
        test_stability=stability_stats, test_angle_count=len(angle_errors),
        test_angle_mae_deg=float(np.mean(angle_errors)) if angle_errors else None,
        test_pair_pattern_counts=dict(pair_counts),
    )
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {k} force components and one stable-angle model to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path, help="New directory for a training run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-tol", type=float, default=0.01)
    args = parser.parse_args()
    if not np.isfinite(args.force_tol) or args.force_tol <= 0:
        parser.error("--force-tol must be positive and finite")
    if args.output.exists():
        parser.error("output already exists; choose a new run directory")
    try:
        train(args)
    except (ValueError, ModuleNotFoundError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
