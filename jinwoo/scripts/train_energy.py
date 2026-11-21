"""Train the energy–displacement surrogate from completed ANSYS runs."""
import argparse
import io
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import xgboost as xgb

from preprocess_forward import GRID
from train_forward import arrays, fit_pca, key, read_csv, rms, write_csv

ENERGY_COLUMNS = [f"energy_mJ_at_{u:05.2f}_mm" for u in GRID]
TARGET = ("bistable", "monostable")


def energy_features(u, energy):
    """Find an interior maximum followed by an interior minimum."""
    steps = np.flatnonzero(np.diff(energy))
    slopes = np.sign(np.diff(energy)[steps])
    changes = np.flatnonzero(slopes[:-1] != slopes[1:])
    maxima = []
    barriers = []
    for j in changes:
        point = (int(steps[j]) + 1 + int(steps[j + 1])) // 2
        if slopes[j] > 0:
            maxima.append(point)
        elif maxima:
            peak = maxima[-1]
            barriers.append((energy[peak] - energy[point], peak, point))
    if barriers:
        reverse, peak, well = max(barriers)
        return dict(label="bistable", u1_mm=float(u[peak]), u2_mm=float(u[well]),
                    forward_barrier_mJ=float(energy[peak] - energy[0]),
                    reverse_barrier_mJ=float(reverse))
    label = "monostable" if len(slopes) and np.all(slopes > 0) else "unknown"
    return dict(label=label, u1_mm=None, u2_mm=None, forward_barrier_mJ=None,
                reverse_barrier_mJ=None)


def prepare(source, data):
    rows = []
    raw_features = {}
    with ExitStack() as stack:
        archives = {}
        for row in read_csv(data / "case_index.csv"):
            if row["force_complete"] != "1" or row["status"] != "ok":
                continue
            name = row["archive"]
            if name not in archives:
                archive = stack.enter_context(ZipFile(source / name))
                members = {n.replace(chr(92), "/"): n for n in archive.namelist()}
                archives[name] = archive, members
            archive, members = archives[name]
            import csv
            samples = list(csv.DictReader(io.StringIO(
                archive.read(members[row["curve_file"]]).decode("utf-8-sig"))))
            u = np.array([float(s["u_applied_mm"]) for s in samples])
            energy = np.array([float(s["U_total_mJ"]) for s in samples])
            if u[0] > 0:
                u, energy = np.r_[0, u], np.r_[0, energy]
            if not np.isfinite(energy).all():
                continue
            raw_features[key(row)] = energy_features(u, energy)
            rows.append({**row, **dict(zip(ENERGY_COLUMNS, np.interp(GRID, u, energy)))})
    return rows, raw_features


def report(rows, predicted, reference):
    counts = Counter()
    pairs = defaultdict(dict)
    false_full = 0
    full_mono = 0
    for row, pred in zip(rows, predicted):
        true = reference[key(row)]
        counts[true, pred] += 1
        pairs[row["geom_id"]][row["state"]] = (true, pred)
        if row["state"] == "full" and true == "monostable":
            full_mono += 1
            false_full += pred == "bistable"
    pair_counts = Counter()
    for pair in pairs.values():
        if set(pair) != {"partial", "full"}:
            continue
        a, b = zip(pair["partial"], pair["full"])
        if "unknown" not in a:
            pair_counts[a == TARGET, b == TARGET] += 1
    tn, fp, fn, tp = (pair_counts[False, False], pair_counts[False, True],
                      pair_counts[True, False], pair_counts[True, True])
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0,
                full_false_bistable=false_full, full_actual_monostable=full_mono,
                confusion={f"{a}_to_{b}": v for (a, b), v in sorted(counts.items())})


def train(source, data, force_run, output):
    rows, raw = prepare(source, data)
    force_labels = {key(r): r["stability_label"] for r in rows}
    raw_labels = {k: v["label"] for k, v in raw.items()}
    force_settings = json.loads((force_run / "metrics.json").read_text(encoding="utf-8"))
    val_ids = set(force_settings["validation_geometry_ids"])
    train_rows = [r for r in rows if r["role"] == "train" and r["geom_id"] not in val_ids]
    val = [r for r in rows if r["role"] == "train" and r["geom_id"] in val_ids]
    test = [r for r in rows if r["role"] == "test"]

    def fit(training, k):
        x, y = arrays(training, ENERGY_COLUMNS)
        mean, components = fit_pca(y)
        components = components[:k]
        params = dict(objective="reg:squarederror", max_depth=3, eta=0.05,
                      subsample=0.85, colsample_bytree=0.9, seed=force_settings["seed"],
                      nthread=2, tree_method="hist")
        models = [xgb.train(params, xgb.DMatrix(x, label=c, nthread=2), num_boost_round=300)
                  for c in ((y - mean) @ components.T).T]
        return mean, components, models

    def scores(x, models):
        return np.column_stack([m.inplace_predict(x) for m in models])

    def labels(curves):
        return [energy_features(GRID, curve)["label"] for curve in curves]

    x, y = arrays(val, ENERGY_COLUMNS)
    mean, basis, models = fit(train_rows, 24)
    v = scores(x, models)
    selection = []
    for k in (4, 8, 12, 16, 24):
        predicted = mean + v[:, :k] @ basis[:k]
        selection.append(dict(k=k, energy_rmse_mJ=rms(predicted, y),
                              vs_raw_energy=report(val, labels(predicted), raw_labels),
                              vs_force=report(val, labels(predicted), force_labels)))
    best = min(selection, key=lambda r: (-r["vs_raw_energy"]["f1"],
                                       r["vs_raw_energy"]["full_false_bistable"],
                                       r["energy_rmse_mJ"]))
    k = best["k"]
    mean, basis, models = fit(train_rows + val, k)
    x, y = arrays(test, ENERGY_COLUMNS)
    predictions = mean + scores(x, models) @ basis
    features = [energy_features(GRID, curve) for curve in predictions]
    force_pred = {key(r): r["predicted_stability"]
                  for r in read_csv(force_run / "test_predictions.csv")}
    baseline = [force_pred[key(r)] for r in test]
    estimated = [f["label"] for f in features]
    prediction_rows = []
    matched = []
    for row, curve, feature in zip(test, predictions, features):
        actual = raw[key(row)]
        if actual["label"] == feature["label"] == "bistable":
            matched.append((actual, feature))
        prediction_rows.append(dict(
            geom_id=row["geom_id"], state=row["state"],
            actual_force_label=force_labels[key(row)], predicted_force_label=force_pred[key(row)],
            **{f"actual_energy_{k}": v for k, v in actual.items()},
            **{f"predicted_energy_{k}": v for k, v in feature.items()},
            **dict(zip(ENERGY_COLUMNS, curve)),
        ))
    error_keys = ("u2_mm", "forward_barrier_mJ", "reverse_barrier_mJ")
    errors = {name: float(np.mean([abs(a[name] - b[name]) for a, b in matched]))
              for name in error_keys} if matched else {}
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "test_predictions.csv", prediction_rows)
    write_csv(output / "energy_curves.csv", [
        {k: row[k] for k in ("geom_id", "state", "role", *force_settings["features"][:-1],
                             *ENERGY_COLUMNS)}
        | {f"actual_energy_{k}": v for k, v in raw[key(row)].items()}
        for row in rows])
    np.savez(output / "energy_pca.npz", mean=mean, components=basis, displacement_mm=GRID)
    for i, model in enumerate(models, 1):
        model.save_model(output / f"energy_pc_{i:02d}.json")
    metrics = dict(
        train_count=len(train_rows) + len(val), validation_count=len(val), test_count=len(test),
        pca_components=k, validation=selection, test_energy_rmse_mJ=rms(predictions, y),
        raw_grid_vs_raw_energy=report(test, labels(y), raw_labels),
        raw_energy_vs_force=report(test, [raw_labels[key(r)] for r in test], force_labels),
        test_vs_raw_energy=report(test, estimated, raw_labels),
        test_force_model_vs_raw_energy=report(test, baseline, raw_labels),
        test_vs_force=report(test, estimated, force_labels),
        test_force_model_vs_force=report(test, baseline, force_labels),
        test_matched_bistable_count=len(matched), test_matched_bistable_mae=errors,
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "validation"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "data", "force_run", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    train(args.source, args.data, args.force_run, args.output)
