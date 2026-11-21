"""Compare five model families for the scalar forward targets."""
import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              VotingClassifier, VotingRegressor)
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier, XGBRegressor
import xgboost

from preprocess_forward import FEATURES
from train_forward import STATE_CODE, design_matrix, read_csv, write_csv

REGRESSION_TARGETS = ("stable_angle_deg", "reverse_barrier_mJ", "deformation_applied_mm")
STRESS_TARGET = "hinge_transition_max_MPa"
ENSEMBLE = "MLP + XGBoost"
TOLERANCES = {"stable_angle_deg": 2.0, "reverse_barrier_mJ": 0.05,
              "deformation_applied_mm": 0.5}
KNOWN = {"bistable": 1, "monostable": 0}


def candidates(seed):
    tree = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.9, random_state=seed, n_jobs=2)
    extra = dict(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=2)
    mlp = dict(hidden_layer_sizes=(32, 16), solver="lbfgs", alpha=0.05,
               max_iter=1000, random_state=seed)
    kernel = Matern(length_scale=2, nu=1.5)
    classifiers = {
        "XGBoost": XGBClassifier(**tree),
        "Gaussian Process": make_pipeline(StandardScaler(), GaussianProcessClassifier(
            kernel=kernel, optimizer=None, random_state=seed)),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(**mlp)),
        "Extra Trees": ExtraTreesClassifier(**extra),
        "RBF SVM": make_pipeline(StandardScaler(), SVC(C=3)),
    }
    regressors = {
        "XGBoost": XGBRegressor(**tree),
        "Gaussian Process": make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel=kernel, alpha=0.01, normalize_y=True, optimizer=None)),
        "MLP": make_pipeline(StandardScaler(), TransformedTargetRegressor(
            regressor=MLPRegressor(**mlp), transformer=StandardScaler())),
        "Extra Trees": ExtraTreesRegressor(**extra),
        "RBF SVM": make_pipeline(StandardScaler(), TransformedTargetRegressor(
            regressor=SVR(C=3, epsilon=0.1), transformer=StandardScaler())),
    }
    return classifiers, regressors


def group_metrics(rows, y, pred):
    by_geom = defaultdict(dict)
    for row, a, b in zip(rows, y, pred):
        by_geom[row["geom_id"]][row["state"]] = (int(a), int(b))
    paired = [states for states in by_geom.values() if set(states) == set(STATE_CODE)]
    actual = [states["partial"][0] == 1 and states["full"][0] == 0 for states in paired]
    predicted = [states["partial"][1] == 1 and states["full"][1] == 0 for states in paired]
    return dict(
        pair_count=len(paired), pair_positive=int(sum(actual)),
        pair_tp=int(sum(a and p for a, p in zip(actual, predicted))),
        pair_fp=int(sum(not a and p for a, p in zip(actual, predicted))),
        pair_f1=float(f1_score(actual, predicted, zero_division=0)),
        pair_precision=float(precision_score(actual, predicted, zero_division=0)),
        pair_recall=float(recall_score(actual, predicted, zero_division=0)),
        accuracy=float(accuracy_score(y, pred)),
        state_accuracy={state: float(accuracy_score(y[mask], pred[mask]))
                        for state in STATE_CODE
                        if (mask := np.array([r["state"] == state for r in rows])).any()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory.")
    rows = read_csv(args.targets)
    tr = [r for r in rows if r["role"] == "train"]
    te = [r for r in rows if r["role"] == "test"]
    tr_ids = sorted({r["geom_id"] for r in tr})
    if set(tr_ids) & {r["geom_id"] for r in te}:
        parser.error("A geometry appears in both TR and TE.")
    folds = GroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_id = {}
    for i, (_, val) in enumerate(folds.split(tr_ids, groups=tr_ids)):
        fold_id.update({tr_ids[j]: i for j in val})
    classifiers, regressors = candidates(args.seed)
    classifiers[ENSEMBLE] = VotingClassifier(
        [("mlp", classifiers["MLP"]), ("xgb", classifiers["XGBoost"])], voting="soft")
    regressors[ENSEMBLE] = VotingRegressor(
        [("mlp", regressors["MLP"]), ("xgb", regressors["XGBoost"])])
    report = {}
    models = {}
    predictions = []
    benchmark = []
    targets = ("stability", *REGRESSION_TARGETS)
    if rows and STRESS_TARGET in rows[0]:
        targets += (STRESS_TARGET,)
    for target in targets:
        classification = target == "stability"
        def eligible(r):
            return (r["stability"] in KNOWN if classification else
                    r["state"] == "partial" and r["stability"] == "bistable" and bool(r[target]))
        def make_xy(source):
            chosen = [r for r in source if eligible(r)]
            return chosen, design_matrix(chosen), np.array([
                KNOWN[r[target]] if classification else float(r[target]) for r in chosen])
        train_rows, x, y = make_xy(tr)
        test_rows, x_test, y_test = make_xy(te)
        split = np.array([fold_id[r["geom_id"]] for r in train_rows])
        tests = classifiers if classification else regressors
        entries = []
        for name, estimator in tests.items():
            oof = np.empty(len(y), dtype=y.dtype)
            per_fold = []
            convergence_warnings = 0
            for i in range(5):
                held = split == i
                with warnings.catch_warnings(record=True) as messages:
                    warnings.simplefilter("always", ConvergenceWarning)
                    fitted = clone(estimator).fit(x[~held], y[~held])
                convergence_warnings += sum(issubclass(w.category, ConvergenceWarning) for w in messages)
                oof[held] = fitted.predict(x[held])
                if classification:
                    metrics = group_metrics([r for r, flag in zip(train_rows, held) if flag],
                                            y[held], oof[held])
                    per_fold.append(metrics["pair_f1"])
                else:
                    per_fold.append(float(mean_absolute_error(y[held], oof[held])))
            values = (group_metrics(train_rows, y, oof) if classification else
                      dict(mae=float(mean_absolute_error(y, oof))))
            if target in TOLERANCES:
                values["within_tolerance"] = float(np.mean(np.abs(y - oof) <= TOLERANCES[target]))
            entries.append(dict(model=name, fold_scores=per_fold,
                                convergence_warnings=convergence_warnings, **values))
            benchmark.append(dict(target=target, model=name,
                                  metric="pair_f1" if classification else "mae",
                                  score=values["pair_f1" if classification else "mae"],
                                  fold_std=float(np.std(per_fold)),
                                  within_tolerance=values.get("within_tolerance"),
                                  convergence_warnings=convergence_warnings))
            print(target, name, values, flush=True)
        winner = (max(entries, key=lambda r: (r["pair_f1"], r["pair_precision"])) if classification
                  else min(entries, key=lambda r: r["mae"]))
        model = clone(tests[winner["model"]]).fit(x, y)
        pred = model.predict(x_test)
        models[target] = model
        test_metrics = (group_metrics(test_rows, y_test, pred) if classification else dict(
            mae=float(mean_absolute_error(y_test, pred)),
            rmse=float(np.sqrt(np.mean((y_test - pred) ** 2))),
        ))
        if classification:
            pair_tp = test_metrics["pair_tp"]
            pair_fp = test_metrics["pair_fp"]
            pair_fn = test_metrics["pair_positive"] - pair_tp
            count = test_metrics["pair_count"]
            test_metrics["pair_accuracy"] = 1 - (pair_fp + pair_fn) / count if count else None
        if target in TOLERANCES:
            test_metrics["tolerance"] = TOLERANCES[target]
            test_metrics["within_tolerance_count"] = int(np.sum(np.abs(y_test - pred) <= TOLERANCES[target]))
            test_metrics["within_tolerance"] = test_metrics["within_tolerance_count"] / len(y_test)
            if target == "stable_angle_deg":
                test_metrics["within_2_deg"] = test_metrics["within_tolerance"]
        if target == STRESS_TARGET:
            threshold = float(np.quantile(y, 0.9))
            high = y_test >= threshold
            shortfall = np.maximum(y_test - pred, 0)
            test_metrics.update(
                underprediction_fraction=float(np.mean(shortfall > 0)),
                max_underprediction_mpa=float(np.max(shortfall)),
                high_stress_threshold_train_p90_mpa=threshold, high_stress_count=int(high.sum()),
                high_stress_mean_shortfall_mpa=float(shortfall[high].mean()) if high.any() else None,
                high_stress_underprediction_count=int(np.sum(high & (shortfall > 0))),
            )
        for row, actual, predicted in zip(test_rows, y_test, pred):
            predictions.append(dict(geom_id=row["geom_id"], state=row["state"], target=target,
                                    actual=float(actual), predicted=float(predicted)))
        report[target] = dict(winner=winner["model"], cv=entries, test=test_metrics,
                              train_count=len(y), test_count=len(y_test))
        print("SELECTED", target, winner["model"], test_metrics, flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "cv_comparison.csv", benchmark)
    write_csv(args.output / "test_predictions.csv", predictions)
    for name, model in models.items():
        joblib.dump(model, args.output / f"{name}.joblib")
    payload = dict(seed=args.seed, sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                   feature_order=[*FEATURES, "state_code"], state_code=STATE_CODE,
                   classification_labels=KNOWN, fold_by_geometry=fold_id, targets=report)
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
