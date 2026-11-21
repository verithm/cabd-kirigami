"""Learning curve: models trained on the first 512 / 1024 / 1536 training geometries, evaluated on the sealed test set.

Training sets follow the Sobol order: base (train_base, 512) -> + ext1 (PC4-6 + retry, 512) -> + ext2 (PC7-9, 512).
Each (model, size, seed) is fitted exactly like `cabd.fit --train all` (same tuning budget): fit them with
`python -m cabd.fit --models mlp --train n1536 ...`; this step only evaluates the saved models
{model}_n{size}_s{seed} (models/ or artifacts/models/, or --models-dir).
Test metrics (256 test geometries; regression on the 207 with partial bi and dE_r > 0, classes on the 241 labelled):
per output RMSE and nRMSE (normalised by the sd of the base training outputs, fixed for every size so the sizes
compare directly), combined nRMSE with config.Y_WEIGHTS, 3-class balanced accuracy and bi>mono balanced accuracy.
cvae and the referees are not compared here (the size-matched referees and cVAEs serve inverse design v2 only).
The original metrics.csv was overwritten with an empty table on 2026-11-19 (an evaluation run with only the referees,
which this step no longer allows); results/learning_curve/metrics.csv was regenerated with the original code from the
original models, matches the summary table in the original evaluation log to all printed digits, and is reproduced by
this step exactly (tests/test_ml_evaluate.py).

usage: python -m cabd.evaluate.learning_curve [--models mlp gp rom_gp] [--sizes 512 1024 1536] [--seeds 0 1 2 3 4]
writes results/learning_curve/metrics.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import data as D
from ..config import RESULTS, SEEDS, TRAIN_SIZES, X_COLS, Y_COLS, Y_WEIGHTS
from ..models import io

OUT = RESULTS / "learning_curve"


def evaluate(models, sizes, seeds, models_dir=None):
    """Rows of metrics.csv for every saved (model, size, seed)."""
    import torch
    from sklearn.metrics import balanced_accuracy_score

    from .. import models as M
    torch.set_num_threads(1)
    te = D.test_set()
    sd_ref = D.train_set(512).loc[lambda d: d.is_reg, Y_COLS].std().to_numpy(float)
    reg = te.is_reg.to_numpy() & te.logE.notna().to_numpy()
    lab = te.label.notna().to_numpy()
    y = te.loc[lab, "label"].map(D.CLASSES.index).to_numpy()
    rows = []
    for model in [m for m in models if m not in ("cvae", "referee", "referee_nogp")]:
        for n in sizes:
            for s in seeds:
                name = io.model_name(model, f"n{n}", s)
                if models_dir is not None and not all(p.exists() for p in io.files(name, models_dir)):
                    continue
                if models_dir is None and not io.exists(name):
                    continue
                mu, _, P = M.predict(io.load(name, models_dir), te[X_COLS].to_numpy(float))
                Y = te.loc[reg, Y_COLS].to_numpy(float)
                e = mu[reg] - Y
                r = dict(model=model, size=n, seed=s)
                for j, c in enumerate(Y_COLS):
                    r[f"rmse_{c}"] = float(np.sqrt(np.mean(e[:, j] ** 2)))
                    r[f"nrmse_{c}"] = r[f"rmse_{c}"] / sd_ref[j]
                r["combined_nrmse"] = sum(Y_WEIGHTS[c] * r[f"nrmse_{c}"] for c in Y_COLS)
                pred = P[lab].argmax(1)
                r["bal_acc_3class"] = balanced_accuracy_score(y, pred)
                r["bal_acc_bimono"] = balanced_accuracy_score(y == 1, pred == 1)
                rows.append(r)
    print(f"test: {int(reg.sum())} regression targets, {int(lab.sum())} labelled")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mlp", "gp", "rom_gp"])
    ap.add_argument("--sizes", type=int, nargs="+", default=TRAIN_SIZES)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--models-dir", type=Path, help="read the models from this folder only")
    ap.add_argument("--out", type=Path, default=OUT / "metrics.csv")
    args = ap.parse_args()
    df = evaluate(args.models, args.sizes, args.seeds, args.models_dir)
    if df.empty:     # never overwrite the evidence file with an empty table
        raise SystemExit("no learning-curve model found: nothing written")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.5f")
    cols = ["combined_nrmse", "nrmse_theta2", "nrmse_u2", "nrmse_logE", "bal_acc_3class", "bal_acc_bimono"]
    print(df.groupby(["model", "size"])[cols].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
