"""Metrics shared by every method (same rules for all, see the CABD training plan doc)."""
import json

import numpy as np
import pandas as pd

from .config import ALPHA, CLASSES, Y_COLS, Y_WEIGHTS


def class_metrics(y, P):
    """y: labels (strings), P: (n, 3) probabilities in CLASSES order."""
    from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
    yi = np.array([CLASSES.index(v) for v in y])
    pred = P.argmax(1)
    onehot = np.eye(3)[yi]
    bm = CLASSES.index("bi>mono")
    return {
        "balanced_accuracy": balanced_accuracy_score(yi, pred),
        "macro_f1": f1_score(yi, pred, average="macro"),
        "bi>mono_precision": precision_score(yi == bm, pred == bm),
        "bi>mono_recall": recall_score(yi == bm, pred == bm),
        "mono>mono_recall": recall_score(yi == 2, pred == 2),
        "bi>mono_auc": roc_auc_score(yi == bm, P[:, bm]),
        "log_loss": log_loss(yi, np.clip(P, 1e-12, 1), labels=[0, 1, 2]),
        "brier": float(((P - onehot) ** 2).sum(1).mean()),
        "confusion": pd.crosstab(pd.Series(yi, name="true"), pd.Series(pred, name="pred"))
                       .reindex(index=range(3), columns=range(3), fill_value=0).to_numpy().tolist(),
    }


def reg_metrics(Y, M, S, lo, hi, sd_train):
    """Y, M: (n, 3) truth and prediction in model space (logE in log10); S: predicted sd or None; lo/hi: 90% interval.

    nRMSE is the RMSE divided by the sd of the training outputs; combined_nrmse weights the outputs by Y_WEIGHTS."""
    out = {}
    for j, c in enumerate(Y_COLS):
        e = M[:, j] - Y[:, j]
        r = {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
             "r2": float(1 - np.sum(e ** 2) / np.sum((Y[:, j] - Y[:, j].mean()) ** 2)),
             "nrmse": float(np.sqrt(np.mean(e ** 2)) / sd_train[j]),
             "coverage90": float(np.mean((Y[:, j] >= lo[:, j]) & (Y[:, j] <= hi[:, j]))),
             "width90": float(np.mean(hi[:, j] - lo[:, j]))}
        if S is not None:
            r["nll"] = float(np.mean(0.5 * np.log(2 * np.pi * S[:, j] ** 2) + e ** 2 / (2 * S[:, j] ** 2)))
        if c == "logE":
            r["median_rel_err_pct"] = float(np.median(np.abs(10 ** e - 1)) * 100)
        out[c] = r
    out["combined_nrmse"] = float(sum(Y_WEIGHTS[c] * out[c]["nrmse"] for c in Y_COLS))
    return out


def conformal_q(scores, alpha=ALPHA):
    """Split-conformal quantile of nonconformity scores (finite-sample corrected)."""
    n = len(scores)
    k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return float(np.sort(scores)[k - 1])


def evaluate_oof(d, oof):
    """Classification and regression metrics of one seed's out-of-fold predictions (columns as written by cv.py)."""
    oof = oof.set_index("geom_id").reindex(d.geom_id)
    r = d.is_reg.to_numpy()
    Y = d.loc[r, Y_COLS].to_numpy(float)
    M = oof.loc[d.geom_id[r], [f"{c}_mu" for c in Y_COLS]].to_numpy(float)
    S = oof.loc[d.geom_id[r], [f"{c}_sd" for c in Y_COLS]].to_numpy(float)
    S = None if np.isnan(S).all() else S
    lo = oof.loc[d.geom_id[r], [f"{c}_lo" for c in Y_COLS]].to_numpy(float)
    hi = oof.loc[d.geom_id[r], [f"{c}_hi" for c in Y_COLS]].to_numpy(float)
    sd_train = d.loc[r, Y_COLS].std().to_numpy(float)
    ok = d.label.notna().to_numpy()
    P = oof.loc[d.geom_id[ok], [f"P_{c}" for c in CLASSES]].to_numpy(float)
    return {"classification": class_metrics(d.label[ok].to_list(), P),
            "regression": reg_metrics(Y, M, S, lo, hi, sd_train)}


def summarize(tag, name, d, folder):
    """Metrics of every seed whose out-of-fold file {tag}_oof_seed{s}.csv exists in folder -> {tag}_metrics.json."""
    per_seed = {}
    for p in sorted(folder.glob(f"{tag}_oof_seed*.csv")):
        s = int(p.stem.split("seed")[1])
        per_seed[s] = m = evaluate_oof(d, pd.read_csv(p))
        print(f"{name} seed {s}: balanced acc {m['classification']['balanced_accuracy']:.3f}, "
              f"combined nRMSE {m['regression']['combined_nrmse']:.3f}", flush=True)
    save_json(folder / f"{tag}_metrics.json", {"method": name, "seeds": per_seed})
    return per_seed


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
