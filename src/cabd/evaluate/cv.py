"""Out-of-fold cross-validation of the forward models on the 512 base training geometries (shared 5 folds).

For every (seed, fold) the model is fitted on the other four folds exactly as cabd.models.<kind>.fit does on a full
training set, and predicts the held-out fold. The 90 % intervals use only the training folds:
  gp, rom_gp      normalized split-conformal, inner 5-fold with the outer fit's kernel fixed (gp.conformal_scale)
  mlp             normalized split-conformal on the inner 20 % hold-out of the tuning split (mlp.fit_kind)
  referee(_nogp)  absolute-residual split-conformal on the stacked inner out-of-fold predictions (Stacked.q)
rom_gp also fits the ROM factors on the training folds only (written to rom_params.csv).

Outputs in --out (default results/forward): {tag}_oof_seed{s}.csv per seed (one row per geometry: geom_id, per output
mu / sd / lo / hi (+ q: gp, mlp; + rom: rom_gp), P_<class>, fold) and {tag}_metrics.json (cabd.metrics.summarize).
  tags: gp_forward, mlp_forward, rom_gp_forward, referee, referee_nogp
Per-fold run times of the original runs (5-15 parallel jobs on 16 cores): mlp 4-6 min, referee_nogp 15-20 min,
gp 15-55 min, rom_gp ~70 min, referee (with the GP member) 3-8 h.
Reproducibility: the GP fits (gp, rom_gp, the referee's gpx member) depend on the OpenBLAS thread count at the 1e-4
level (restarted L-BFGS on the marginal likelihood). joblib's loky workers get cpu_count // n_jobs BLAS threads, so
rerun with the default --jobs on a 16-core machine to get the original thread counts (ORIG_THREADS: gp 3, every other
model 1). Checked by refitting one (seed 0, fold 0) slice: gp matches the stored OOF exactly only with 3 threads,
rom_gp only with 1 thread (3 threads: max |diff| 2.4e-5), so the stored rom_gp run used >= 9 jobs, not 5.

usage: python -m cabd.evaluate.cv --model gp [--seeds 0 1 2 3 4] [--folds 0 1 2 3 4] [--jobs 5] [--out DIR]
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .. import data as D
from ..config import N_FOLDS, RESULTS, SEEDS, Y_COLS
from ..metrics import summarize

TAG = {"gp": "gp_forward", "mlp": "mlp_forward", "rom_gp": "rom_gp_forward", "referee": "referee",
       "referee_nogp": "referee_nogp"}
NAME = {"gp": "GP", "mlp": "MLP ensemble", "rom_gp": "ROM + GP", "referee": "referee (poly+gbm+gpx)",
        "referee_nogp": "referee_nogp (poly+gbm)"}
JOBS = {"gp": 5, "mlp": 15, "rom_gp": 16, "referee": 15, "referee_nogp": 15}   # default --jobs (16 cores)
ORIG_THREADS = {"gp": 3, "mlp": 1, "rom_gp": 1, "referee": 1, "referee_nogp": 1}   # BLAS threads of the stored runs
OUT = RESULTS / "forward"


def _split(d, fold, k):
    out = pd.DataFrame(index=np.where(fold == k)[0])
    out["geom_id"] = d.geom_id.iloc[out.index].to_numpy()
    reg_te = np.where(d.is_reg & (fold == k))[0]
    ok = pd.notna(d.label.to_numpy(object))
    te = np.where(ok & (fold == k))[0]
    return out, reg_te, te, d[fold != k].reset_index(drop=True)


def _put_P(out, te, P):
    out.loc[te, "P_bi>bi"], out.loc[te, "P_bi>mono"], out.loc[te, "P_mono>mono"] = P[:, 0], P[:, 1], P[:, 2]


def fold_gp(d, fold, k, seed):
    from ..models import gp
    out, reg_te, te, tr = _split(d, fold, k)
    m = gp.fit(tr, seed)
    X, Xtr = D.scale_x(d), D.scale_x(tr)
    r = tr.is_reg.to_numpy()
    for c, g in zip(Y_COLS, m["reg"]):
        q = gp.conformal_scale(Xtr[r], tr.loc[r, c].to_numpy(float), g.kernel_, seed)
        mu, sd = g.predict(X[reg_te], return_std=True)
        out.loc[reg_te, f"{c}_mu"], out.loc[reg_te, f"{c}_sd"] = mu, sd
        out.loc[reg_te, f"{c}_lo"], out.loc[reg_te, f"{c}_hi"] = mu - q * sd, mu + q * sd
        out.loc[reg_te, f"{c}_q"] = q
    _put_P(out, te, gp.three_class(gp.p_true(m["c1"], X[te]), gp.p_true(m["c2"], X[te])))
    return out, None


def fold_mlp(d, fold, k, seed):
    from ..models import mlp
    out, reg_te, te, tr = _split(d, fold, k)
    m = mlp.fit(tr, seed)
    X = (D.scale_x(d) * 2 - 1).astype(np.float32)
    mu, s = mlp.ensemble("reg", m["reg"], X[reg_te])
    for j, c in enumerate(Y_COLS):
        m_, s_ = mu[:, j] * m["sd"][j] + m["mean"][j], s[:, j] * m["sd"][j]
        out.loc[reg_te, f"{c}_mu"], out.loc[reg_te, f"{c}_sd"] = m_, s_
        out.loc[reg_te, f"{c}_lo"], out.loc[reg_te, f"{c}_hi"] = m_ - m["q"][j] * s_, m_ + m["q"][j] * s_
        out.loc[reg_te, f"{c}_q"] = m["q"][j]
    _put_P(out, te, mlp.ensemble("cls", m["cls"], X[te]))
    return out, dict(cfg_r=m["cfg_r"], cfg_c=m["cfg_c"])


def fold_rom_gp(d, fold, k, seed):
    from ..models import gp, rom, rom_gp
    out, reg_te, te, tr = _split(d, fold, k)
    th = rom.fit_theta(tr)
    m = rom_gp.fit(tr, seed, th["theta"])
    X = D.scale_x(d)
    f = rom.features(d, m["theta"])
    XR, X1, X2 = rom_gp.inputs(m, X, f=f)
    B = rom_gp.rom_inputs(f)
    reg_tr = np.where(d.is_reg & (fold != k))[0]
    for j, (c, g) in enumerate(zip(Y_COLS, m["reg"])):
        q = gp.conformal_scale(XR[reg_tr], d[c].to_numpy(float)[reg_tr], g.kernel_, seed)
        mu, sd = g.predict(XR[reg_te], return_std=True)
        out.loc[reg_te, f"{c}_mu"], out.loc[reg_te, f"{c}_sd"] = mu, sd
        out.loc[reg_te, f"{c}_lo"], out.loc[reg_te, f"{c}_hi"] = mu - q * sd, mu + q * sd
        out.loc[reg_te, f"{c}_rom"] = B[reg_te, j]
    _put_P(out, te, gp.three_class(gp.p_true(m["c1"], X1[te]), gp.p_true(m["c2"], X2[te])))
    return out, dict(seed=seed, fold=k, rms=th["rms"], **{f"theta{i}": t for i, t in enumerate(th["theta"])})


def fold_referee(d, fold, k, seed, kind):
    from ..models import referee
    from ..models.gp import three_class
    out, reg_te, te, tr = _split(d, fold, k)
    m = referee.fit(tr, seed, referee.MEMBERS[kind])
    X = D.scale_x(d) * 2 - 1
    for c, s in zip(Y_COLS, m["reg"]):
        mu = s.predict(X[reg_te])
        out.loc[reg_te, f"{c}_mu"], out.loc[reg_te, f"{c}_sd"] = mu, np.nan
        out.loc[reg_te, f"{c}_lo"], out.loc[reg_te, f"{c}_hi"] = mu - s.q, mu + s.q
    _put_P(out, te, three_class(m["c1"].predict(X[te]), m["c2"].predict(X[te])))
    w = {c: dict(zip(s.members, np.round(s.w, 3).tolist())) for c, s in zip(Y_COLS, m["reg"])}
    w["stage1"], w["stage2"] = (dict(zip(m[s].members, np.round(m[s].w, 3).tolist())) for s in ("c1", "c2"))
    return out, w


def run_fold(kind, d, fold, k, seed):
    """Out-of-fold predictions of fold k (DataFrame indexed by row of d) and fold info (ROM params, configs, weights)."""
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    if kind in ("referee", "referee_nogp"):
        out, info = fold_referee(d, fold, k, seed, kind)
    else:
        out, info = {"gp": fold_gp, "mlp": fold_mlp, "rom_gp": fold_rom_gp}[kind](d, fold, k, seed)
    out["fold"] = k
    print(f"  {kind} seed {seed} fold {k}: {time.time() - t0:.0f} s | {info}", flush=True)
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(TAG))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)),
                    help="a subset of folds writes {tag}_oof_seed{s}_folds{...}.csv and no metrics")
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    from joblib import Parallel, delayed
    d = D.load()
    fold = D.folds(d)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = TAG[args.model]
    jobs = [(s, k) for s in args.seeds for k in args.folds]
    res = Parallel(n_jobs=args.jobs or JOBS[args.model])(delayed(run_fold)(args.model, d, fold, k, s) for s, k in jobs)
    full = sorted(args.folds) == list(range(N_FOLDS))
    suffix = "" if full else "_folds" + "".join(map(str, sorted(args.folds)))
    if args.model == "rom_gp":
        pd.DataFrame([r[1] for r in res]).to_csv(args.out / f"rom_params{suffix}.csv", index=False)
    for s in args.seeds:
        oof = pd.concat([r[0] for r, (s2, _) in zip(res, jobs) if s2 == s]).sort_index()
        oof.to_csv(args.out / f"{tag}_oof_seed{s}{suffix}.csv", index=False)
    if full:
        summarize(tag, NAME[args.model], d, args.out)


if __name__ == "__main__":
    main()
