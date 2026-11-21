"""Referee scoring of the inverse-design result files (so that no method grades itself).

Rule (2026-11-16): success = max_j |mu_ref_j - t_j| / tau_j <= 1 and P_ref(bi>mono) >= P_MIN, for each candidate.
Two referees: referee (poly + LightGBM + GP) and referee_nogp (poly + LightGBM; checks whether GP-like methods are
favored by the GP member). Columns added per referee r: {r}_ok, {r}_theta2, {r}_u2, {r}_logE, {r}_maxerr, {r}_Pbm
(empty for rank-0 rows, which count as failures).
Which referees:
  v1 (--train all): the referees fitted on the 512 base geometries, seed 0 (referee_all_s0, referee_nogp_all_s0),
      for every file of the query set (results/inverse/*_{set}_*.csv), whatever the method's seed
  v2 (--train n1024 / n1536): the referees fitted on the same training size as the candidate's models
      (referee_n{size}_s0, referee_nogp_n{size}_s0); adds the column referee_train_size
The files are rewritten in place (the other columns unchanged).

usage: python -m cabd.inverse.score [--train all] [--query-set final]
       python -m cabd.inverse.score --train n1024 n1536
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import X_COLS, Y_COLS
from .core import P_MIN, TAU
from .run import OUT

REFEREES = ("referee", "referee_nogp")


def referee_names(train):
    tr = "all" if train == "all" else train
    return {r: f"{r}_{tr}_s0" for r in REFEREES}


def load_referees(train):
    """{referee: predictor} for the training set's rule (see the module docstring)."""
    from ..models import load, predictor
    return {r: predictor(load(name)) for r, name in referee_names(train).items()}


def add_referee_columns(df, refs):
    """Referee columns for every candidate row of df (rank > 0). Modifies and returns df."""
    has = df["rank"] > 0
    X = df.loc[has, [f"u_{c}" for c in X_COLS]].to_numpy(float)
    T = df.loc[has, ["t_theta2", "t_u2", "t_logE"]].to_numpy(float)
    for r, pred in refs.items():
        df[f"{r}_ok"] = False
        if has.any():
            mu, _, pbm = pred(X)
            err = np.abs(mu - T) / TAU
            for j, c in enumerate(Y_COLS):
                df.loc[has, f"{r}_{c}"] = mu[:, j]
            df.loc[has, f"{r}_maxerr"] = err.max(1)
            df.loc[has, f"{r}_Pbm"] = pbm
            df.loc[has, f"{r}_ok"] = (err.max(1) <= 1) & (pbm >= P_MIN)
    return df


def score_frame(df, train, refs=None):
    """Referee columns for one result table of the given training set ('all', 'n1024', 'n1536')."""
    df = add_referee_columns(df, refs or load_referees(train))
    if train != "all":
        df["referee_train_size"] = int(train[1:])
    return df


def files(train, qset="final", out=OUT):
    out = Path(out)
    if train == "all":
        return sorted(out.glob(f"*_{qset}_*.csv"))
    return sorted((out / "v2").glob(f"*_final_train{train[1:]}_n*_s0.csv"))


def score(train, qset="final", out=OUT):
    refs = load_referees(train)
    for path in files(train, qset, out):
        df = score_frame(pd.read_csv(path), train, refs)
        df.to_csv(path, index=False)
        top = df[df["rank"] <= 1]
        print(f"{path.name}: top-1 success referee {top.referee_ok.mean():.3f}, "
              f"no-GP referee {top.referee_nogp_ok.mean():.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", default=["all"], choices=["all", "n1024", "n1536"])
    ap.add_argument("--query-set", default="final", choices=["final", "dev"])
    ap.add_argument("--out", type=Path, default=OUT, help="result folder (default results/inverse)")
    a = ap.parse_args()
    for tr in a.train:
        score(tr, a.query_set, a.out)


if __name__ == "__main__":
    main()
