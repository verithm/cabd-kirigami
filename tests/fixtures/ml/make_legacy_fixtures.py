"""Baseline fixtures from the ORIGINAL code and models (needs the original project folder, CABD_LEGACY).

Run once; the tests then compare the repo code against these files without the original folder.
  legacy_pred/{name}.npz   for every legacy model (ml/results/models and ml/results/learning_curve/models), on the
                           256 test geometries (data/CABD_test_results.csv design variables):
      forward models  mu_lc, P_lc         legacy learning_curve.predict (physical inputs; not for referees)
                      mu_inv, sd_inv, pbm_inv   legacy inverse.predictor on scale_x(X) (sd = 0 for referees)
                      p1, p2              referees only: the two stacked stages
      cvae            samples (3 targets x 200 x 8), cand_{i} (candidates with the legacy inverse.feasible)
      rom_theta       theta, rms
  The legacy code only reads its files; torch runs single-threaded so float32 results are reproducible.

usage: python -X utf8 tests/fixtures/ml/make_legacy_fixtures.py [--only NAME ...]   (CABD_LEGACY set)
"""
import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / "legacy_pred"
N_CVAE_TARGETS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    legacy = Path(os.environ["CABD_LEGACY"])
    sys.path.insert(0, str(legacy / "ml"))
    import joblib
    import torch
    torch.set_num_threads(1)
    import common as C
    import inverse as I
    import learning_curve as L

    te = pd.read_csv(REPO / "data" / "CABD_test_results.csv")
    X = te[C.X_COLS].to_numpy(float)
    U = C.scale_x(pd.DataFrame(X, columns=C.X_COLS))
    q = pd.read_csv(REPO / "data" / "queries.csv")
    targets = q[q.set == "final"][C.Y_COLS].to_numpy(float)[:N_CVAE_TARGETS]
    OUT.mkdir(exist_ok=True)
    res = legacy / "ml" / "results"
    for src in sorted((res / "models").glob("*.joblib")) + sorted((res / "learning_curve" / "models").glob("*.joblib")):
        name = src.stem
        if args.only and name not in args.only:
            continue
        m = joblib.load(src)
        kind = re.match(r"(.+?)_(all|fold0|n\d+)(_s\d+)?$", name).group(1)
        out = {}
        if kind == "rom_theta":
            out = {"theta": np.asarray(m["theta"]), "rms": np.array(m["rms"])}
        elif kind == "cvae":
            seed = m.seed
            out["samples"] = np.stack([m.sample(t, seed=seed) for t in targets])
            for i, t in enumerate(targets):
                out[f"cand_{i}"] = m.candidates(t, I.feasible, seed=seed)
        else:
            if kind not in ("referee", "referee_nogp"):
                mu, P = L.predict(kind, m, X)
                out["mu_lc"], out["P_lc"] = mu, P
            else:
                Xs = U * 2 - 1
                out["p1"], out["p2"] = m["c1"].predict(Xs), m["c2"].predict(Xs)
            mu, sd, pbm = I.predictor(kind, m)(U)
            out["mu_inv"], out["sd_inv"], out["pbm_inv"] = mu, sd, pbm
        np.savez_compressed(OUT / f"{name}.npz", **out)
        print(name, {k: v.shape for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    main()
