"""Candidate lists of the two Ansys verification kits, taken from the inverse-design results.

Rule (fixed 2026-11-17, before any Ansys result): seed 0 of every method, final query set, the 40 ansys40 queries,
rank 1 only.
  kit 1: forward models trained on 512 geometries, files <method>_final_n20_s0.csv         -> 4 x 40 = 160 candidates
  kit 2: models refitted on 1024 and 1536 geometries, v2/<method>_final_train<n>_n20_s0.csv -> 2 x 4 x 40 = 320
Geometry ids: kit 1 V<method letter><query number> (VG0018 = GP candidate for target TE0018); kit 2
<size letter><method letter><query number>, size W = 1024, X = 1536 (letters in cabd.ansys.kits).
Identical geometries (max |du| < 1e-6 in the unit cube) are solved once: they share the solve_id of the first one.
Kit 2 checks this within the kit; matches with kit 1 are only reported (column same_as_kit1).
A kit-2 method may have fewer than 40 rank-1 candidates (no feasible candidate for a query): nothing to solve, and
the scoring counts it as a failure.

Outputs (float_format %.9g; the kit builder reads these files):
  results/inverse/ansys_verify_candidates.csv, results/inverse/v2/ansys_verify2_candidates.csv

usage: python -m cabd.verify.candidates [--kit 1 2] [--inverse-dir DIR] [--out-dir DIR]
  --inverse-dir: folder with the inverse result csvs (default results/inverse); --out-dir default results/inverse
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cabd.ansys.kits import METHODS, SIZES2
from cabd.config import RESULTS
from cabd.geometry.params import VAR_NAMES as V

INVERSE = RESULTS / "inverse"
SEED = 0
FILES = {1: "ansys_verify_candidates.csv", 2: "v2/ansys_verify2_candidates.csv"}


def _dedupe(c):
    U = c[[f"u_{v}" for v in V]].to_numpy(float)
    c["solve_id"] = c.geom_id
    for i in range(len(c)):
        for j in range(i):
            if np.abs(U[i] - U[j]).max() < 1e-6:
                c.loc[i, "solve_id"] = c.solve_id[j]
                break
    return c


def kit1(inv=INVERSE):
    rows = []
    for m, letter in METHODS.items():
        d = pd.read_csv(inv / f"{m}_final_n20_s{SEED}.csv")
        d = d[(d["rank"] == 1) & d.ansys40]
        assert len(d) == 40 and d.geom_id.is_unique, (m, len(d))
        rows.append(d.assign(method=m, query_id=d.geom_id, geom_id=[f"V{letter}{g[2:]}" for g in d.geom_id]))
    c = _dedupe(pd.concat(rows, ignore_index=True))
    keep = ["solve_id", "geom_id", "method", "query_id", "seed", "t_theta2", "t_u2", "t_logE", "own_obj", "own_Pbm"] + V \
        + [f"u_{v}" for v in V] + [k for k in c.columns if k.startswith("referee")]
    return c[keep]


def kit2(inv=INVERSE, kit1_table=None):
    rows = []
    for n, s in SIZES2.items():
        for m, letter in METHODS.items():
            d = pd.read_csv(inv / "v2" / f"{m}_final_train{n}_n20_s{SEED}.csv")
            d = d[(d["rank"] == 1) & d.ansys40]
            assert len(d) <= 40 and d.geom_id.is_unique, (n, m, len(d))
            if len(d) < 40:   # no feasible candidate for some queries: nothing to solve, a failure when scoring
                print(f"{n} {m}: no rank-1 candidate for {40 - len(d)} queries")
            rows.append(d.assign(method=m, train_size=n, query_id=d.geom_id, geom_id=[f"{s}{letter}{g[2:]}" for g in d.geom_id]))
    c = _dedupe(pd.concat(rows, ignore_index=True))
    old = kit1_table if kit1_table is not None else pd.read_csv(inv / FILES[1])
    U = c[[f"u_{v}" for v in V]].to_numpy(float)
    Uo = old[[f"u_{v}" for v in V]].to_numpy(float)
    c["same_as_kit1"] = [next((old.geom_id[k] for k in range(len(old)) if np.abs(u - Uo[k]).max() < 1e-6), "") for u in U]
    keep = ["solve_id", "geom_id", "train_size", "method", "query_id", "seed", "t_theta2", "t_u2", "t_logE", "own_obj", "own_Pbm"] \
        + V + [f"u_{v}" for v in V] + ["same_as_kit1"] + [k for k in c.columns if k.startswith("referee")]
    return c[keep]


def write(c, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    c.to_csv(path, index=False, float_format="%.9g")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    ap.add_argument("--inverse-dir", type=Path, default=INVERSE)
    ap.add_argument("--out-dir", type=Path, default=INVERSE)
    a = ap.parse_args()
    if 1 in a.kit:
        c = kit1(a.inverse_dir)
        write(c, a.out_dir / FILES[1])
        print(f"kit 1: {len(c)} candidates, {c.solve_id.nunique()} distinct geometries")
    if 2 in a.kit:
        c = kit2(a.inverse_dir, pd.read_csv(a.out_dir / FILES[1]))
        write(c, a.out_dir / FILES[2])
        print(f"kit 2: {len(c)} candidates, {c.solve_id.nunique()} distinct, {int((c.same_as_kit1 != '').sum())} identical to kit 1")


if __name__ == "__main__":
    main()
