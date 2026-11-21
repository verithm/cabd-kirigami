"""Inverse-design queries (fixed 2026-11-16, before any inverse result was seen).

  final: the 106 test geometries with partial bi -> full mono and a real second stable point (partial_class == "bi").
         target = their Ansys (theta2, u2, log10 dE_r).
  ansys40: 40 of the 106 drawn at random with seed 20261116 (the Ansys paired comparison).
  dev:   the bi>mono geometries of fold 0 with a real second stable point (the dev check uses models fitted on folds 1-4).
The table is data/queries.csv (columns: set, geom_id, theta2, u2, logE, ansys40). This step rebuilds it and checks
that it is unchanged; --write overwrites the file.

usage: python -m cabd.evaluate.queries [--write]
"""
import argparse

import numpy as np
import pandas as pd

from .. import data as D

SEED = 20261116


def build():
    t = pd.read_csv(D.TEST_RESULTS)
    p, f = t.partial_class.map(D.state_label), t.full_class.map(D.state_label)
    t = t[(p == "bi") & (f == "mono") & (t.partial_class == "bi")].sort_values("geom_id").reset_index(drop=True)
    final = pd.DataFrame({"set": "final", "geom_id": t.geom_id, "theta2": t.partial_theta2_deg,
                          "u2": t.partial_u2_mm, "logE": np.log10(t.partial_dE_r_mJ)})
    pick = np.sort(np.random.default_rng(SEED).choice(len(final), 40, replace=False))
    final["ansys40"] = False
    final.loc[pick, "ansys40"] = True
    d = D.load()
    fold = D.folds(d)
    dv = d[(fold == 0) & (d.label == "bi>mono") & d.is_reg]
    dev = pd.DataFrame({"set": "dev", "geom_id": dv.geom_id, "theta2": dv.theta2, "u2": dv.u2, "logE": dv.logE,
                        "ansys40": False})
    return pd.concat([final, dev], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="overwrite data/queries.csv")
    args = ap.parse_args()
    q = build()
    print(q.groupby("set").size().to_dict(), "ansys40", int(q.ansys40.sum()))
    print("ansys40 ids:", " ".join(q.geom_id[q.ansys40]))
    if args.write:
        q.to_csv(D.QUERIES, index=False)
        print(f"wrote {D.QUERIES}")
    else:
        same = q.equals(pd.read_csv(D.QUERIES))
        print(f"{D.QUERIES.name} {'unchanged' if same else 'DIFFERS (use --write to overwrite)'}")


if __name__ == "__main__":
    main()
