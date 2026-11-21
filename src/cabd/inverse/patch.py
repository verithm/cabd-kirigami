"""Rerun single queries of an inverse-design result file and replace their rows.

Why: the per-query cap (core.TIME_CAP) is wall-clock time. During the sequential v2 rerun of 2026-11-19 the laptop
went into Windows Modern Standby (lid closed); the standby time counted as query time, so some queries stopped after
a few starts (e.g. ROM+GP 1024 TE0211: 2 of 20 starts, 0 valid). Those 6 queries were rerun one at a time with this
tool on 2026-11-21 (rom_gp:n1024:TE0194 rom_gp:n1024:TE0211 rom_gp:n1536:TE0102 rom_gp:n1536:TE0186
gp:n1536:TE0221 gp:n1536:TE0224); the v2 files in results/inverse/v2 contain the rerun rows.
Same rules as cabd.inverse.run (one_query unchanged), run one at a time in this process. Before the first change
each touched file is copied once to build/inverse_patch_backup/ (the original tool kept that copy next to the results
as v2/standby_hit_backup/). The referee columns of the new rows are empty until cabd.inverse.score is run again.

usage: python -m cabd.inverse.patch METHOD:TRAIN:GEOM_ID [...] [--seed 0] [--query-set final] [--starts 20]
       e.g. python -m cabd.inverse.patch rom_gp:n1024:TE0194 gp:n1536:TE0221     (TRAIN: all, n1024, n1536)
"""
import argparse
import shutil
import time
from pathlib import Path

import pandas as pd

from .. import config, data
from . import core
from .run import OUT, one_query, out_path

BACKUP = config.BUILD / "inverse_patch_backup"


def patch(method, train, geom_id, seed=0, qset="final", n_start=core.N_START, out=OUT, backup=BACKUP):
    if train.isdigit():
        train = f"n{train}"                      # the original tool's form rom_gp:1024:TE0194
    q = data.queries(qset).set_index("geom_id", drop=False)
    path = out_path(method, train, qset, seed, n_start, out)
    bdir = Path(backup) / path.parent.relative_to(Path(out))
    bdir.mkdir(parents=True, exist_ok=True)
    if not (bdir / path.name).exists():
        shutil.copy2(path, bdir / path.name)
    t0 = time.strftime("%H:%M:%S")
    rows = one_query(method, train, qset, seed, n_start, q.loc[geom_id])
    df = pd.read_csv(path)
    old = df[df.geom_id == geom_id].iloc[0]
    new = pd.DataFrame(rows)
    df = pd.concat([df[df.geom_id != geom_id], new], ignore_index=True)
    df["_o"] = df.geom_id.map({k: i for i, k in enumerate(q.index)})
    df = df.sort_values(["_o", "rank"], kind="stable").drop(columns="_o")
    df.to_csv(path, index=False)
    r = new.iloc[0]
    print(f"{method} {train} {geom_id} [{t0}-{time.strftime('%H:%M:%S')}]: old {old.time:.0f} s / {old.starts_done} "
          f"starts / {old.starts_valid} valid -> new {r.time:.0f} s / {r.starts_done} starts / {r.starts_valid} valid, "
          f"rank-1 {'yes' if (new['rank'] >= 1).any() else 'NO'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="+", help="METHOD:TRAIN:GEOM_ID")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--query-set", default="final", choices=["final", "dev"])
    ap.add_argument("--starts", type=int, default=core.N_START)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    for j in a.jobs:
        m, tr, g = j.split(":")
        patch(m, tr, g, a.seed, a.query_set, a.starts, a.out)


if __name__ == "__main__":
    main()
