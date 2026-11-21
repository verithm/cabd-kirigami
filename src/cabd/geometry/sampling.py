"""Sobol parameter tables: train core 512 and test 256 (base), the extra training batches (ext1, ext2), and the
parameter tables of the inverse-design verification kits.

Geometry model: cabd.geometry.params (param_geom_v3: 8 variables incl. the common fillet ratio rho, base 25.4 mm
fixed, t_w = 3 fixed, alpha derived). Distribution: uniform in all 8 variables (scrambled Sobol). Filter:
params.check_constraints (incl. the 128k mesh-node limit and the 0.1 mm minimum feature).

Base tables (seed A = 20261101 for train, seed B = 20261102 for test; 20261103 reserved for an adaptive pool):
  random_base2 with 2x oversampling, keep the first n points (in Sobol order) that pass the validity checks.
  The test set is split into stage 1 (first half of the Sobol indices) and stage 2.
Extra training batches (stage k -> params_train_ext<k>.csv): the next n points of the SAME training sequence (seed A),
  continuing after the last Sobol index used by params_train*.csv. The step stops if the seed no longer reproduces the
  existing tables (|du| < 1e-5). geom_id = TR<sobol_index>, so a point rejected by the checks leaves a gap in the
  numbers (ext2: Sobol index 1269 rejected, so ext2 runs TR1024..TR1536 without TR1269). The original ext1 script
  numbered rows consecutively from TR0512 instead; ext1 had no rejected point, so both rules give the same ids
  (the equivalence test checks both tables byte for byte).
Verification tables (params_verify*.csv): one row per distinct inverse-design candidate, same columns, role "verify",
  no Sobol seed/index; written by the verification kit builders (cabd.ansys.kit).

All tables are written with float_format %.6g (the STEP files and the kits are built from these rounded values).

usage:
  python -m cabd.geometry.sampling base [--out DIR] [--preview]       params_train.csv, params_test.csv, params_meta.json
  python -m cabd.geometry.sampling ext --stage 2 [--n 512] [--out DIR] [--tables DIR]
default --out is build/sampling (the committed tables in data/sampling are never overwritten unless --out points there
and --force is given).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import qmc

from cabd.config import BUILD, SAMPLING
from cabd.geometry import params as pg

V = pg.VAR_NAMES
LO = {v: pg.RANGES[v][0] for v in V}
HI = {v: pg.RANGES[v][1] for v in V}
SEEDS = dict(train=20261101, test=20261102, candidates=20261103)  # candidates: reserved for an adaptive pool
ROLES = [("train", 512), ("test", 256)]
FLOAT_FORMAT = "%.6g"
COLUMNS = (["geom_id", "role", "sobol_seed", "sobol_index"] + V + [f"u_{v}" for v in V]
           + ["alpha_deg", "R_slot", "R_low", "H", "Z_o", "W", "x_fixed_face", "x_displaced_face", "area_mm2",
              "est_nodes", "test_stage"])


def point(u):
    """Design dict (REF supplies the fixed t_w) of one point of the unit cube."""
    return dict(pg.REF, **{v: LO[v] + u[i] * (HI[v] - LO[v]) for i, v in enumerate(V)})


def row(geom_id, role, p, u, sobol_seed="", sobol_index="", test_stage=""):
    """One table row: identity, design values, unit-cube coordinates and the derived quantities."""
    d = pg.derived(p)
    return dict(geom_id=geom_id, role=role, sobol_seed=sobol_seed, sobol_index=sobol_index,
                **{v: p[v] for v in V}, **{f"u_{v}": u[i] for i, v in enumerate(V)},
                alpha_deg=d["alpha"], R_slot=d["R_slot"], R_low=d["R_low"], H=d["H"], Z_o=d["Z_o"], W=d["W"],
                x_fixed_face=d["x_W"], x_displaced_face=d["x_D"],
                area_mm2=pg.total_area(p), est_nodes=pg.est_nodes(p), test_stage=test_stage)


def valid(p):
    return all(pg.check_constraints(p).values())


def draw(role, n):
    """First n valid points (Sobol order) of 2^(log2 n + 1) drawn points; returns (table, rejected, drawn)."""
    m = int(np.log2(n)) + 1
    U = qmc.Sobol(d=len(V), scramble=True, seed=SEEDS[role]).random_base2(m)
    rows = []
    for idx, x in enumerate(U):
        p = point(x)
        if not valid(p):
            continue
        stage = (1 if idx < n // 2 else 2) if role == "test" else ""
        rows.append(row(f"{role[:2].upper()}{len(rows):04d}", role, p, x, SEEDS[role], idx, stage))
        if len(rows) == n:
            break
    rejected = rows[-1]["sobol_index"] + 1 - n
    return pd.DataFrame(rows), rejected, 2 ** m


def base_tables():
    """(tables by role, meta dict) of the base train/test sets, exactly as written to params_meta.json."""
    meta = dict(seeds=SEEDS, scipy=scipy.__version__, ranges={v: [LO[v], HI[v]] for v in V},
                geometry_model="param_geom_v3 (base 25.4 mm fixed, t_w = 3 fixed, alpha derived, rho variable)",
                distribution="uniform for all 8 variables",
                filter="param_geom_v3.check_constraints (incl. 128k nodes, min feature 0.1 mm)", roles={})
    tables = {}
    for role, n in ROLES:
        df, rej, drawn = draw(role, n)
        tables[role] = df
        meta["roles"][role] = dict(n=n, drawn=drawn, rejected_by_filter=rej)
    all_df = pd.concat(tables.values())
    # balance check: the first 2^k points of each role fill all 2^k 1-D bins in every variable
    for role, n in ROLES:
        u = all_df[all_df.role == role][[f"u_{v}" for v in V]].to_numpy()
        meta["roles"][role]["1d_balanced"] = bool(all(len(set((u[:n, j] * n).astype(int))) == n for j in range(8)))
    # closest train-test pair in normalised space
    ut = tables["train"][[f"u_{v}" for v in V]].to_numpy()
    us = tables["test"][[f"u_{v}" for v in V]].to_numpy()
    dmin = np.sqrt(((us[:, None] - ut[None]) ** 2).sum(-1)).min(1)
    meta["test_to_train_min_dist"] = dict(min=float(dmin.min()), median=float(np.median(dmin)))
    meta["area_mm2"] = dict(min=float(all_df.area_mm2.min()), max=float(all_df.area_mm2.max()))
    meta["est_nodes_max"] = float(all_df.est_nodes.max())
    meta["alpha_deg"] = dict(min=float(all_df.alpha_deg.min()), max=float(all_df.alpha_deg.max()),
                             frac_35_55=float(all_df.alpha_deg.between(35, 55).mean()))
    return tables, meta


def earlier_tables(tables_dir, stage):
    files = [tables_dir / "params_train.csv"] + [tables_dir / f"params_train_ext{k}.csv" for k in range(1, stage)]
    missing = [p.name for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"earlier tables missing: {missing}")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def ext_table(old, n=512):
    """Next n valid points of the training sequence after the last index in `old` (the earlier train tables)."""
    seed = SEEDS["train"]
    start = int(old.sobol_index.max()) + 1
    m = int(np.ceil(np.log2(start + 2 * n)))
    U = qmc.Sobol(d=len(V), scramble=True, seed=seed).random_base2(m)
    diff = np.abs(U[old.sobol_index.to_numpy()] - old[[f"u_{v}" for v in V]].to_numpy()).max()
    assert diff < 1e-5, f"sequence not reproduced: {diff}"
    rows, rejected = [], []
    for idx in range(start, len(U)):
        p = point(U[idx])
        if not valid(p):
            rejected.append(idx)
            continue
        rows.append(row(f"TR{idx:04d}", "train", p, U[idx], seed, idx))
        if len(rows) == n:
            break
    df = pd.DataFrame(rows)
    assert len(df) == n and list(df.columns) == list(old.columns) and not set(df.geom_id) & set(old.geom_id)
    return df, dict(start=start, rejected=rejected, reproduced_within=float(diff))


def table_from_candidates(c):
    """Parameter table (sampling-table format, role verify) of the distinct candidates (column solve_id) in c."""
    rows = []
    for _, r in c.drop_duplicates("solve_id").iterrows():
        p = dict(pg.REF, **{v: float(r[v]) for v in V})
        assert valid(p), r.solve_id
        for v in V:
            lo, hi = pg.RANGES[v]
            assert lo - 1e-9 <= p[v] <= hi + 1e-9, (r.solve_id, v, p[v])
        rows.append(row(r.solve_id, "verify", p, [r[f"u_{v}"] for v in V]))
    return pd.DataFrame(rows)


def write(df, path, force=False):
    if path.exists() and not force:
        raise SystemExit(f"{path} already exists (use --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=FLOAT_FORMAT, lineterminator=CRLF)


CRLF = "\r\n"   # line ends of the original tables (written on Windows); explicit so every platform writes the same bytes


def preview(df, path):
    """Preview figure: the first 16 training cross-sections (same scale) and 2-D projections train vs test."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(3, 8, height_ratios=[1.15, 1.15, 1.0], hspace=0.35, wspace=0.15)
    tr = df[df.role == "train"].reset_index(drop=True)
    for k in range(16):
        ax = fig.add_subplot(gs[k // 8, k % 8])
        r = tr.iloc[k]
        p = dict(pg.REF, **{v: r[v] for v in V})
        for kind, P in pg.bodies(p, n=24).values():
            ax.add_patch(Polygon(P, closed=True, fc="#c9d3df" if kind == "frame" else "#f2b48a", ec="#55606e", lw=0.5))
        ax.set_xlim(-16.5, 1); ax.set_ylim(-15.5, 15.5); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{r.geom_id}  L={r.L:.1f}  α={r.alpha_deg:.0f}° (유도)", fontsize=9)
    pairs = [("t_h", "L_h"), ("t_a", "L_a"), ("L", "t_f"), ("rho", "c")]
    te = df[df.role == "test"]
    for j, (a, b) in enumerate(pairs):
        ax = fig.add_subplot(gs[2, 2 * j:2 * j + 2])
        ax.scatter(tr[a], tr[b], s=7, c="#1f5fbf", label="학습 512 (seed A)", alpha=0.75)
        ax.scatter(te[a], te[b], s=9, c="#d62728", marker="x", label="시험 256 (seed B)", alpha=0.8, lw=0.8)
        ax.set_xlabel(a); ax.set_ylabel(b); ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=9, loc="upper right")
    fig.suptitle("Sobol 파라미터 표 미리보기 — 위: 학습 세트 첫 16개 형상(같은 축척), 아래: 변수 쌍 투영", fontsize=15,
                 fontweight="bold", x=0.02, ha="left")
    fig.savefig(path, dpi=130, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["base", "ext"])
    ap.add_argument("--stage", type=int, help="ext: batch number k -> params_train_ext<k>.csv")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", type=Path, default=BUILD / "sampling")
    ap.add_argument("--tables", type=Path, default=SAMPLING, help="ext: folder with the earlier train tables")
    ap.add_argument("--preview", action="store_true", help="base: also save params_preview.png")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.what == "base":
        tables, meta = base_tables()
        for role, df in tables.items():
            write(df, a.out / f"params_{role}.csv", a.force)
        (a.out / "params_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8", newline=CRLF)
        print(json.dumps(meta, indent=2))
        if a.preview:
            preview(pd.concat(tables.values()), a.out / "params_preview.png")
    else:
        if not a.stage:
            raise SystemExit("ext needs --stage")
        old = earlier_tables(a.tables, a.stage)
        df, info = ext_table(old, a.n)
        out = a.out / f"params_train_ext{a.stage}.csv"
        write(df, out, a.force)
        print(f"saved {out}: {len(df)} rows, {df.geom_id.iloc[0]}..{df.geom_id.iloc[-1]}, sobol {info['start']}.."
              f"{df.sobol_index.max()}, rejected {info['rejected']}, earlier {len(old)} points reproduced within "
              f"{info['reproduced_within']:.1e}, est_nodes max {df.est_nodes.max():.0f}")


if __name__ == "__main__":
    main()
