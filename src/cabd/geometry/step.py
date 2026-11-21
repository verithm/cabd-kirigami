"""STEP export of the parameter tables: one STEP file per row (6 bodies, exact arcs, extruded 6 mm in +y).

Needs cadquery, which lives in a separate environment: run this module with that python, or call `run_cad(...)`
from any environment (it starts the python given by the environment variable CABD_CAD_PYTHON, default the
running interpreter, with this package on PYTHONPATH).

Geometry comes from cabd.geometry.params.bodies(); arcs are rebuilt as true circular arcs (threePointArc), so the STEP
matches the original CAD convention (section in the X-Z plane, y = 0..6 mm, units mm). Each body is a named child of
a cadquery Assembly: frame, arm_upper, arm_lower (stiff), hinge_tip_upper, hinge_tip_lower, hinge_apex (soft).
step_build_log.csv records volume / 6 mm against the fine-polygon section area (n = 400 points per arc) per file.
The STEP writer embeds a timestamp, so files are compared by geometry (`compare`), never by bytes.

verify: re-import every STEP file and check 6 valid solids, total volume = 6 mm x section area, bounding box (fixed
face x, base z = +-12.7, y = 0..6) and no overlap between bodies. Tolerances reflect the 6-significant-digit CSV
rounding (~1e-5 mm).

usage (CAD python, PYTHONPATH=src):
  python -m cabd.geometry.step build  <out_dir> [table.csv ...] [--limit N]     (tables default: params_train, params_test)
  python -m cabd.geometry.step verify <step_dir> [table.csv ...]
  python -m cabd.geometry.step compare <dir_a> <dir_b> [geom_id ...]            area, bounding box, face count per file
  python -m cabd.geometry.step fingerprint <file.step ...>                       json: solids, volumes, areas, faces, bbox
Table names without a folder are looked up in data/sampling.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cabd.config import CAD_PYTHON, SAMPLING
from cabd.geometry import params as pg

DEPTH = 6.0
TOL = 1e-9
DEFAULT_TABLES = ["params_train.csv", "params_test.csv"]


def table_path(name):
    p = Path(name)
    return p if p.parent != Path(".") or p.exists() else SAMPLING / p


def run_cad(*args, check=True):
    """Run `python -m cabd.geometry.step <args>` with the CAD python; returns the CompletedProcess."""
    python = CAD_PYTHON or sys.executable
    src = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([python, "-X", "utf8", "-m", "cabd.geometry.step", *map(str, args)], capture_output=True,
                       text=True, encoding="utf-8", env=env)
    if check and r.returncode != 0:
        raise RuntimeError(f"STEP step failed ({python}):\n{r.stdout}\n{r.stderr[-3000:]}")
    return r


# ----------------------------------------------------------------------------------------------------- build
class _ArcLog:
    """While active, params.arc is computed with 3 points per arc and every arc is logged (to find arc segments);
    `fine()` switches to 400 points per arc for the reference area."""

    def __init__(self):
        self.arcs, self._orig = [], pg.arc

    def _logged(self, c, R, a0, a1, n=16):
        P = self._orig(c, R, a0, a1, 3)
        self.arcs.append(P.copy())
        return P

    def __enter__(self):
        pg.arc = self._logged
        return self

    def __exit__(self, *exc):
        pg.arc = self._orig

    def fine_area(self, p):
        pg.arc = lambda c, R, a0, a1, n=16: self._orig(c, R, a0, a1, 400)
        try:
            return pg.total_area(p)
        finally:
            pg.arc = self._logged


def primitives(P, arcs):
    """Split a polygon (3-point arcs embedded) into ('line', a, b) / ('arc', a, m, b) segments."""
    keep = [P[0]]
    for q in P[1:]:
        if np.linalg.norm(q - keep[-1]) > TOL:
            keep.append(q)
    if np.linalg.norm(keep[-1] - keep[0]) < TOL:
        keep.pop()
    P = np.array(keep)
    keys = []
    for A in arcs:
        for B in (A, A[::-1], A * [1, -1], (A * [1, -1])[::-1]):
            keys.append(B)
    segs, i, n = [], 0, len(P)
    while i < n:
        a, m, b = P[i], P[(i + 1) % n], P[(i + 2) % n]
        if any(np.allclose([a, m, b], K, atol=1e-7) for K in keys):
            segs.append(("arc", a, m, b)); i += 2
        else:
            segs.append(("line", a, P[(i + 1) % n])); i += 1
    return segs


def _plane():
    import cadquery as cq
    # sketch plane: local x = global X, local y = global Z, normal = -Y; extrude(-DEPTH) fills y = 0..DEPTH
    return cq.Plane(origin=(0, 0, 0), xDir=(1, 0, 0), normal=(0, -1, 0))


def solid(segs):
    import cadquery as cq
    wp = cq.Workplane(_plane()).moveTo(*segs[0][1])
    for s in segs:
        if s[0] == "line":
            wp = wp.lineTo(*s[2])
        else:
            wp = wp.threePointArc(tuple(s[2]), tuple(s[3]))
    return wp.close().extrude(-DEPTH)


def build_one(p, log):
    import cadquery as cq
    log.arcs.clear()
    bd = pg.bodies(p, n=3)
    arcs = list(log.arcs)
    asm = cq.Assembly()
    vols = {}
    for name, (kind, P) in bd.items():
        s = solid(primitives(P, arcs))
        vols[name] = s.val().Volume()
        asm.add(s, name=name, color=cq.Color(0.6, 0.65, 0.75) if kind == "frame" else cq.Color(0.95, 0.6, 0.4))
    return asm, vols


def build(tables, out, limit=None):
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    with _ArcLog() as log:
        for t in tables:
            df = pd.read_csv(table_path(t))
            if limit:
                df = df.head(limit)
            for _, r in df.iterrows():
                p = dict(pg.REF, **{v: float(r[v]) for v in pg.VAR_NAMES})  # REF supplies fixed values (t_w)
                asm, vols = build_one(p, log)
                path = out / f"{r.geom_id}.step"
                asm.save(str(path), exportType="STEP")
                area_ref = log.fine_area(p)
                rows.append(dict(geom_id=r.geom_id, file=path.name, volume_mm3=sum(vols.values()),
                                 area_from_volume=sum(vols.values()) / DEPTH, area_polygon=area_ref,
                                 area_diff=sum(vols.values()) / DEPTH - area_ref))
    pd.DataFrame(rows).to_csv(out / "step_build_log.csv", index=False, float_format="%.6g")
    print(f"built {len(rows)} files in {time.time() - t0:.1f} s")


# ----------------------------------------------------------------------------------------------------- verify
def verify(step_dir, tables):
    import cadquery as cq
    params = pd.concat([pd.read_csv(table_path(t)) for t in tables]).set_index("geom_id")
    log = pd.read_csv(step_dir / "step_build_log.csv").set_index("geom_id")
    bad = [(gid, ["not built"]) for gid in params.index.difference(log.index)]
    params = params.loc[params.index.isin(log.index)]   # a --limit build checks only what it built
    for gid, r in params.iterrows():
        sol = cq.importers.importStep(str(step_dir / f"{gid}.step")).val().Solids()
        vol = sum(s.Volume() for s in sol)
        bb = cq.Compound.makeCompound(sol).BoundingBox()
        overlap = sum(sol[i].intersect(sol[j]).Volume() for i in range(len(sol)) for j in range(i + 1, len(sol)))
        checks = {
            "6 solids": len(sol) == 6,
            "valid": all(s.isValid() for s in sol),
            "volume": abs(vol - log.loc[gid, "volume_mm3"]) < 1e-2 and abs(vol / 6 - r.area_mm2) < 0.05,
            "fixed face x": abs(bb.xmin - r.x_fixed_face) < 1e-4 and abs(bb.xmax) < 1e-6,
            "base z": abs(bb.zmin + 12.7) < 1e-6 and abs(bb.zmax - 12.7) < 1e-6,
            "y 0..6": abs(bb.ymin) < 1e-6 and abs(bb.ymax - 6) < 1e-6,
            "no overlap": overlap < 1e-6,
        }
        if not all(checks.values()):
            bad.append((gid, [k for k, v in checks.items() if not v]))
    print(f"checked {len(params)} files, failures: {len(bad)}")
    for b in bad[:20]:
        print(b)
    return bad


# ----------------------------------------------------------------------------------------------------- compare
def summary(path):
    """Geometry fingerprint of one STEP file: per-solid volume, area, face count, and the overall bounding box."""
    import cadquery as cq
    sol = cq.importers.importStep(str(path)).val().Solids()
    bb = cq.Compound.makeCompound(sol).BoundingBox()
    per = sorted((s.Volume(), s.Area(), len(s.Faces())) for s in sol)
    return dict(n_solids=len(sol), volumes=[x[0] for x in per], areas=[x[1] for x in per], faces=[x[2] for x in per],
                bbox=[bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax])


def compare(dir_a, dir_b, ids):
    """Largest differences between same-named STEP files of two folders (json on stdout)."""
    ids = ids or sorted(p.stem for p in dir_a.glob("*.step"))
    worst = dict(volume=0.0, area=0.0, bbox=0.0, faces_differ=[], solids_differ=[])
    for g in ids:
        a, b = summary(dir_a / f"{g}.step"), summary(dir_b / f"{g}.step")
        if a["n_solids"] != b["n_solids"]:
            worst["solids_differ"].append(g)
            continue
        if a["faces"] != b["faces"]:
            worst["faces_differ"].append(g)
        worst["volume"] = max(worst["volume"], float(np.abs(np.subtract(a["volumes"], b["volumes"])).max()))
        worst["area"] = max(worst["area"], float(np.abs(np.subtract(a["areas"], b["areas"])).max()))
        worst["bbox"] = max(worst["bbox"], float(np.abs(np.subtract(a["bbox"], b["bbox"])).max()))
    worst["n"] = len(ids)
    print(json.dumps(worst))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("out", type=Path); b.add_argument("tables", nargs="*")
    b.add_argument("--limit", type=int)
    v = sub.add_parser("verify"); v.add_argument("step_dir", type=Path); v.add_argument("tables", nargs="*")
    c = sub.add_parser("compare"); c.add_argument("dir_a", type=Path); c.add_argument("dir_b", type=Path)
    c.add_argument("ids", nargs="*")
    f = sub.add_parser("fingerprint"); f.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.tables or DEFAULT_TABLES, a.out, a.limit)
    elif a.cmd == "verify":
        sys.exit(1 if verify(a.step_dir, a.tables or DEFAULT_TABLES) else 0)
    elif a.cmd == "compare":
        compare(a.dir_a, a.dir_b, a.ids)
    else:   # json {file stem: summary} on stdout
        print(json.dumps({p.stem: summary(p) for p in a.files}))


if __name__ == "__main__":
    main()
