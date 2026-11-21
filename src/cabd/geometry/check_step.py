"""Round-trip check of the parametrisation against the team's reference STEP files (the __main__ of param_geom_v3).

For each reproducible reference (params.VARIANTS, base half-length Z_o = 12.7 mm) every body section is read from the
STEP file (cabd.geometry.stepparse, no CAD kernel), matched to the generated body with the closest area and centroid,
and compared by area difference and Hausdorff distance. The references with another base length
(params.NOT_REPRODUCIBLE: Z_o = 12.3 / 13.0 mm) are excluded by design; for them only the base half-length is printed.

The reference files (bistable_ref*.stp, from the shared Ansys project) are not part of the repository; put them in
--ref-dir (default data/sampling/reference_step). Missing files are reported and skipped.

usage: python -m cabd.geometry.check_step [--ref-dir DIR]
"""
import argparse
from pathlib import Path

import numpy as np

from cabd.config import SAMPLING
from cabd.geometry import params as pg
from cabd.geometry.stepparse import describe, loop_polygon

REF_DIR = SAMPLING / "reference_step"


def check_variant(path, p):
    """Max Hausdorff distance (mm) between STEP sections and generated bodies, with per-body lines."""
    s, res = describe(str(path))
    gen = pg.bodies(p, n=200)
    lines, hmax = [], 0.0
    for bid, dd in res.items():
        P = loop_polygon(dd["loops"][0], n_arc=200)
        cen = P.mean(0); A = pg.poly_area(P)
        best = min(gen.items(), key=lambda kv: abs(pg.poly_area(kv[1][1]) - A) + np.linalg.norm(kv[1][1].mean(0) - cen))
        Q = best[1][1]
        dPQ = np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1))
        h = max(dPQ.min(1).max(), dPQ.min(0).max()); hmax = max(hmax, h)
        lines.append(f"   step #{bid} -> {best[0]:16s} dA={A - pg.poly_area(Q):+.5f}  hausdorff={h:.5f}")
    return hmax, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", type=Path, default=REF_DIR)
    d = ap.parse_args().ref_dir
    print(f"Z_o fixed = {pg.Z_O}, rho(REF) = {pg.REF['rho']:.4f};  derived(REF) alpha = {pg.derived(pg.REF)['alpha']:.4f} deg")
    for fn, p in pg.VARIANTS.items():
        if not (d / fn).exists():
            print(f"--- {fn}: not found in {d}, skipped")
            continue
        print(f"--- {fn}  (t_h={p['t_h']}, t_a={p['t_a']})  total area {pg.total_area(p):.3f}  est nodes {pg.est_nodes(p):.0f}")
        hmax, lines = check_variant(d / fn, p)
        print("\n".join(lines))
        print(f"   max Hausdorff over bodies = {hmax:.5f} mm")
        print("   constraints all OK:", all(pg.check_constraints(p).values()))
    for fn, zo in pg.NOT_REPRODUCIBLE.items():
        if not (d / fn).exists():
            print(f"--- {fn}: not found in {d}, skipped")
            continue
        s, res = describe(str(d / fn))
        frame = [dd for dd in res.values() if abs(pg.poly_area(loop_polygon(dd["loops"][0]))) > 50][0]
        P = loop_polygon(frame["loops"][0])
        print(f"--- {fn}: base half-length from STEP = {P[:, 1].max():.3f} (expected {zo}) -> not reproducible with Z_o = {pg.Z_O}")


if __name__ == "__main__":
    main()
