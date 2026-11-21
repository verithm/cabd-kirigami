"""Check every corner of the v3 design box plus 4096 scrambled-Sobol points (seed 7) for geometric validity.

Result for params.RANGES: 22 of the 4352 points invalid, all by "x_t >= -t_h" (thinnest tip hinge with the largest
fillet); alpha 29.7-62.4 deg, estimated nodes 62.6k-100.5k. The design-variable summary figure quotes 4 of the 4096
Sobol points (0.1%) as invalid; the table generator (cabd.geometry.sampling) skips such points.

usage: python -m cabd.geometry.range_check
"""
import itertools
from collections import Counter

from scipy.stats import qmc

from cabd.geometry import params as pg


def run():
    V = pg.VAR_NAMES
    pts = [dict(pg.REF, **{v: pg.RANGES[v][b] for v, b in zip(V, bits)}) for bits in itertools.product([0, 1], repeat=len(V))]
    U = qmc.Sobol(d=len(V), scramble=True, seed=7).random_base2(12)
    pts += [dict(pg.REF, **{v: pg.RANGES[v][0] + u[i] * (pg.RANGES[v][1] - pg.RANGES[v][0]) for i, v in enumerate(V)}) for u in U]
    fails, al, nodes, bad = Counter(), [], [], []
    for p in pts:
        c = pg.check_constraints(p)
        if not all(c.values()):
            fails.update(k for k, ok in c.items() if not ok); bad.append(p)
        al.append(pg.derived(p)["alpha"]); nodes.append(pg.est_nodes(p))
    return dict(points=len(pts), invalid=len(bad), fails=dict(fails), alpha=(min(al), max(al)), nodes=(min(nodes), max(nodes)),
                bad=bad)


def main():
    r = run()
    print("points", r["points"], "invalid", r["invalid"], r["fails"])
    print("alpha %.1f-%.1f, nodes %.0f-%.0f" % (*r["alpha"], *r["nodes"]))
    for p in r["bad"][:5]:
        print({k: round(p[k], 3) for k in pg.VAR_NAMES})


if __name__ == "__main__":
    main()
