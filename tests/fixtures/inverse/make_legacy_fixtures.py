"""Baseline fixtures from the ORIGINAL inverse-design code and demo app (needs the original project folder, CABD_LEGACY).

Run once; tests/test_inverse_*.py and tests/test_app_*.py then compare the repo code against these files.
  inverse/legacy_core.npz   legacy ml/inverse.py on random inputs (seed 7):
      obj_mu, obj_sd, obj_t -> obj_val             objective_values
      feas_u -> feas_ok                            feasible (points in and slightly outside the box)
      sel_X{k}, sel_s{k} -> sel_idx{k}             select, 5 random sets
      batch_*                                      the finite-difference batch of optimize_query, captured by
                                                   replacing scipy's minimize with a recorder (no optimization run):
                                                   synthetic predictor (pure numpy) and the legacy gp_all_s0 model,
                                                   evaluated at the starts, random points and points on the upper bound
      starts20                                     the 20 Sobol starts
  app/legacy_engine.npz     legacy viz_app Engine on the 256 test geometries: p1, p2 (classify), mu, sd (regress)
  app/legacy_engine.json    legacy forward() for 3 designs (frames as sha256 of their JSON) and one inverse() request
The legacy code only reads its files; torch runs single-threaded.

usage: python -X utf8 tests/fixtures/inverse/make_legacy_fixtures.py [--skip-app-inverse]   (CABD_LEGACY set)
"""
import argparse
import hashlib
import json
import os
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
APP_OUT = HERE.parent / "app"
SEED = 7
APP_TARGET = {"theta2": 65.0, "u2": 12.0, "dE": 0.3}
APP_FORWARD_IDS = ["TE0018", "TE0094"]          # plus the app's default design


def synthetic_pred(U):
    """Smooth deterministic stand-in for a model: U (n, 8) -> mu (n, 3), sd (n, 3), P (n,)."""
    U = np.asarray(U, float)
    w = np.linspace(0.5, 2.0, 8)
    a = np.sin(U @ w)
    b = np.cos(U @ w[::-1]) * U[:, 0]
    mu = np.column_stack([60 + 10 * a, 12 + 2 * b, -1 + a * b])
    sd = np.column_stack([1 + 0.5 * U[:, 1] ** 2, 0.1 + 0.05 * U[:, 2], 0.2 + 0.1 * np.abs(a)])
    return mu, sd, 1 / (1 + np.exp(-3 * (a + b)))


def capture_batch(I, pred, t, points):
    """Run legacy optimize_query with minimize replaced by a recorder that evaluates fun / jac / constraint at points."""
    rec = []

    def fake_minimize(fun, x0, jac=None, constraints=None, **kw):
        for x in [x0] + list(points):
            x = np.asarray(x, float)
            c = constraints[0]
            rec.append(np.r_[fun(x), jac(x), c["fun"](x), c["jac"](x)])
        return types.SimpleNamespace(x=x0)

    orig = I.minimize
    I.minimize = fake_minimize
    try:
        res, info = I.optimize_query(pred, t, 3)
    finally:
        I.minimize = orig
    return np.array(rec), res, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-app-inverse", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    legacy = Path(os.environ["CABD_LEGACY"])
    sys.path.insert(0, str(legacy / "ml"))
    import joblib
    import torch
    torch.set_num_threads(1)
    import inverse as I

    rng = np.random.default_rng(SEED)
    out = {}
    # objective
    mu = rng.normal(0, 3, (200, 3))
    sd = np.abs(rng.normal(0, 1, (200, 3)))
    t = rng.normal(0, 3, 3)
    out.update(obj_mu=mu, obj_sd=sd, obj_t=t, obj_val=I.objective_values(mu, sd, t))
    # feasible
    U = np.vstack([rng.random((300, 8)), rng.uniform(-0.05, 1.05, (40, 8)), I.starts(20)])
    out.update(feas_u=U, feas_ok=np.array([I.feasible(u) for u in U]))
    print(f"feasible: {out['feas_ok'].sum()} of {len(U)}")
    # select
    for k in range(5):
        X = np.vstack([I.starts(20)[rng.permutation(20)[:12]], rng.random((40, 8)) * 0.3 + 0.35])
        s = rng.normal(size=len(X))
        out[f"sel_X{k}"], out[f"sel_s{k}"] = X, s
        out[f"sel_idx{k}"] = np.array(I.select(X, s), int)
        print(f"select {k}: {out[f'sel_idx{k}']}")
    out["starts20"] = I.starts(20)
    # batch (finite differences), synthetic predictor and the legacy GP model
    pts = rng.random((4, 8))
    pts_top = pts.copy()
    pts_top[:, [1, 5]] = 1.0                       # on the upper bound -> backward differences
    points = np.vstack([pts, pts_top])
    q = pd.read_csv(REPO / "data" / "queries.csv")
    tq = q[q.set == "final"][["theta2", "u2", "logE"]].to_numpy(float)[0]
    out["batch_points"], out["batch_t"] = points, tq
    for name, pred in [("synth", synthetic_pred),
                       ("gp", I.predictor("gp", joblib.load(legacy / "ml" / "results" / "models" / "gp_all_s0.joblib")))]:
        rec, res, info = capture_batch(I, pred, tq, points)
        out[f"batch_{name}"] = rec
        out[f"batch_{name}_cand"] = np.array([np.r_[x, fv, p] for x, fv, p in res]).reshape(-1, 10)
        out[f"batch_{name}_info"] = np.array([info["starts_done"], info["starts_valid"], info["best_any"]])
    np.savez_compressed(HERE / "legacy_core.npz", **out)
    print(f"wrote {HERE / 'legacy_core.npz'}")

    # ------------------------------------------------------------------ demo app
    sys.path.insert(0, str(legacy / "viz_app"))
    import engine as E
    eng = E.Engine()
    te = pd.read_csv(REPO / "data" / "CABD_test_results.csv")
    X = te[E.X].to_numpy(float)
    Ut = (X - E.LO) / (E.HI - E.LO)
    p1, p2 = eng.classify(Ut)
    m, s = eng.regress(Ut)
    APP_OUT.mkdir(exist_ok=True)
    np.savez_compressed(APP_OUT / "legacy_engine.npz", U=Ut, p1=p1, p2=p2, mu=m, sd=s)
    js = {"forward": {}}
    designs = {"default": [E.DEFAULT[c] for c in E.X], **{g: te.set_index("geom_id").loc[g, E.X].tolist() for g in APP_FORWARD_IDS}}
    for k, x in designs.items():
        js["forward"][k] = {"x": x, "res": strip_frames(eng.forward(np.array(x)))}
    js["geometry_default"] = hashlib.sha256(json.dumps(E.geometry_payload(np.array(designs["default"])), default=to_json).encode()).hexdigest()
    js["meta"] = E.meta()
    if not a.skip_app_inverse:
        r = eng.inverse(dict(APP_TARGET))
        r["forward"] = strip_frames(r["forward"])
        js["inverse"] = {"targets": APP_TARGET, "res": r}
        print(f"app inverse: {r['time_s']} s, achievable {r['achievable']}, x {np.round(r['x'], 4)}")
    (APP_OUT / "legacy_engine.json").write_text(json.dumps(js, indent=1, default=to_json), encoding="utf-8")
    print(f"wrote {APP_OUT / 'legacy_engine.npz'}, {APP_OUT / 'legacy_engine.json'}")


def to_json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def strip_frames(res):
    """forward() result with the (large) frames replaced by the sha256 of their JSON."""
    res = dict(res)
    if "frames" in res:
        res["frames_sha256"] = hashlib.sha256(json.dumps(res.pop("frames"), default=to_json).encode()).hexdigest()
    return res


if __name__ == "__main__":
    main()
