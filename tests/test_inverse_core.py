"""cabd.inverse.core reproduces the original ml/inverse.py rules exactly (fixtures from fixtures/inverse/make_legacy_fixtures.py).

  objective_values, feasible, select, the Sobol starts       random inputs, tolerance 1e-12 (bools / indices exact)
  finite-difference batch of optimize_query                  captured by replacing minimize with a recorder:
      synthetic numpy predictor (tolerance 1e-12) and the converted gp_all_s0 (needs artifacts/models)
"""
import types

import numpy as np
from _util import FIXTURES, Skip, assert_close

from cabd.inverse import core

FIX = FIXTURES / "inverse" / "legacy_core.npz"
TOL = 1e-12


def fx():
    return np.load(FIX)


def synthetic_pred(U):
    """Same stand-in model as the fixture script."""
    U = np.asarray(U, float)
    w = np.linspace(0.5, 2.0, 8)
    a = np.sin(U @ w)
    b = np.cos(U @ w[::-1]) * U[:, 0]
    mu = np.column_stack([60 + 10 * a, 12 + 2 * b, -1 + a * b])
    sd = np.column_stack([1 + 0.5 * U[:, 1] ** 2, 0.1 + 0.05 * U[:, 2], 0.2 + 0.1 * np.abs(a)])
    return mu, sd, 1 / (1 + np.exp(-3 * (a + b)))


def test_objective_matches_legacy():
    f = fx()
    d = assert_close(core.objective_values(f["obj_mu"], f["obj_sd"], f["obj_t"]), f["obj_val"], TOL, "objective")
    # the app's masked objective equals the plain one when every output is targeted (where the plain one does not
    # overflow: exp(K e) is inf for e > ~35, which the legacy objective shares and the optimizer never reaches)
    fin = np.isfinite(f["obj_val"])
    m = core.objective_masked(f["obj_mu"], f["obj_sd"], f["obj_t"], np.array([True, True, True]))
    d2 = assert_close(m[fin], f["obj_val"][fin], 1e-12, "masked objective, full mask")
    print(f"    objective max |diff| {d:.3g} ({(~fin).sum()} of {fin.size} overflow to inf in both); "
          f"masked (all targets) {d2:.3g}")


def test_feasible_matches_legacy():
    f = fx()
    ok = np.array([core.feasible(u) for u in f["feas_u"]])
    assert (ok == f["feas_ok"]).all(), f"{(ok != f['feas_ok']).sum()} points differ"
    print(f"    {len(ok)} points identical ({ok.sum()} feasible)")


def test_select_and_starts_match_legacy():
    f = fx()
    for k in range(5):
        idx = core.select(f[f"sel_X{k}"], f[f"sel_s{k}"])
        assert list(idx) == list(f[f"sel_idx{k}"]), (k, idx, f[f"sel_idx{k}"])
    assert_close(core.starts(20), f["starts20"], 0, "Sobol starts")
    print("    5 candidate selections and the 20 starts identical")


def capture(pred, t, points):
    """New optimize_query with minimize replaced by the same recorder as the fixture script."""
    rec = []

    def fake_minimize(fun, x0, jac=None, constraints=None, **kw):
        for x in [x0] + list(points):
            x = np.asarray(x, float)
            c = constraints[0]
            rec.append(np.r_[fun(x), jac(x), c["fun"](x), c["jac"](x)])
        return types.SimpleNamespace(x=x0)

    orig = core.minimize
    core.minimize = fake_minimize
    try:
        res, info = core.optimize_query(pred, t, 3)
    finally:
        core.minimize = orig
    return np.array(rec), np.array([np.r_[x, fv, p] for x, fv, p in res]).reshape(-1, 10), info


def compare_batch(name, pred, tol):
    f = fx()
    rec, cand, info = capture(pred, f["batch_t"], f["batch_points"])
    d = assert_close(rec, f[f"batch_{name}"], tol, f"batch {name}")
    d2 = assert_close(cand, f[f"batch_{name}_cand"], tol, f"candidates {name}")
    assert_close([info["starts_done"], info["starts_valid"], info["best_any"]], f[f"batch_{name}_info"], tol, "info")
    return d, d2


def test_batch_gradient_synthetic_matches_legacy():
    d, d2 = compare_batch("synth", synthetic_pred, TOL)
    print(f"    synthetic predictor: objective / gradient / constraint max |diff| {d:.3g}, candidates {d2:.3g}")


def test_batch_gradient_gp_matches_legacy():
    import torch
    from cabd.models import exists, load, predictor
    torch.set_num_threads(1)
    if not exists("gp_all_s0"):
        raise Skip("gp_all_s0 not present (unpack the models Release asset into artifacts/)")
    d, d2 = compare_batch("gp", predictor(load("gp_all_s0")), TOL)
    print(f"    gp_all_s0: objective / gradient / constraint max |diff| {d:.3g}, candidates {d2:.3g}")
