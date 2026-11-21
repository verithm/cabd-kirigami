"""Refits with the repo code reproduce the original fits (slow: opt-in with CABD_SLOW=1).

  cv_slice_<kind>   one (seed 0, fold 0) slice of the 5-fold CV vs the rows of results/forward/<tag>_oof_seed0.csv
  refit_rom_theta   ROM factors on all 512 base geometries vs models/rom_theta_all.json
  refit_gp_theta2   one GP regressor (theta2, 512 geometries, seed 0) vs gp_all_s0 (artifacts)
  refit_mlp         MLP ensemble on 512 geometries, seed 0 vs mlp_all_s0 (artifacts)
  refit_cvae        cVAE seed 0 on the 408 partial-bi base geometries vs models/cvae_all_s0
Tolerance 1e-9 (absolute, on predictions of the 256 test geometries or on the OOF rows); the max |diff| is printed.
Approximate single-process run times are in the comments.

BLAS threads: the GP fits (restarted L-BFGS on the marginal likelihood) depend on the OpenBLAS thread count at the
1e-4 level, so each refit runs with the thread count the original run had inside its joblib (loky) workers
(cabd.evaluate.cv.ORIG_THREADS): 3 for the gp CV (5 jobs), 1 for the rom_gp CV (it reproduces only with 1 thread; with
3 threads max |diff| 2.4e-5, same for the original code), 1 for the referee CV (15 jobs) and for cabd.fit (16 jobs).
The count must be set when OpenBLAS loads (environment variables, as loky does for its workers; limiting it later with
threadpoolctl does not reproduce the fits), so these refits run in one loky worker (in_worker).
Measured: GP theta2 refit with 1 thread = gp_all_s0 bit for bit (two runs); with 3 threads or with 16 threads limited
to 1 by threadpoolctl, max |diff| 1.1e-4; the two original fits gp_all_s0 and gp_n512_s0 (same data and seed, run
with different job counts) differ by 1.2e-4.
"""
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from _util import Skip, assert_close

from cabd import config, data
from cabd.models import io

TOL = 1e-9


def slow():
    if os.environ.get("CABD_SLOW") != "1":
        raise Skip("slow refit: set CABD_SLOW=1")


def in_worker(threads, fn, *args):
    """fn(*args) in a fresh loky worker whose BLAS / OpenMP thread count is fixed at start-up."""
    from joblib import Parallel, delayed, parallel_config
    with parallel_config(backend="loky", inner_max_num_threads=threads):
        return Parallel(n_jobs=2)([delayed(fn)(*args)])[0]


def _cv_slice(kind, seed=0, k=0):
    slow()
    from cabd.evaluate import cv
    d = data.load()
    fold = data.folds(d)
    t0 = time.time()
    out, info = in_worker(cv.ORIG_THREADS[kind], cv.run_fold, kind, d, fold, k, seed)   # original thread count
    old = pd.read_csv(config.RESULTS / "forward" / f"{cv.TAG[kind]}_oof_seed{seed}.csv")
    old = old[old.fold == k].reset_index(drop=True)
    with tempfile.TemporaryDirectory() as t:    # through CSV like cv.main: float32 P columns are written as text
        out.to_csv(Path(t) / "oof.csv", index=False)
        new = pd.read_csv(Path(t) / "oof.csv")
    assert list(new.columns) == list(old.columns), (list(new.columns), list(old.columns))
    assert (new.geom_id == old.geom_id).all()
    num = [c for c in old.columns if c != "geom_id"]
    diff = assert_close(new[num].to_numpy(float), old[num].to_numpy(float), TOL, f"{kind} seed {seed} fold {k}")
    print(f"    {kind} seed {seed} fold {k}: max |diff| {diff:.3g}, {time.time() - t0:.0f} s")
    return info


def test_cv_slice_mlp():        # ~5 min
    _cv_slice("mlp")


def test_cv_slice_gp():         # ~15 min
    _cv_slice("gp")


def test_cv_slice_rom_gp():     # ~30 min
    info = _cv_slice("rom_gp")
    p = pd.read_csv(config.RESULTS / "forward" / "rom_params.csv")
    p = p[(p.seed == 0) & (p.fold == 0)].iloc[0]
    assert_close([info[c] for c in ("rms", "theta0", "theta1", "theta2", "theta3", "theta4")],
                 p[["rms", "theta0", "theta1", "theta2", "theta3", "theta4"]].to_numpy(float), TOL, "rom params")


def test_cv_slice_referee_nogp():   # ~20 min
    _cv_slice("referee_nogp")


def test_cv_slice_referee_gpx():      # several hours (GP member with RBF + linear kernel)
    _cv_slice("referee")


def _test_preds(m):
    from cabd import models as M
    return M.predict(m, data.test_set()[config.X_COLS].to_numpy(float))


def test_refit_rom_theta():         # ~10 min
    slow()
    from cabd.models import rom
    new = rom.fit_theta(data.load())
    old = io.load("rom_theta_all")
    d = assert_close(new["theta"], old["theta"], TOL, "theta")
    print(f"    theta max |diff| {d:.3g}, rms {new['rms']:.6f} vs {old['rms']:.6f}")


def test_refit_gp_theta2():         # ~10 min
    slow()
    if not io.exists("gp_all_s0"):
        raise Skip("gp_all_s0 not present (Release asset)")
    from cabd.models import gp
    d = data.load()
    r = d.is_reg.to_numpy()
    t0 = time.time()
    # cabd.fit / the original full_fit: 16 jobs -> 1 BLAS thread per worker
    g = in_worker(1, gp.reg_model(0).fit, data.scale_x(d)[r], d.loc[r, "theta2"].to_numpy(float))
    U = data.scale_x(data.test_set())
    mu, sd = g.predict(U, return_std=True)
    mo, so = io.load("gp_all_s0")["reg"][0].predict(U, return_std=True)
    print(f"    GP theta2 refit {time.time() - t0:.0f} s, max |diff| mu {np.abs(mu - mo).max():.3g}, sd {np.abs(sd - so).max():.3g}")
    assert_close(mu, mo, TOL, "mu"), assert_close(sd, so, TOL, "sd")


def test_refit_mlp():               # ~8 min
    slow()
    if not io.exists("mlp_all_s0"):
        raise Skip("mlp_all_s0 not present (Release asset)")
    from cabd.models import mlp
    m = mlp.fit(data.load(), 0)
    for a, b, w in zip(_test_preds(m), _test_preds(io.load("mlp_all_s0")), ("mu", "sd", "P")):
        print(f"    MLP refit {w}: max |diff| {np.abs(a - b).max():.3g}")
        assert_close(a, b, TOL, w)


def test_refit_cvae():              # ~20 min
    slow()
    from cabd.models import cvae
    g = cvae.fit(data.load(), 0)
    old = io.load("cvae_all_s0")
    assert g.cfg == old.cfg and g.epochs == old.epochs, (g.cfg, g.epochs, old.cfg, old.epochs)
    t = data.queries("final")[config.Y_COLS].to_numpy(float)[0]
    d = assert_close(g.sample(t, seed=0), old.sample(t, seed=0), TOL, "cVAE samples")
    print(f"    cVAE refit: same config, samples max |diff| {d:.3g}")
