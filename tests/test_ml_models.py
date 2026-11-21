"""Converted models predict exactly what the original models predicted (fixtures from make_legacy_fixtures.py).

Every model in models/MANIFEST.csv is checked when its files are present (models/ always; artifacts/models/ after the
Release asset is unpacked). Tolerance 1e-10 on mu, sd and P for the 256 test geometries, both input paths:
  physical inputs  cabd.models.predict(model, X)       vs legacy learning_curve.predict (mu, P)
  unit inputs      cabd.models.predictor(model)(U)     vs legacy inverse.predictor (mu, sd, P(bi>mono))
cVAE: samples and candidates for 3 query targets; ROM factors: theta and rms.
"""
import csv

import numpy as np
import pandas as pd
from _util import FIXTURES, Skip, assert_close

from cabd import config, data
from cabd.models import io

FIX = FIXTURES / "ml" / "legacy_pred"
TOL = 1e-10
ALLOWED = ("numpy", "sklearn", "scipy", "lightgbm", "torch", "cabd", "builtins", "collections", "copyreg", "_codecs",
           "joblib", "functools")


def manifest():
    with open(config.MODELS / "MANIFEST.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def names():
    return sorted({r["name"] for r in manifest()})


def feasible(u):
    """Same rule as the legacy inverse.feasible: inside the bounds and every geometry constraint satisfied."""
    from cabd.geometry import params as G
    if np.any(u < -1e-9) or np.any(u > 1 + 1e-9):
        return False
    p = dict(zip(config.X_COLS, data.to_phys(u)))
    p["t_w"] = G.T_W
    try:
        return all(G.check_constraints(p).values())
    except Exception:
        return False


def compare(name):
    """Max |diff| of every compared array for one model."""
    import torch
    from cabd import models as M
    torch.set_num_threads(1)
    fx = np.load(FIX / f"{name}.npz")
    m = io.load(name)
    kind = io.parse(name)[0]
    diffs = {}
    if kind == "rom_theta":
        diffs["theta"] = assert_close(m["theta"], fx["theta"], 0, f"{name} theta")
        diffs["rms"] = assert_close(m["rms"], fx["rms"], 0, f"{name} rms")
        return diffs
    if kind == "cvae":
        q = data.queries("final")[config.Y_COLS].to_numpy(float)
        for i in range(fx["samples"].shape[0]):
            diffs[f"samples{i}"] = assert_close(m.sample(q[i], seed=m.seed), fx["samples"][i], TOL, f"{name} sample")
            diffs[f"cand{i}"] = assert_close(m.candidates(q[i], feasible, seed=m.seed), fx[f"cand_{i}"], TOL, f"{name} cand")
        return diffs
    te = data.test_set()
    X = te[config.X_COLS].to_numpy(float)
    mu, sd, P = M.predict(m, X)
    if "mu_lc" in fx:
        diffs["mu"] = assert_close(mu, fx["mu_lc"], TOL, f"{name} mu")
        diffs["P"] = assert_close(P, fx["P_lc"], TOL, f"{name} P")
    else:   # referees: no learning_curve path; P from the legacy stages
        p1, p2 = fx["p1"], fx["p2"]
        diffs["mu"] = assert_close(mu, fx["mu_inv"], TOL, f"{name} mu")
        diffs["P"] = assert_close(P, np.column_stack([p1 * (1 - p2), p1 * p2, 1 - p1]), TOL, f"{name} P")
        assert sd is None
    mu_i, sd_i, pbm_i = M.predictor(m)(data.scale_x(X))
    diffs["mu_unit"] = assert_close(mu_i, fx["mu_inv"], TOL, f"{name} mu (unit inputs)")
    diffs["sd_unit"] = assert_close(sd_i, fx["sd_inv"], TOL, f"{name} sd (unit inputs)")
    diffs["pbm_unit"] = assert_close(pbm_i, fx["pbm_inv"], TOL, f"{name} P(bi>mono) (unit inputs)")
    if sd is not None:
        diffs["sd"] = assert_close(sd, fx["sd_inv"], TOL, f"{name} sd")
    return diffs


def test_converted_models_match_legacy():
    present = [n for n in names() if io.exists(n)]
    missing = sorted(set(names()) - set(present))
    worst = {}
    for n in present:
        d = compare(n)
        worst[n] = max(d.values())
        print(f"    {n:24s} max |diff| {worst[n]:.3g}")
    print(f"    {len(present)} models compared, overall max |diff| {max(worst.values()):.3g}")
    if missing:
        raise Skip(f"{len(present)} models match; {len(missing)} not present (unpack the Release asset): {missing[:4]} ...")


def test_manifest_matches_files():
    import hashlib
    rows = manifest()
    assert len({r["name"] for r in rows}) == len(list(FIX.glob("*.npz"))), "one manifest model per legacy model"
    checked = 0
    for r in rows:
        p = (config.MODELS if r["location"] == "models" else config.ARTIFACTS / "models") / r["file"]
        if not p.exists():
            assert r["location"] == "artifacts", f"{p} missing from models/"
            continue
        assert int(r["bytes"]) == p.stat().st_size and hashlib.sha256(p.read_bytes()).hexdigest() == r["sha256"], p
        checked += 1
    print(f"    {checked} of {len(rows)} files checked")


def test_pickles_reference_only_allowed_modules():
    """joblib files: record every module the unpickler imports; .pt files: must load with torch weights_only=True."""
    import joblib
    import torch
    from joblib import numpy_pickle
    seen = set()
    orig = numpy_pickle.NumpyUnpickler.find_class

    def find_class(self, mod, name):
        seen.add(mod)
        return orig(self, mod, name)
    numpy_pickle.NumpyUnpickler.find_class = find_class
    try:
        n = 0
        for r in manifest():
            p = (config.MODELS if r["location"] == "models" else config.ARTIFACTS / "models") / r["file"]
            if not p.exists() or p.suffix == ".json":
                continue
            if p.suffix == ".pt":
                torch.load(p, weights_only=True)
            else:
                joblib.load(p)
            n += 1
    finally:
        numpy_pickle.NumpyUnpickler.find_class = orig
    bad = {m for m in seen if m.split(".")[0] not in ALLOWED}
    print(f"    {n} files, modules: {sorted({m.split('.')[0] for m in seen})}")
    assert not bad, bad


def test_folds_and_queries_data():
    d = data.load()
    f = data.folds(d)
    assert len(d) == 512 and set(f) == {0, 1, 2, 3, 4}
    assert int(d.is_reg.sum()) == 408
    tr = data.train_set(1536)
    assert len(tr) == 1536 and int((tr.is_reg & tr.logE.notna()).sum()) == 1217
    te = data.test_set()
    assert int((te.is_reg & te.logE.notna()).sum()) == 207 and int(te.label.notna().sum()) == 241
    c = data.load_curves(["TR0000", "TR1535"])
    assert set(c) == {"TR0000", "TR1535"}


def test_queries_rebuild():
    from cabd.evaluate.queries import build
    new = build()
    old = pd.read_csv(data.QUERIES)
    pd.testing.assert_frame_equal(new.reset_index(drop=True), old, check_exact=True)
