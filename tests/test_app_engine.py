"""Demo app: the new bundle (models/app_models_n1536.joblib, cabd.app.bundle) and engine give the original app's outputs.

Fixtures from the original viz_app engine with its own bundle (fixtures/inverse/make_legacy_fixtures.py ->
fixtures/app/legacy_engine.npz / .json):
  bundle            < 50 MB, unpickles only numpy / scikit-learn / scipy / plain-Python objects (no torch, no legacy modules)
  classify/regress  P(partial bi), P(full mono | bi), mean and sd of (theta2, u2, log10 dE_r) on the 256 test
                    geometries, tolerance 1e-10
  forward           3 designs: every number 1e-10, deformation frames identical (sha256 of their JSON), geometry payload
                    and meta identical
  inverse           (slow, CABD_SLOW=1) one request {theta2 65, u2 12, dE 0.3}: every number 1e-10 (time_s excepted)
  server            start on a free port, GET /, GET /api/meta, POST /api/forward, stop
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from _util import FIXTURES, Skip, assert_close

FIX = FIXTURES / "app"
HERE = Path(__file__).resolve().parent
TOL = 1e-10
ALLOWED = ("numpy", "sklearn", "scipy", "cabd", "builtins", "collections", "copyreg", "_codecs", "joblib", "functools")
_ENGINE = None


def engine():
    global _ENGINE
    from cabd.app import bundle
    if not bundle.BUNDLE.exists():
        raise Skip("models/app_models_n1536.joblib missing: python -m cabd.app.bundle")
    if _ENGINE is None:
        import torch
        torch.set_num_threads(1)
        from cabd.app.engine import Engine
        _ENGINE = Engine()
    return _ENGINE


def fixture_json():
    return json.loads((FIX / "legacy_engine.json").read_text(encoding="utf-8"))


def to_json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def strip_frames(res):
    res = dict(res)
    if "frames" in res:
        res["frames_sha256"] = hashlib.sha256(json.dumps(res.pop("frames"), default=to_json).encode()).hexdigest()
    return res


def compare(a, b, path="", skip=("time_s",)):
    """Nested dict / list comparison: same keys and strings, numbers within TOL. Returns max |diff|."""
    if isinstance(b, dict):
        assert isinstance(a, dict) and set(a) == set(b), f"{path}: keys {sorted(a)} != {sorted(b)}"
        return max([compare(a[k], b[k], f"{path}.{k}") for k in b if k not in skip] + [0.0])
    if isinstance(b, list):
        assert isinstance(a, list) and len(a) == len(b), f"{path}: length differs"
        return max([compare(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b))] + [0.0])
    if isinstance(b, bool) or b is None or isinstance(b, str):
        assert a == b, f"{path}: {a!r} != {b!r}"
        return 0.0
    d = abs(float(a) - float(b))
    assert d <= TOL, f"{path}: {a} != {b}"
    return d


def test_bundle_size_and_pickle_content():
    import joblib
    from joblib import numpy_pickle
    from cabd.app import bundle
    if not bundle.BUNDLE.exists():
        raise Skip("models/app_models_n1536.joblib missing: python -m cabd.app.bundle")
    mb = bundle.BUNDLE.stat().st_size / 1e6
    seen = set()
    orig = numpy_pickle.NumpyUnpickler.find_class

    def find_class(self, mod, name):
        seen.add(mod)
        return orig(self, mod, name)
    numpy_pickle.NumpyUnpickler.find_class = find_class
    try:
        b = joblib.load(bundle.BUNDLE)
    finally:
        numpy_pickle.NumpyUnpickler.find_class = orig
    bad = {m for m in seen if m.split(".")[0] not in ALLOWED}
    assert not bad, bad
    assert mb < 50, f"bundle {mb:.1f} MB"
    assert len(b["mlp"]) == 5 and all(len(m["reg"]) == 5 for m in b["mlp"]) and b["n_train"] == 1536
    print(f"    {mb:.1f} MB, modules {sorted({m.split('.')[0] for m in seen})}")


def test_classify_regress_match_legacy():
    e = engine()
    f = np.load(FIX / "legacy_engine.npz")
    p1, p2 = e.classify(f["U"])
    mu, sd = e.regress(f["U"])
    d = {"p1": assert_close(p1, f["p1"], TOL, "P(partial bi)"), "p2": assert_close(p2, f["p2"], TOL, "P(full mono | bi)"),
         "mu": assert_close(mu, f["mu"], TOL, "mean"), "sd": assert_close(sd, f["sd"], TOL, "sd")}
    print("    256 test geometries, max |diff| " + ", ".join(f"{k} {v:.3g}" for k, v in d.items()))


def test_forward_geometry_meta_match_legacy():
    from cabd.app import engine as E
    e = engine()
    js = fixture_json()
    worst = 0.0
    for k, v in js["forward"].items():
        res = strip_frames(e.forward(np.array(v["x"])))
        worst = max(worst, compare(json.loads(json.dumps(res, default=to_json)), v["res"], k))
    g = hashlib.sha256(json.dumps(E.geometry_payload(np.array(js["forward"]["default"]["x"])), default=to_json).encode()).hexdigest()
    assert g == js["geometry_default"], "geometry payload differs"
    assert json.loads(json.dumps(E.meta())) == js["meta"]
    print(f"    {len(js['forward'])} designs (frames identical), max |diff| {worst:.3g}; geometry payload and meta identical")


def test_inverse_matches_legacy():
    if os.environ.get("CABD_SLOW") != "1":
        raise Skip("slow (about 1.5 min, 10 worker processes): set CABD_SLOW=1")
    e = engine()
    js = fixture_json()["inverse"]
    res = e.inverse(dict(js["targets"]))
    res["forward"] = strip_frames(res["forward"])
    d = compare(json.loads(json.dumps(res, default=to_json)), js["res"], "inverse")
    print(f"    target {js['targets']}: achievable {res['achievable']}, {res['n_valid']} valid starts, "
          f"x {np.round(res['x'], 4).tolist()}, max |diff| {d:.3g}, {res['time_s']} s")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_smoke():
    engine()                                  # skips if the bundle is missing
    port = free_port()
    env = dict(os.environ, PYTHONPATH=str(HERE.parent / "src"))
    p = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "cabd.app.server", "--port", str(port), "--no-warm-up"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        line = p.stdout.readline()
        assert "running at" in line, line
        base = f"http://127.0.0.1:{port}"
        html = urllib.request.urlopen(base + "/", timeout=30).read().decode()
        assert "Bistable Particle Designer" in html
        meta = json.loads(urllib.request.urlopen(base + "/api/meta", timeout=30).read())
        x = [v["default"] for v in meta["vars"]]
        req = urllib.request.Request(base + "/api/forward", data=json.dumps({"x": x}).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        fw = json.loads(urllib.request.urlopen(req, timeout=120).read())
        assert fw["ok"] and fw["combination_ok"] and len(fw["frames"]) > 0
        print(f"    port {port}: GET / ({len(html)} bytes), /api/meta, POST /api/forward ({time.time() - t0:.1f} s, "
              f"theta2 {fw['theta2']['mean']:.2f} deg, u2 {fw['u2']['mean']:.2f} mm)")
    finally:
        p.terminate()
        p.wait(10)
