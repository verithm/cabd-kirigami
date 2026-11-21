"""Helpers shared by the tests."""
import os
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class Skip(Exception):
    """Raised by a test that cannot run here (missing legacy folder or artifacts)."""


def legacy_root():
    """The original (pre-repo) project folder, for legacy-equivalence tests; skip if it is not on this machine."""
    p = os.environ.get("CABD_LEGACY")
    if not p or not Path(p).exists():
        raise Skip("set CABD_LEGACY to the original project folder to run this comparison")
    return Path(p)


def assert_close(a, b, tol, what):
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, f"{what}: shape {a.shape} != {b.shape}"
    same_nan = np.isnan(a) == np.isnan(b)
    assert same_nan.all(), f"{what}: NaN pattern differs"
    d = np.nanmax(np.abs(a - b)) if a.size and not np.isnan(a).all() else 0.0
    assert d <= tol, f"{what}: max |diff| {d:.3g} > {tol:.3g}"
    return float(d)
