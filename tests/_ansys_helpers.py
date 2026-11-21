"""Helpers for the geometry / Ansys tests (skips for slow runs, raw zips and the CAD python; image comparison)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from _util import Skip


def slow():
    """Heavy tests run only with CABD_SLOW=1."""
    if os.environ.get("CABD_SLOW") != "1":
        raise Skip("heavy test: set CABD_SLOW=1 to run it")


def raw():
    """The raw Ansys result zips (GitHub Release asset) must be unpacked into ANSYS_RAW."""
    from cabd.ansys.results import zips
    from cabd.config import ANSYS_RAW
    found = zips()
    if not {"1", "2", "3", "final"} <= set(found):
        raise Skip(f"raw Ansys result zips not found in {ANSYS_RAW} (Release asset ansys_raw)")
    return found


def cad():
    """A python with cadquery (CABD_CAD_PYTHON, or the running one)."""
    from cabd.config import CAD_PYTHON
    py = CAD_PYTHON or sys.executable
    if subprocess.run([py, "-c", "import cadquery"], capture_output=True).returncode != 0:
        raise Skip("no cadquery: set CABD_CAD_PYTHON to the CAD environment's python")
    return py


def tmp(name):
    d = Path(tempfile.mkdtemp(prefix=f"cabd_{name}_"))
    return d


def image_diff(a, b):
    """Max absolute pixel difference of two PNG files (raises if the sizes differ)."""
    import matplotlib.image as mpimg
    import numpy as np
    x, y = mpimg.imread(a), mpimg.imread(b)
    assert x.shape == y.shape, f"image size {x.shape} != {y.shape}"
    return float(np.abs(x.astype(float) - y.astype(float)).max())
