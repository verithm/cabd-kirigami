"""STEP export (CAD python): regenerated files match the original STEP files by geometry, pass the re-import checks,
and parse back (pure-Python parser) onto the parametrised sections. Heavy: CABD_SLOW=1 and CABD_CAD_PYTHON."""
import json

import numpy as np
import pandas as pd

from _ansys_helpers import cad, slow, tmp
from _util import FIXTURES
from cabd.config import SAMPLING
from cabd.geometry import params as pg
from cabd.geometry.step import run_cad

TABLES = {"base": ["params_train.csv", "params_test.csv"], "ext1": ["params_train_ext1.csv"], "ext2": ["params_train_ext2.csv"],
          "verify": ["params_verify.csv"], "verify2": ["params_verify2.csv"]}


def _build_sample(d):
    fx = json.loads((FIXTURES / "geometry" / "step_fingerprints.json").read_text())
    for name, prints in fx.items():
        t = pd.concat([pd.read_csv(SAMPLING / f) for f in TABLES[name]])
        sub = t[t.geom_id.isin(prints)]
        sub.to_csv(d / f"sample_{name}.csv", index=False)
    run_cad("build", d / "step", *[d / f"sample_{n}.csv" for n in fx])
    return fx


def test_step_files_match_original_geometry():
    slow(); cad()
    d = tmp("step")
    fx = _build_sample(d)
    files = [d / "step" / f"{g}.step" for prints in fx.values() for g in prints]
    new = json.loads(run_cad("fingerprint", *files).stdout.strip().splitlines()[-1])
    worst = {"volume": 0.0, "area": 0.0, "bbox": 0.0}
    for prints in fx.values():
        for g, old in prints.items():
            n = new[g]
            assert n["n_solids"] == old["n_solids"] == 6 and n["faces"] == old["faces"], g
            for k, key in (("volume", "volumes"), ("area", "areas"), ("bbox", "bbox")):
                worst[k] = max(worst[k], float(np.abs(np.subtract(n[key], old[key])).max()))
    print(f"{len(new)} files, max |diff| {worst}")
    assert max(worst.values()) <= 1e-9, worst
    # re-import checks of the original verify_step_files (6 valid solids, volume, bounding box, no overlap)
    r = run_cad("verify", d / "step", *[d / f"sample_{n}.csv" for n in fx], check=False)
    assert r.returncode == 0, r.stdout


def test_stepparse_round_trip():
    """A generated STEP file parsed by cabd.geometry.stepparse reproduces params.bodies (the original reference STEP
    files of check_step are not available; this checks the same parser and matching on our own export)."""
    slow(); cad()
    from cabd.geometry.check_step import check_variant
    d = tmp("stepref")
    row = dict(geom_id="REF", **{v: pg.REF[v] for v in pg.VAR_NAMES})
    pd.DataFrame([row]).to_csv(d / "ref.csv", index=False)
    run_cad("build", d, d / "ref.csv")
    hmax, lines = check_variant(d / "REF.step", dict(pg.REF))
    print("\n".join(lines), f"\nmax Hausdorff {hmax:.2e} mm")
    assert hmax < 1e-3, hmax
