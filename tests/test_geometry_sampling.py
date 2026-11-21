"""Sobol sampling tables, range check and the design-variable figure regenerate the committed files exactly."""
import io
import json

import pandas as pd

from _ansys_helpers import image_diff, slow, tmp
from _util import assert_close
from cabd.config import RESULTS, SAMPLING
from cabd.geometry import sampling


def _csv(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False, float_format=sampling.FLOAT_FORMAT, lineterminator=sampling.CRLF)
    return buf.getvalue().encode("utf-8")


def test_base_tables_and_meta_identical():
    tables, meta = sampling.base_tables()
    for role, df in tables.items():
        assert _csv(df) == (SAMPLING / f"params_{role}.csv").read_bytes(), role
    text = json.dumps(meta, indent=2).replace("\n", sampling.CRLF).encode("utf-8")
    assert text == (SAMPLING / "params_meta.json").read_bytes()


def test_ext_tables_identical():
    for stage in (1, 2):
        df, info = sampling.ext_table(sampling.earlier_tables(SAMPLING, stage))
        assert _csv(df) == (SAMPLING / f"params_train_ext{stage}.csv").read_bytes(), stage
        print(f"ext{stage}: rejected {info['rejected']}, earlier points reproduced within {info['reproduced_within']:.1e}")
    # the original ext1 script numbered rows consecutively; ext1 had no rejection, so the ids agree
    ext1 = pd.read_csv(SAMPLING / "params_train_ext1.csv")
    assert list(ext1.geom_id) == [f"TR{512 + i:04d}" for i in range(512)]


def test_verify_tables_from_candidate_lists():
    """params_verify*.csv from the 9-digit candidate lists (exact ids, values within the 6-digit rounding).
    The byte-exact rebuild from the full-precision inverse results is in test_ansys_kits."""
    from cabd.config import RESULTS
    from cabd.verify.candidates import FILES
    worst = 0.0
    for k, name in ((1, "params_verify.csv"), (2, "params_verify2.csv")):
        c = pd.read_csv(RESULTS / "inverse" / FILES[k], keep_default_na=False, na_values=[""])
        new = pd.read_csv(io.BytesIO(_csv(sampling.table_from_candidates(c))))
        old = pd.read_csv(SAMPLING / name)
        assert list(new.geom_id) == list(old.geom_id) and list(new.columns) == list(old.columns)
        num = [c for c in old.columns if c not in ("geom_id", "role")]
        rel = (new[num] - old[num]).abs() / old[num].abs().clip(lower=1)
        worst = max(worst, float(rel.max().max()))
    assert worst <= 1e-5, worst
    print(f"max relative difference {worst:.2g} (6-significant-digit rounding)")


def test_range_check():
    slow()
    from cabd.geometry import range_check
    r = range_check.run()
    assert (r["points"], r["invalid"]) == (4352, 22)
    assert r["fails"] == {"x_t>=-t_h (C2 stays right of ligament inner face)": 22}
    assert_close([*r["alpha"], *r["nodes"]], [29.7, 62.4, 62613, 100479], 0.5, "alpha / node range")


def test_variables_figure_identical():
    from cabd.geometry import variables_figure
    out = tmp("varfig") / "v.png"
    variables_figure.draw(out)
    d = image_diff(out, RESULTS / "figures" / "geometry_design_variables_v3.png")
    print(f"max pixel difference {d}")
    assert d == 0.0
