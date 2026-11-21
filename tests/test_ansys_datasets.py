"""Datasets regenerated from the raw Ansys result zips (artifacts/ansys_raw) equal the committed files in data/.

Fast check: the scalars of a sample of geometries. Heavy (CABD_SLOW=1): every dataset file byte for byte, and the
curve files (decompressed JSON text; gzip headers carry a time stamp)."""
import gzip

import numpy as np
import pandas as pd

from _ansys_helpers import raw, slow, tmp
from cabd.ansys import dataset as D
from cabd.ansys.results import Results, scalars
from cabd.config import DATA


def test_scalars_of_sample_match_dataset():
    """Training rows finished on PC1-3 and not touched by the final kit: scalars recomputed from the raw zips."""
    raw()
    data = pd.read_csv(DATA / "CABD_dataset.csv").set_index("geom_id")
    assign = D.assignment("base")
    final = Results("final", overlay=False).read("progress.csv")
    redone = set(final.geom_id)
    ids = [g for g in data.index[data.group == "train_base"] if g not in redone][::25]
    worst = 0.0
    for g in ids:
        res = Results(int(assign.loc[g, "pc"]))
        for s in D.STATES:
            if data.loc[g, f"{s}_run_status"] != "finished":
                continue
            sc = scalars(res, g, s)
            assert sc.get("class", np.nan) == data.loc[g, f"{s}_class"] or pd.isna(data.loc[g, f"{s}_class"]), (g, s)
            for k in D.NUMERIC:
                a, b = sc.get(k, np.nan), data.loc[g, f"{s}_{k}"]
                assert np.isnan(a) == pd.isna(b), (g, s, k)
                if not np.isnan(a):
                    worst = max(worst, abs(a - b))
    print(f"{len(ids)} geometries, max |diff| {worst:.2g}")
    assert worst <= 1e-9


def _same_json_gz(a, b):
    return gzip.open(a, "rt", encoding="ascii").read() == gzip.open(b, "rt", encoding="ascii").read()


def test_base_dataset_identical():
    slow(); raw()
    out = tmp("ds_base")
    D.base(out)
    for f in ("CABD_dataset.csv", "CABD_test_results.csv", "CABD_dataset_columns.csv"):
        assert (out / f).read_bytes() == (DATA / f).read_bytes(), f
    for f in ("curves_base.json.gz", "curves_test.json.gz"):
        assert _same_json_gz(out / "curves" / f, DATA / "curves" / f), f


def test_ext_dataset_identical():
    slow(); raw()
    out = tmp("ds_ext")
    D.ext(out)
    assert (out / "CABD_dataset_ext.csv").read_bytes() == (DATA / "CABD_dataset_ext.csv").read_bytes()
    assert _same_json_gz(out / "curves" / "curves_ext.json.gz", DATA / "curves" / "curves_ext.json.gz")


def test_geometry_table_identical():
    """geometry_table.csv is the snapshot of 2026-11-08 23:55 (final kit just built): --final built reproduces it."""
    slow(); raw()
    out = tmp("ds_table")
    D.geometry_table(out, final="built")
    assert (out / "geometry_table.csv").read_bytes() == (DATA / "geometry_table.csv").read_bytes()
