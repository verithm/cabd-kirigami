"""Verification scoring, candidate lists, shared viewer and the Ansys figures regenerate the committed evidence."""
import json

import pandas as pd

from _ansys_helpers import image_diff, raw, tmp
from _util import Skip, legacy_root
from cabd.config import RESULTS
from cabd.verify import candidates as C
from cabd.verify import score as S


def test_candidate_lists_from_inverse_results():
    """results/inverse/ansys_verify*_candidates.csv (copies) regenerate from the original inverse result csvs."""
    inv = legacy_root() / "ml" / "results" / "inverse"
    out = tmp("cand")
    c1 = C.kit1(inv)
    C.write(c1, out / C.FILES[1])
    assert (out / C.FILES[1]).read_bytes() == (RESULTS / "inverse" / C.FILES[1]).read_bytes()
    c2 = C.kit2(inv, pd.read_csv(out / C.FILES[1]))
    C.write(c2, out / C.FILES[2])
    assert (out / C.FILES[2]).read_bytes() == (RESULTS / "inverse" / C.FILES[2]).read_bytes()
    print(f"kit 1: {len(c1)} candidates, kit 2: {len(c2)}")


def test_score_kit1_identical():
    """results/verify/ (scored 2026-11-18, kit 1 only) byte for byte, and the figure pixel for pixel."""
    raw()
    out = tmp("score")
    S.run(out, kits=(1,))
    for f in ("ansys_verify_scored.csv", "ansys_verify_summary.csv"):
        assert (out / f).read_bytes() == (RESULTS / "verify" / f).read_bytes(), f
    d = image_diff(out / "ansys_verify.png", RESULTS / "verify" / "ansys_verify.png")
    print(f"figure max pixel difference {d}")
    assert d == 0.0


def test_score_self_test():
    """The 40 ansys40 test geometries as their own candidates: every solved one succeeds with zero error.
    (PC1-3 records only, as the original: 24 of 40 are solved there, the rest were finished by the final kit.)"""
    raw()
    s = S.self_test()
    assert int(s.solved.sum()) == 24 and bool(s[s.solved].ansys_ok.all()) and float(s.ansys_maxerr_tau.max()) == 0.0


def test_viewer_answers():
    from cabd.ansys.viewer import app
    rows = json.loads(app.rows_json())
    assert len(rows["rows"]) == 1000 and len(rows["columns"]) == 37
    g = json.loads(app.geometry_json("TR0000"))
    assert g["found"] and not g["sealed"] and g["curve"].startswith("data:image/svg+xml")
    assert json.loads(app.geometry_json("TE0000"))["sealed"]


def test_figures_identical():
    from cabd.ansys import figures as F
    out = tmp("figs")
    F.classes("partial", out)
    done = ["ansys_class_examples_partial.png"]
    try:
        raw()
    except Skip as e:   # no raw zips: only the figure made from the shared files
        print("raw-result figures skipped:", e)
    else:
        F.kit_check(out)
        for g in ("TR0032", "TR0036", "TR0293"):
            F.theta(g, ["partial", "full"], out)
        done += ["ansys_kit_verification.png"] + [f"ansys_theta_{g}.png" for g in ("TR0032", "TR0036", "TR0293")]
    for f in done:
        d = image_diff(out / f, RESULTS / "figures" / f)
        assert d == 0.0, (f, d)
    print(f"{len(done)} figures pixel-identical")
