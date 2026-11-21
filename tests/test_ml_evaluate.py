"""The evaluation steps regenerate the evidence files in results/ (forward, learning_curve, joint_output, figures).

  oof_metrics            *_metrics.json recomputed from the *_oof_seed*.csv files (cabd.metrics), tolerance 1e-12
  learning_curve         metrics.csv recomputed from the converted models (all rows whose models are present), 1e-9
  joint_output_summary   cv_summary.csv recomputed from cv_predictions.csv, 1e-9
  kin_theta              the joint-output 'kin' variant's ROM factors (rom_theta_n1536) = the legacy source
                         (theta of rom_gp_n1536_s0)
  figures                the three figures re-rendered pixel-identical to results/figures/*.png
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from _util import assert_close

from cabd import config, data
from cabd.models import io

FWD = config.RESULTS / "forward"


def _flatten(o, prefix=""):
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            out.update(_flatten(v, f"{prefix}/{k}"))
        return out
    return {prefix: np.asarray(o, float)}


def test_oof_metrics():
    from cabd.evaluate.cv import NAME, TAG
    from cabd.metrics import evaluate_oof
    d = data.load()
    worst = 0.0
    for kind, tag in TAG.items():
        old = json.loads((FWD / f"{tag}_metrics.json").read_text(encoding="utf-8"))
        assert old["method"] == NAME[kind], (old["method"], NAME[kind])
        for s, m_old in old["seeds"].items():
            new = json.loads(json.dumps(evaluate_oof(d, pd.read_csv(FWD / f"{tag}_oof_seed{s}.csv")), default=float))
            a, b = _flatten(new), _flatten(m_old)
            assert a.keys() == b.keys(), set(a) ^ set(b)
            for k in a:
                worst = max(worst, assert_close(a[k], b[k], 1e-12, f"{tag} seed {s} {k}"))
    print(f"    5 models x 5 seeds, max |diff| {worst:.3g}")


def test_learning_curve_metrics():
    from cabd.evaluate.learning_curve import evaluate
    old = pd.read_csv(config.RESULTS / "learning_curve" / "metrics.csv")
    new = evaluate(["mlp", "gp", "rom_gp"], config.TRAIN_SIZES, config.SEEDS)
    with tempfile.TemporaryDirectory() as t:      # same rounding as the file (float_format %.5f)
        new.to_csv(Path(t) / "m.csv", index=False, float_format="%.5f")
        new = pd.read_csv(Path(t) / "m.csv")
    key = ["model", "size", "seed"]
    m = old.merge(new, on=key, suffixes=("_old", "_new"))
    assert len(m) == len(new) and len(m) > 0
    cols = [c for c in old.columns if c not in key]
    d = assert_close(m[[c + "_new" for c in cols]].to_numpy(), m[[c + "_old" for c in cols]].to_numpy(), 1e-9, "metrics")
    print(f"    {len(m)} of {len(old)} rows recomputed (models present), max |diff| {d:.3g}")


def test_joint_output_summary():
    from cabd.evaluate.joint_output import OUT, summarize
    new = summarize(pd.read_csv(OUT / "cv_predictions.csv"))
    with tempfile.TemporaryDirectory() as t:
        new.to_csv(Path(t) / "s.csv", index=False, float_format="%.5f")
        new = pd.read_csv(Path(t) / "s.csv")
    old = pd.read_csv(OUT / "cv_summary.csv")
    assert list(new.variant) == list(old.variant)
    num = [c for c in old.columns if c != "variant"]
    d = assert_close(new[num].to_numpy(float), old[num].to_numpy(float), 1e-9, "cv_summary")
    print(f"    max |diff| {d:.3g}")


def test_kin_theta():
    th = io.load("rom_theta_n1536")["theta"]
    if io.exists("rom_gp_n1536_s0"):
        assert_close(th, io.load("rom_gp_n1536_s0")["theta"], 0, "rom_theta_n1536 vs rom_gp_n1536_s0 theta")
    assert len(th) == 5


def test_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as im
    import matplotlib.pyplot as plt

    from cabd.plots import correlation, distributions, learning_curve
    with tempfile.TemporaryDirectory() as t:
        for mod, name in ((correlation, "correlation_R"), (distributions, "distributions"), (learning_curve, "learning_curve")):
            mod.main(t)
            plt.close("all")
            a, b = im.imread(Path(t) / f"{name}.png"), im.imread(config.RESULTS / "figures" / f"{name}.png")
            assert a.shape == b.shape, name
            d = float(np.abs(a - b).max())
            print(f"    {name}.png max pixel diff {d}")
            assert d == 0, name
        for f in ("correlation_R_inputs_outputs.csv", "correlation_R_outputs.csv"):
            assert (Path(t) / f).read_bytes() == (config.RESULTS / "figures" / f).read_bytes(), f
