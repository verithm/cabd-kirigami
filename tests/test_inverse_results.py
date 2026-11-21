"""cabd.inverse regenerates the original result files in results/inverse (copied unchanged from the original project).

  cVAE candidates          cvae_final_n20_s0.csv (v1, first 5 final queries) and v2/cvae_final_train1536_n20_s0.csv
                           (first 3 queries): every row of the result, tolerance 1e-12
  referee scoring          gp_final_n20_s0.csv (v1 rule, 512 referees) and v2/gp_final_train1536_n20_s0.csv (v2 rule,
                           size-matched referees): referee columns dropped and regenerated, tolerance 1e-10
  full optimize_query      (slow, CABD_SLOW=1) 3 ansys40 queries x gp / mlp / rom_gp, seed 0, 512 'all' models, and
                           1 query with the 1536 GP (v2): rank-1..5 candidates, objective, P(bi>mono), starts_done and
                           starts_valid. Only queries whose original run finished all 20 starts (the wall-clock cap
                           did not cut them). Each query runs in its own process, torch single-threaded; v1 also with
                           BLAS single-threaded (the original ran in loky workers with 16 jobs on 16 cores, which limits
                           BLAS to 1 thread), the v2 GP query with the default BLAS threads (the original v2 GP file
                           was rerun with --jobs 1, i.e. in the main process).
"""
import io
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from _util import Skip, assert_close

from cabd import config, data
from cabd.config import X_COLS

INV = config.RESULTS / "inverse"
HERE = Path(__file__).resolve().parent
SLOW_QUERIES = ["TE0186", "TE0254", "TE0252"]
NUM = ["t_theta2", "t_u2", "t_logE", "own_obj", "own_Pbm"] + X_COLS + [f"u_{c}" for c in X_COLS]


def need(*names):
    from cabd.models import exists
    missing = [n for n in names if not exists(n)]
    if missing:
        raise Skip(f"models not present (unpack the models Release asset into artifacts/): {missing}")


def compare_rows(new, old, what, tol=1e-12):
    """new: rows from one_query (or the csv table of them); old: the rows of the same query in the original csv.

    new goes through the same csv round trip as the result files (cVAE designs and MLP probabilities are float32,
    written in float32 repr)."""
    if not isinstance(new, pd.DataFrame):
        buf = io.StringIO()
        pd.DataFrame(new).to_csv(buf, index=False)
        buf.seek(0)
        new = pd.read_csv(buf)
    assert list(new["rank"]) == list(old["rank"]), f"{what}: ranks {list(new['rank'])} != {list(old['rank'])}"
    for c in ["starts_done", "starts_valid", "ansys40", "geom_id", "method"]:
        assert list(new[c]) == list(old[c]), f"{what}: {c} differs"
    cols = [c for c in NUM if c in old and old[c].notna().any()]
    return assert_close(new[cols].to_numpy(float), old[cols].to_numpy(float), tol, what)


def test_cvae_candidates_match_result_files():
    from cabd.inverse.run import one_query
    need("cvae_all_s0", "cvae_n1536_s0")
    q = data.queries("final")
    worst = 0.0
    for path, train, n in [(INV / "cvae_final_n20_s0.csv", "all", 5), (INV / "v2" / "cvae_final_train1536_n20_s0.csv", "n1536", 3)]:
        old = pd.read_csv(path)
        qq = q[q.geom_id.isin(old.geom_id)].head(n)
        for _, row in qq.iterrows():
            new = one_query("cvae", train, "final", 0, 20, row)
            worst = max(worst, compare_rows(new, old[old.geom_id == row.geom_id], f"{path.name} {row.geom_id}"))
        if train != "all":
            assert (pd.DataFrame(new).train_size == 1536).all()
    print(f"    8 queries, every candidate identical, max |diff| {worst:.3g}")


def test_referee_scoring_matches_result_files():
    from cabd.inverse.score import score_frame
    worst = {}
    for path, train in [(INV / "gp_final_n20_s0.csv", "all"), (INV / "v2" / "gp_final_train1536_n20_s0.csv", "n1536")]:
        need(*(f"{r}_{train}_s0" for r in ("referee", "referee_nogp")))
        old = pd.read_csv(path)
        ref_cols = [c for c in old.columns if c.startswith("referee")]
        new = score_frame(old.drop(columns=ref_cols), train)
        assert list(new.columns) == list(old.columns), f"{path.name}: columns {list(new.columns)[-14:]}"
        num = [c for c in ref_cols if not c.endswith("_ok")]
        worst[path.name] = assert_close(new[num].to_numpy(float), old[num].to_numpy(float), 1e-10, path.name)
        for c in ref_cols:
            if c.endswith("_ok"):
                assert (new[c].astype(bool) == old[c].astype(bool)).all(), f"{path.name} {c}"
        top = new[new["rank"] <= 1]
        print(f"    {path.name}: {int((old['rank'] > 0).sum())} candidates, max |diff| {worst[path.name]:.3g}, "
              f"top-1 success referee {top.referee_ok.mean():.3f}")


def run_query(method, train, geom, single_thread):
    env = dict(os.environ, PYTHONPATH=str(HERE.parent / "src"))
    if single_thread:
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    p = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "_inverse_query.py"), method, train, "0", geom],
                       capture_output=True, text=True, env=env)
    assert "ROWS\n" in p.stdout, f"{method} {train} {geom} failed:\n{p.stderr[-2000:]}"
    return pd.read_csv(io.StringIO(p.stdout.split("ROWS\n", 1)[1]))


def test_optimize_query_matches_result_files():
    if os.environ.get("CABD_SLOW") != "1":
        raise Skip("slow (about 5 min with 6 processes): set CABD_SLOW=1")
    need("gp_all_s0", "mlp_all_s0", "rom_gp_all_s0", "gp_n1536_s0")
    jobs = [(m, "all", g, True) for g in SLOW_QUERIES for m in ("rom_gp", "gp", "mlp")] + [("gp", "n1536", "TE0186", False)]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda j: run_query(*j), jobs))
    for (m, train, g, _), rows in zip(jobs, res):
        path = INV / (f"{m}_final_n20_s0.csv" if train == "all" else f"v2/{m}_final_train1536_n20_s0.csv")
        old = pd.read_csv(path)
        old = old[old.geom_id == g]
        assert old.starts_done.iloc[0] == 20
        d = compare_rows(rows, old, f"{path.name} {g}", tol=1e-12)
        print(f"    {m:6s} {train:5s} {g}: {len(rows)} candidates, starts {rows.starts_done[0]}/{rows.starts_valid[0]} "
              f"valid, {rows.time[0]:.0f} s (original {old.time.iloc[0]:.0f} s), max |diff| {d:.3g}")
