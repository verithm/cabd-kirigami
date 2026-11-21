"""Run inverse design for a set of queries with the four methods (rules in cabd.inverse.core).

Two studies share this one entry point:
  v1 (--train all, 2026-11-16/17): models fitted on the 512 base geometries, seeds 0-4.
      final queries use the models fitted on all 512 ({method}_all_s{seed}); dev queries use the models fitted on
      folds 1-4 ({method}_fold0_s{seed}, the dev queries are the bi>mono geometries of fold 0).
      writes results/inverse/{method}_{set}_n{starts}_s{seed}.csv
      parallel unit: one (method, seed) file per worker, its queries in order (as the original inverse.py)
  v2 (--train n1024 / n1536, 2026-11-18): the same methods and rules with the forward models refitted on 1024 or 1536
      training geometries (cabd.evaluate.learning_curve, seed 0), final queries only; the rank-1 answers of the 40
      ansys40 queries went to the second Ansys verification kit.
      writes results/inverse/v2/{method}_final_train{size}_n{starts}_s{seed}.csv (extra column train_size)
      parallel unit: one (method, size, query) per worker so all cores stay busy (as the original inverse_v2.py)
Existing result files are never overwritten (delete one to rerun it); single queries: cabd.inverse.patch.
Referee columns are added afterwards by cabd.inverse.score.

Each worker runs torch single-threaded (torch.set_num_threads(1)); joblib's loky workers also limit BLAS to
cpu_count // jobs threads. The per-query cap is wall-clock time, so heavy contention cuts starts (see core):
the v2 GP and ROM+GP files were rerun with --jobs 1 after a contended first run with 12 workers.

How the result files in results/inverse were produced (the commands of the original scripts, mapped to this CLI):
  v1 final:  python -m cabd.inverse.run --train all --query-set final --methods gp mlp rom_gp cvae --seeds 0 1 2 3 4
  v1 dev:    python -m cabd.inverse.run --train all --query-set dev --seeds 0 [--starts 20 | --starts 40]
  v2:        python -m cabd.inverse.run --train n1024 n1536 --ansys40 --jobs 12           (MLP, cVAE)
             python -m cabd.inverse.run --train n1024 n1536 --ansys40 --methods rom_gp gp --jobs 1
             then python -m cabd.inverse.patch for 6 standby-hit queries (see cabd.inverse.patch)

usage: python -m cabd.inverse.run [--train all | n1024 n1536] [--methods gp mlp rom_gp cvae] [--seeds 0 1 2 3 4]
                                  [--query-set final|dev] [--ansys40] [--starts 20] [--jobs N] [--out DIR]
"""
import argparse
import time
import warnings
from pathlib import Path

import pandas as pd

from .. import config, data
from ..config import X_COLS, Y_COLS
from . import core

OUT = config.RESULTS / "inverse"
METHODS = ["gp", "mlp", "rom_gp", "cvae"]
METHODS_V2 = ["rom_gp", "gp", "mlp", "cvae"]          # slowest first, for the per-query job queue


def is_v1(train):
    return train == "all"


def model_name(method, train, qset, seed):
    """Name of the model file a run uses (cabd.models.io)."""
    if is_v1(train):
        return f"{method}_{'all' if qset == 'final' else 'fold0'}_s{seed}"
    return f"{method}_{train}_s{seed}"


def out_path(method, train, qset, seed, n_start, out=OUT):
    out = Path(out)
    if is_v1(train):
        return out / f"{method}_{qset}_n{n_start}_s{seed}.csv"
    return out / "v2" / f"{method}_{qset}_train{train[1:]}_n{n_start}_s{seed}.csv"


_MODELS = {}


def load_model(name):
    """Load once per worker process."""
    if name not in _MODELS:
        from ..models import load
        _MODELS[name] = load(name)
    return _MODELS[name]


def _worker_setup():
    warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)


def one_query(method, train, qset, seed, n_start, row):
    """Candidates for one query row (from data.queries) -> list of result rows (rank 0 = no candidate)."""
    _worker_setup()
    m = load_model(model_name(method, train, qset, seed))
    t = row[Y_COLS].to_numpy(float)
    if method == "cvae":
        res, info = core.cvae_query(m, t, seed)
    else:
        from ..models import predictor
        res, info = core.optimize_query(predictor(m), t, n_start)
    base = dict(method=method, set=qset, seed=seed, n_start=n_start)
    if not is_v1(train):
        base["train_size"] = int(train[1:])
    base.update(geom_id=row.geom_id, t_theta2=t[0], t_u2=t[1], t_logE=t[2], ansys40=row.ansys40, **info)
    if not res:
        return [dict(base, rank=0)]          # no candidate: counts as a failure
    return [dict(base, rank=k, own_obj=fv, own_Pbm=pbm, **dict(zip(X_COLS, data.to_phys(x))),
                 **{f"u_{c}": v for c, v in zip(X_COLS, x)}) for k, (x, fv, pbm) in enumerate(res, 1)]


def queries(qset, ansys40=False):
    q = data.queries(qset)
    return q[q.ansys40].reset_index(drop=True) if ansys40 else q


def run_file(method, train, qset, seed, n_start, ansys40, out):
    """v1 unit: every query of the set in order, one file."""
    path = out_path(method, train, qset, seed, n_start, out)
    if path.exists():
        return f"{path.name}: exists"
    q = queries(qset, ansys40)
    t_all = time.time()
    rows = [r for _, row in q.iterrows() for r in one_query(method, train, qset, seed, n_start, row)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    msg = f"{path.name}: {len(q)} queries, {time.time() - t_all:.0f} s"
    print(msg, flush=True)
    return msg


def run_v2(methods, trains, qset, seed, n_start, ansys40, jobs, out):
    """v2 unit: one job per (method, size, query); files written when all their queries are done."""
    from joblib import Parallel, delayed
    q = queries(qset, ansys40)
    todo = [(m, tr) for m in methods for tr in trains if not out_path(m, tr, qset, seed, n_start, out).exists()]
    jobs_list = [(m, tr, r) for m, tr in todo for _, r in q.iterrows()]
    print(f"{len(todo)} (method, size) pairs, {len(jobs_list)} query jobs", flush=True)
    t0 = time.time()
    rows = Parallel(n_jobs=jobs, verbose=5)(delayed(one_query)(m, tr, qset, seed, n_start, r) for m, tr, r in jobs_list)
    for m, tr in todo:
        part = [r for (jm, jt, _), rs in zip(jobs_list, rows) if (jm, jt) == (m, tr) for r in rs]
        df = pd.DataFrame(part)
        path = out_path(m, tr, qset, seed, n_start, out)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        top = df[df["rank"] == 1]
        print(f"{path.name}: {df.geom_id.nunique()} queries, rank-1 found {len(top)}, "
              f"starts done mean {df.groupby('geom_id').starts_done.first().mean():.1f}", flush=True)
    print(f"total {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", default=["all"], choices=["all", "n1024", "n1536"],
                    help="all = v1 (512 geometries); n1024 / n1536 = v2 (may be given together)")
    ap.add_argument("--methods", nargs="+", choices=METHODS, help="default gp mlp rom_gp cvae")
    ap.add_argument("--seeds", type=int, nargs="+", help="default 0-4 for v1, 0 for v2")
    ap.add_argument("--query-set", default="final", choices=["final", "dev"])
    ap.add_argument("--ansys40", action="store_true", help="only the 40 ansys40 queries of the final set")
    ap.add_argument("--starts", type=int, default=core.N_START)
    ap.add_argument("--jobs", type=int, help="parallel workers (default 16 for v1, 12 for v2)")
    ap.add_argument("--out", type=Path, default=OUT, help="result folder (default results/inverse)")
    a = ap.parse_args()
    v1 = [t for t in a.train if is_v1(t)]
    v2 = [t for t in a.train if not is_v1(t)]
    if v1 and v2:
        ap.error("--train all (v1) and n1024 / n1536 (v2) are separate runs")
    if v2 and a.query_set != "final":
        ap.error("v2 runs use the final query set only")
    if v1:
        from joblib import Parallel, delayed
        methods, seeds = a.methods or METHODS, a.seeds or config.SEEDS
        jobs = [(m, "all", a.query_set, s, a.starts, a.ansys40, a.out) for m in methods for s in seeds]
        for msg in Parallel(n_jobs=a.jobs or 16)(delayed(run_file)(*j) for j in jobs):
            print(msg)
    else:
        seeds = a.seeds or [0]
        for s in seeds:
            run_v2(a.methods or METHODS_V2, v2, a.query_set, s, a.starts, a.ansys40, a.jobs or 12, a.out)


if __name__ == "__main__":
    main()
