"""Fit forward models, referees and the cVAE once on a training set and save them (cabd.models.io).

  --train all    all 512 base training geometries (inverse design v1: final queries come from the test set)
  --train fold0  folds 1-4 only (dev queries = bi>mono geometries of fold 0, used only to check the optimizer)
  --train n512 / n1024 / n1536   the learning-curve training sets (Sobol order: base -> + ext1 -> + ext2);
                 n512 holds the same geometries as 'all'. Each size is fitted exactly like 'all' (same tuning budget).
Each (model, train, seed) is saved as {model}_{train}_s{seed} in --out as soon as it is done, so an interrupted run
resumes where it stopped (existing models are skipped). rom_gp first fits the ROM factors of that training set once
(rom_theta_{train}, no seed; reused when present in --out, models/ or artifacts/models/).
Default --out is artifacts/models (the converted original models live in models/ and artifacts/models/ under the
same names; pass --out explicitly to write elsewhere).
Torch training is single-threaded; parallel jobs run in separate processes.
GP fits depend on the OpenBLAS thread count at the 1e-4 level; joblib's loky workers get cpu_count // jobs threads,
so --jobs 16 on a 16-core machine (1 thread each) reproduces the saved GP models bit for bit (checked for gp_all_s0).

usage: python -m cabd.fit --models gp mlp rom_gp cvae referee referee_nogp --train all --seeds 0 1 2 3 4 [--jobs 16]
"""
import argparse
import time
import warnings
from pathlib import Path

from . import config, data
from .models import io

# slowest first so the pool stays busy
ORDER = {"referee": 0, "rom_gp": 1, "gp": 2, "referee_nogp": 3, "cvae": 4, "mlp": 5}


def rom_theta(train, out):
    """ROM factors of a training set: loaded if already fitted, otherwise fitted on its force curves and saved."""
    from .models import rom
    name = io.model_name("rom_theta", train)
    if (Path(out) / f"{name}.json").exists():
        return io.load(name, out)
    if io.exists(name):
        return io.load(name)
    th = rom.fit_theta(data.training_data(train))
    io.save(th, name, out)
    return th


def job(kind, train, seed, out):
    warnings.filterwarnings("ignore")
    from . import models as M
    name = io.model_name(kind, train, seed)
    if all(p.exists() for p in io.files(name, out)):
        return f"{name}: exists"
    t0 = time.time()
    d = data.training_data(train)
    theta = rom_theta(train, out)["theta"] if kind == "rom_gp" else None
    io.save(M.fit(kind, d, seed, theta=theta), name, out)
    msg = f"{name}: {time.time() - t0:.0f} s"
    print(msg, flush=True)
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, choices=["gp", "mlp", "rom_gp", "cvae", "referee", "referee_nogp"])
    ap.add_argument("--train", default="all", choices=["all", "fold0", "n512", "n1024", "n1536"])
    ap.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", type=Path, default=config.ARTIFACTS / "models")
    args = ap.parse_args()
    from joblib import Parallel, delayed
    args.out.mkdir(parents=True, exist_ok=True)
    if "rom_gp" in args.models:          # the ROM fit has no seed: do it once before the parallel jobs
        rom_theta(args.train, args.out)
    jobs = sorted([(m, args.train, s) for m in args.models for s in args.seeds], key=lambda j: ORDER[j[0]])
    for msg in Parallel(n_jobs=args.jobs)(delayed(job)(*j, args.out) for j in jobs):
        print(msg)


if __name__ == "__main__":
    main()
