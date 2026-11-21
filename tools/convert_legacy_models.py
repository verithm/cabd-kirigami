"""One-time conversion of the original project's fitted models into the repo formats (cabd.models.io).

Sources (in the original project folder, given by --legacy or the CABD_LEGACY environment variable):
  ml/results/models/{kind}_{all|fold0}_s{seed}.joblib, rom_theta_{all|fold0}.joblib
      the 512-geometry models of inverse design v1 ('all' = 512 base geometries, 'fold0' = folds 1-4 for the dev check)
  ml/results/learning_curve/models/{kind}_n{512|1024|1536}_s{seed}.joblib, rom_theta_n{size}.joblib
      the learning-curve models (also used by inverse design v2 and the demo app)
The legacy pickles reference the original modules (cvae.Generator, referee.Stacked), so the legacy ml/ folder is put
on sys.path to read them. Nothing is refitted: weights, configurations and scalings are copied as they are.

Layout (both folders have the same flat layout, cabd.models.io.load looks in MODELS first, then ARTIFACTS/models):
  models/                    in git, small: the demo app's 1536-geometry MLP ensembles (seeds 0-4), every ROM factor
                             set (rom_theta_*) and the cVAEs of inverse design v1 and v2 (cvae_all_s*, cvae_n1536_s0)
  artifacts/models/          GitHub Release asset: everything else (all GP, ROM+GP and referee models, the other MLP
                             sizes / seeds, cvae_fold0_s0, cvae_n1024_s0)
  models/MANIFEST.csv        one row per converted file: name, kind, train, n_train, seed, file, location, bytes, sha256
File formats: see cabd.models.io (joblib dicts of scikit-learn / lightgbm objects for GP, ROM+GP and referees; torch
state_dict .pt + .json for MLP and cVAE; .json for ROM factors).

usage: python -X utf8 tools/convert_legacy_models.py --legacy <original project folder> [--only NAME ...]
"""
import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from cabd import config, data  # noqa: E402
from cabd.models import io  # noqa: E402

IN_GIT = {f"mlp_n1536_s{s}" for s in range(5)} | {f"cvae_all_s{s}" for s in range(5)} | {"cvae_n1536_s0"}


def location(name):
    return "models" if name in IN_GIT or name.startswith("rom_theta_") else "artifacts"


def root_of(loc):
    return config.MODELS if loc == "models" else config.ARTIFACTS / "models"


def legacy_sources(legacy):
    ml = Path(legacy) / "ml" / "results"
    return sorted((ml / "models").glob("*.joblib")) + sorted((ml / "learning_curve" / "models").glob("*.joblib"))


def convert(name, m):
    """Legacy object -> repo object (same weights)."""
    kind = io.parse(name)[0]
    if kind == "rom_theta":
        return {"kind": "rom_theta", "theta": np.asarray(m["theta"], float), "rms": float(m["rms"])}
    if kind in ("gp", "rom_gp", "mlp"):
        return {"kind": kind, **m}
    if kind in ("referee", "referee_nogp"):
        from cabd.models.referee import Stacked

        def restack(s):
            new = Stacked.__new__(Stacked)
            new.__dict__.update(s.__dict__)
            return new
        return {"kind": kind, "reg": [restack(s) for s in m["reg"]], "c1": restack(m["c1"]), "c2": restack(m["c2"])}
    if kind == "cvae":
        from cabd.models.cvae import CVAE, Generator
        g = Generator()
        for k in ("cfg", "mean", "sd", "epochs", "score", "seed"):
            setattr(g, k, getattr(m, k))
        g.model = CVAE(m.cfg)
        g.model.load_state_dict(m.model.state_dict())
        return g
    raise ValueError(name)


def n_train(train):
    if train.startswith("n"):
        return int(train[1:])
    return len(data.training_data(train))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_manifest(rows):
    path = config.MODELS / "MANIFEST.csv"
    cols = ["name", "kind", "train", "n_train", "seed", "file", "location", "bytes", "sha256"]
    rows = sorted(rows, key=lambda r: (r["location"] != "models", r["name"], r["file"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


def manifest_rows(names):
    rows, sizes = [], {}
    for name in names:
        kind, train, seed = io.parse(name)
        loc = location(name)
        if train not in sizes:
            sizes[train] = n_train(train)
        for p in io.files(name, root_of(loc)):
            rows.append(dict(name=name, kind=kind, train=train, n_train=sizes[train], seed="" if seed is None else seed,
                             file=p.name, location=loc, bytes=p.stat().st_size, sha256=sha256(p)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", default=os.environ.get("CABD_LEGACY"), help="original project folder")
    ap.add_argument("--only", nargs="*", help="convert only these model names")
    args = ap.parse_args()
    if not args.legacy:
        ap.error("give --legacy or set CABD_LEGACY")
    sys.path.insert(0, str(Path(args.legacy) / "ml"))     # legacy pickles need cvae.py / referee.py
    names = []
    for src in legacy_sources(args.legacy):
        name = src.stem
        names.append(name)
        if args.only and name not in args.only:
            continue
        paths = io.save(convert(name, joblib.load(src)), name, root_of(location(name)))
        print(f"{name:24s} -> {location(name)}: {', '.join(p.name for p in paths)}", flush=True)
    path = write_manifest(manifest_rows(names))
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for loc in ("models", "artifacts"):
        print(f"{loc}: {len({r['name'] for r in rows if r['location'] == loc})} models, "
              f"{sum(int(r['bytes']) for r in rows if r['location'] == loc) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
