"""Datasets, labels, folds, training sets and force curves: the same data rules for every method.

Labels (decided 2026-11-15):
  partial/full class from the sign of the force: bi, mono; neg_end counts as bi; unknown is dropped.
  3 classes of the pair: bi>bi, bi>mono, mono>mono (mono>bi never occurs).
Regression: only geometries whose partial curve really crossed back (partial_class == "bi"), 408 of the 512 base
  training geometries. Outputs theta2 [deg], u2 [mm], log10 dE_r [mJ].
Training sets follow the Sobol order: base (train_base, 512) -> + ext1 (PC4-6 + retry, 512) -> + ext2 (PC7-9, 512).
The sealed test set (256) has its outputs in CABD_test_results.csv (the shared dataset keeps them empty).

Files (all under config.DATA):
  CABD_dataset.csv        base dataset (train_base, test design variables, adaptive, inverse_validation groups)
  CABD_dataset_ext.csv    train_ext1 + train_ext2 (1024 extra training geometries)
  CABD_test_results.csv   the 256 test geometries with their Ansys outputs
  folds.csv               the fixed stratified 5-fold split of the 512 base geometries (read only, never rewritten)
  queries.csv             the inverse-design queries (see cabd.evaluate.queries)
  curves/curves_base.json.gz  force curves of the 512 base geometries (the shared viewer file: u, F, U, S, ...)
  curves/curves_ext.json.gz   force curves of the 1024 extra geometries (u, F only)
  curves/curves_test.json.gz  force curves of the 256 test geometries (viewer format; used by the viewer / app only)
"""
import gzip
import json

import numpy as np
import pandas as pd

from .config import BOUNDS, CLASSES, DATA, X_COLS

DATASET = DATA / "CABD_dataset.csv"
DATASET_EXT = DATA / "CABD_dataset_ext.csv"
TEST_RESULTS = DATA / "CABD_test_results.csv"
FOLDS = DATA / "folds.csv"
QUERIES = DATA / "queries.csv"
CURVES = {"base": DATA / "curves" / "curves_base.json.gz", "ext": DATA / "curves" / "curves_ext.json.gz",
          "test": DATA / "curves" / "curves_test.json.gz"}

LO = np.array([BOUNDS[c][0] for c in X_COLS])
HI = np.array([BOUNDS[c][1] for c in X_COLS])


# ------------------------------------------------------------------ design variables
def scale_x(x):
    """Design variables mapped to [0, 1] by their sampling bounds. x: DataFrame with X_COLS, or (n, 8) array."""
    a = x[X_COLS].to_numpy(float) if isinstance(x, pd.DataFrame) else np.asarray(x, float)
    return (a - LO) / (HI - LO)


def to_phys(u):
    """Inverse of scale_x: [0, 1] units -> physical design variables (n, 8)."""
    return LO + np.asarray(u) * (HI - LO)


# ------------------------------------------------------------------ labels
def state_label(cls):
    return {"bi": "bi", "neg_end": "bi", "mono": "mono"}.get(cls)   # unknown -> None


def prepare(d):
    """Derived columns: label (3-class string or None), is_reg, theta2, u2, logE."""
    d = d.reset_index(drop=True).copy()
    p, f = d.partial_class.map(state_label), d.full_class.map(state_label)
    d["label"] = np.where(p.isna() | f.isna(), None, p + ">" + f)
    d["is_reg"] = d.partial_class == "bi"
    d["theta2"] = d.partial_theta2_deg
    d["u2"] = d.partial_u2_mm
    d["logE"] = np.log10(d.partial_dE_r_mJ.where(d.partial_dE_r_mJ > 0))
    return d


def class_targets(d):
    """Targets of the two-stage classifiers.

    ok     labelled rows                      -> stage 1 trains on d[ok], y1 = partial is bi
    is_bi  labelled rows whose partial is bi  -> stage 2 trains on d[is_bi], y2 = full is mono (given partial bi)
    """
    lab = d.label.to_numpy(object)
    ok = np.array([v is not None and v == v for v in lab])
    is_bi = np.array([ok[i] and lab[i].startswith("bi") for i in range(len(lab))])
    y1 = np.array([v.startswith("bi") for v in lab[ok]])
    y2 = np.array([v.endswith("mono") for v in lab[is_bi]])
    return ok, is_bi, y1, y2


def class_index(labels):
    """3-class labels -> integer index in config.CLASSES."""
    return np.array([CLASSES.index(v) for v in labels])


# ------------------------------------------------------------------ datasets
def load(group="train_base"):
    """One group of the base dataset with the derived columns (default: the 512 base training geometries)."""
    d = pd.read_csv(DATASET)
    return prepare(d[d.group == group])


def train_set(size):
    """The first `size` training geometries in Sobol order (512, 1024 or 1536)."""
    base = pd.read_csv(DATASET)
    base = base[base.group == "train_base"]
    ext = pd.read_csv(DATASET_EXT)
    parts = {512: [base], 1024: [base, ext[ext.group == "train_ext1"]],
             1536: [base, ext[ext.group == "train_ext1"], ext[ext.group == "train_ext2"]]}[size]
    d = prepare(pd.concat(parts, ignore_index=True))
    assert len(d) == size and d.geom_id.is_unique
    return d


def test_set():
    """The 256 sealed test geometries with their Ansys outputs."""
    return prepare(pd.read_csv(TEST_RESULTS))


def training_data(train):
    """Named training sets: 'all' (the 512 base geometries), 'fold0' (folds 1-4 only), 'n512' / 'n1024' / 'n1536'."""
    if train == "all":
        return load()
    if train == "fold0":
        d = load()
        return d[folds(d) != 0].reset_index(drop=True)
    if train.startswith("n"):
        return train_set(int(train[1:]))
    raise ValueError(f"unknown training set {train!r}")


def folds(d):
    """Fold index (0-4) of every row of d, from the fixed split in data/folds.csv (never recomputed here).

    The split was made once (2026-11-16) over all 512 base geometries with
    StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED) on the 3-class label ('unknown' as its own stratum),
    and saved so every method uses the same one."""
    f = pd.read_csv(FOLDS).set_index("geom_id").fold
    out = d.geom_id.map(f)
    if out.isna().any():
        raise KeyError(f"geometries without a fold in {FOLDS.name}: {list(d.geom_id[out.isna()])[:5]}")
    return out.to_numpy()


def queries(qset=None):
    """Inverse-design queries (columns: set, geom_id, theta2, u2, logE, ansys40), optionally one set."""
    q = pd.read_csv(QUERIES)
    return q if qset is None else q[q.set == qset].reset_index(drop=True)


# ------------------------------------------------------------------ force curves
_CURVE_CACHE = {}


def _curve_file(name):
    if name not in _CURVE_CACHE:
        with gzip.open(CURVES[name], "rt") as f:
            _CURVE_CACHE[name] = json.load(f)
    return _CURVE_CACHE[name]


def load_curves(ids, sources=("base", "ext")):
    """Force-displacement curves {geom_id: {state: (u, F)}} for the given geometries.

    base = the 512 base geometries (the shared viewer file), ext = the 1024 extra training geometries (of the final
    and first-attempt curves the one that reached further, u_applied_mm and fx_disp_N, no origin; the same rule as
    the viewer file), test = the 256 test geometries."""
    allc = {}
    for s in sources:
        allc.update(_curve_file(s))
    return {g: {s: (np.array(allc[g][s]["u"]), np.array(allc[g][s]["F"])) for s in ("partial", "full")} for g in ids}
