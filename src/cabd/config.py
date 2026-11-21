"""Paths and constants shared by every step of the pipeline.

All paths are relative to the repository root, so the repo works wherever it is cloned.
Large artifacts (GP / referee weights for every training size and seed, raw Ansys result zips) are not in git:
they are published as GitHub Release assets and unpacked into ARTIFACTS (default <repo>/artifacts, git-ignored;
override with the environment variable CABD_ARTIFACTS).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"                 # datasets, sampling tables, queries, folds (small, in git)
MODELS = ROOT / "models"             # small weights kept in git (app bundle, MLP, cVAE, ROM factors)
RESULTS = ROOT / "results"           # metrics, predictions, inverse results, figures (in git)
ANSYS_KIT = ROOT / "ansys_kit"       # files copied into every Ansys collection / verification kit
ARTIFACTS = Path(os.environ.get("CABD_ARTIFACTS", ROOT / "artifacts"))   # Release assets, not in git
BUILD = ROOT / "build"               # regenerable outputs (STEP files, kit folders, kit zips), not in git

# ----------------------------------------------------------------- design variables and outputs
X_COLS = ["t_h", "L_h", "t_a", "L_a", "L", "t_f", "c", "rho"]
BOUNDS = {"t_h": (0.1, 0.4), "L_h": (2, 4), "t_a": (0.1, 0.4), "L_a": (2, 4),
          "L": (5, 8), "t_f": (0.8, 1.8), "c": (2, 4), "rho": (0.3, 0.8)}
Y_COLS = ["theta2", "u2", "logE"]                                  # theta2 [deg], u2 [mm], log10 dE_r [mJ]
Y_WEIGHTS = {"theta2": 0.25, "u2": 0.25, "logE": 0.5}             # theta2 and u2 correlate (r = 0.93)
CLASSES = ["bi>bi", "bi>mono", "mono>mono"]                         # (partial state > full state)

# ----------------------------------------------------------------- evaluation protocol
N_FOLDS, FOLD_SEED = 5, 20261116
SEEDS = [0, 1, 2, 3, 4]
ALPHA = 0.10                                                        # 90% intervals
TRAIN_SIZES = [512, 1024, 1536]

# ----------------------------------------------------------------- inverse design rules
TAU = [1.0049735794420784, 0.1382365916643131, 0.14554776403125505]   # tolerance per output (theta2, u2, logE)
P_MIN = 0.9                                                         # required P(bi>mono)

# ----------------------------------------------------------------- geometry sampling and Ansys collection (added by the
# geometry / Ansys port: cabd.geometry, cabd.ansys, cabd.verify)
SAMPLING = DATA / "sampling"         # Sobol parameter tables params_*.csv and params_meta.json
# raw Ansys result zips collect_result_PC<label>_<date>_<time>.zip (Release asset, read-only inputs; scrubbed copies of
# the kits' collect_result_PC<label>_<computer>_<date>_<time>.zip, see cabd.ansys.scrub)
ANSYS_RAW = Path(os.environ.get("CABD_ANSYS_RAW", ARTIFACTS / "ansys_raw"))
# python with cadquery for STEP export (separate environment); default: the running interpreter
CAD_PYTHON = os.environ.get("CABD_CAD_PYTHON", "")
