"""Bundle the models the demo app uses into one file: models/app_models_n1536.joblib.

Choice per output (learning-curve test results, 1536 training geometries, results/learning_curve/metrics.csv):
  theta2, u2     MLP deep ensemble, the 5 seeds pooled into one 25-member ensemble (the 5-seed mean was lowest)
  log10 dE_r     ROM+GP regressor (lowest)                      - only its logE GP and the ROM factors are kept
  stability      GP two-stage classifier (best 3-class and bi>mono balanced accuracy)
  inverse        GP surrogate (the same GP as the classifier, so the inverse constraint and the forward verdict agree)
Note: these choices were made on the test set, so the test numbers of the chosen combination are no longer an
untouched estimate.

Sources (cabd.models.load): mlp_n1536_s0..s4, rom_gp_n1536_s0, gp_n1536_s0 (the latter two from artifacts/models).
Format (only numpy / scikit-learn objects and plain Python containers are pickled; joblib compress=3):
  {"format": 1, "n_train": 1536,
   "mlp": 5 x {"reg": 5 x state_dict as {name: numpy array}, "cfg_r": config of cabd.models.mlp.net, "mean", "sd"},
   "rom_logE": {"theta", "b_mean", "b_sd", "reg": GaussianProcessRegressor of log10 dE_r},
   "gp": the gp model dict (cabd.models.gp: "kind", "reg", "c1", "c2")}
load() rebuilds the torch networks. Keep the file below 50 MB (GitHub warns above that); the build reports its size.

usage: python -m cabd.app.bundle [--out models/app_models_n1536.joblib]
"""
import argparse
from pathlib import Path

import numpy as np

from .. import config

BUNDLE = config.MODELS / "app_models_n1536.joblib"
N_TRAIN = 1536
SEEDS = range(5)


def build():
    from ..models import load
    mlps = [load(f"mlp_n{N_TRAIN}_s{s}") for s in SEEDS]
    rg = load(f"rom_gp_n{N_TRAIN}_s0")
    gp = load(f"gp_n{N_TRAIN}_s0")
    return {"format": 1, "n_train": N_TRAIN,
            "mlp": [{"reg": [{k: v.detach().cpu().numpy().copy() for k, v in n.state_dict().items()} for n in m["reg"]],
                     "cfg_r": m["cfg_r"], "mean": np.asarray(m["mean"]), "sd": np.asarray(m["sd"])} for m in mlps],
            "rom_logE": {"theta": np.asarray(rg["theta"]), "b_mean": rg["b_mean"], "b_sd": rg["b_sd"], "reg": rg["reg"][2]},
            "gp": gp}


def load(path=BUNDLE):
    """The bundle with the MLP networks rebuilt: mlp = 5 x {"reg": [torch nets], "mean", "sd"}."""
    import joblib
    import torch
    from ..models.mlp import net
    b = joblib.load(path)
    for m in b["mlp"]:
        nets = []
        for sd in m["reg"]:
            n = net(6, m["cfg_r"])
            n.load_state_dict({k: torch.from_numpy(v) for k, v in sd.items()})
            n.eval()
            nets.append(n)
        m["reg"] = nets
    return b


def main():
    import joblib
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=BUNDLE)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(build(), a.out, compress=3)
    mb = a.out.stat().st_size / 1e6
    print(f"wrote {a.out} ({mb:.1f} MB)" + ("  WARNING: above 50 MB" if mb > 50 else ""))


if __name__ == "__main__":
    main()
