"""Saving and loading fitted models.

Names follow the fit that produced them:
  {kind}_{train}_s{seed}   kind: gp, mlp, rom_gp, referee, referee_nogp, cvae;
                           train: all (512 base geometries), fold0 (folds 1-4), n512 / n1024 / n1536 (learning curve)
  rom_theta_{train}        ROM factors fitted on that training set (no seed)
Formats (no pickled module or class from outside numpy / scikit-learn / scipy / lightgbm / torch / cabd):
  gp, rom_gp, referee(_nogp)  {name}.joblib   dict of scikit-learn / lightgbm estimators (+ cabd.models.referee.Stacked)
  mlp, cvae                   {name}.pt (torch state_dicts, loads with weights_only=True) + {name}.json (config, scaling)
  rom_theta                   {name}.json
Search order on load: config.MODELS (small weights kept in git), then config.ARTIFACTS / "models" (GitHub Release
asset, unpacked by the user). tools/convert_legacy_models.py and models/MANIFEST.csv list where each model lives.
"""
import json
import re
from pathlib import Path

import numpy as np

from .. import config

EXT = {"gp": [".joblib"], "rom_gp": [".joblib"], "referee": [".joblib"], "referee_nogp": [".joblib"],
       "mlp": [".pt", ".json"], "cvae": [".pt", ".json"], "rom_theta": [".json"]}
NAME = re.compile(r"^(gp|mlp|rom_gp|referee_nogp|referee|cvae|rom_theta)_(all|fold0|n\d+)(?:_s(\d+))?$")


def search_dirs():
    return [config.MODELS, config.ARTIFACTS / "models"]


def parse(name):
    """name -> (kind, train, seed or None)."""
    m = NAME.match(name)
    if not m:
        raise ValueError(f"not a model name: {name!r}")
    return m.group(1), m.group(2), None if m.group(3) is None else int(m.group(3))


def model_name(kind, train, seed=None):
    return f"{kind}_{train}" + ("" if seed is None else f"_s{seed}")


def files(name, root):
    kind = parse(name)[0]
    return [Path(root) / f"{name}{e}" for e in EXT[kind]]


def locate(name):
    """Folder that holds the model's files (MODELS first, then ARTIFACTS/models)."""
    for root in search_dirs():
        if all(p.exists() for p in files(name, root)):
            return root
    raise FileNotFoundError(
        f"model {name!r} is neither in {config.MODELS} nor in {config.ARTIFACTS / 'models'}. The big models "
        "(GP, ROM+GP, referees, every training size and seed) are not in git: download the models asset of the "
        "GitHub Release and unpack it so that artifacts/models/ exists (or point CABD_ARTIFACTS to its parent folder), "
        "or refit with `python -m cabd.fit`.")


def exists(name):
    try:
        locate(name)
        return True
    except FileNotFoundError:
        return False


def _kind(model):
    return model["kind"] if isinstance(model, dict) else model.kind


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def save(model, name, root):
    """Write the model under root in its format. Returns the written paths."""
    import joblib
    kind = _kind(model)
    assert parse(name)[0] == kind, f"{name} does not match model kind {kind}"
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = files(name, root)
    if kind in ("gp", "rom_gp", "referee", "referee_nogp"):
        joblib.dump(model, paths[0])
    elif kind == "rom_theta":
        paths[0].write_text(json.dumps(_jsonable(model), indent=1), encoding="utf-8")
    elif kind == "mlp":
        import torch
        torch.save({"reg": [n.state_dict() for n in model["reg"]], "cls": [n.state_dict() for n in model["cls"]]}, paths[0])
        meta = {k: model.get(k) for k in ("kind", "cfg_r", "cfg_c", "mean", "sd", "q")}
        paths[1].write_text(json.dumps(_jsonable(meta), indent=1), encoding="utf-8")
    elif kind == "cvae":
        import torch
        torch.save(model.model.state_dict(), paths[0])
        meta = {"kind": "cvae", **{k: getattr(model, k) for k in ("cfg", "mean", "sd", "epochs", "score", "seed")}}
        paths[1].write_text(json.dumps(_jsonable(meta), indent=1), encoding="utf-8")
    else:
        raise ValueError(kind)
    return paths


def load(name, root=None):
    """Load a model by name (from root, or from MODELS / ARTIFACTS/models)."""
    import joblib
    kind = parse(name)[0]
    paths = files(name, root if root is not None else locate(name))
    if kind in ("gp", "rom_gp", "referee", "referee_nogp"):
        return joblib.load(paths[0])
    meta = json.loads(paths[-1].read_text(encoding="utf-8"))
    if kind == "rom_theta":
        return {"kind": "rom_theta", "theta": np.array(meta["theta"]), "rms": meta["rms"]}
    import torch
    state = torch.load(paths[0], weights_only=True)
    if kind == "mlp":
        from .mlp import net
        reg, cls = [], []
        for sd_, out, n_out, cfg in [(state["reg"], reg, 6, meta["cfg_r"]), (state["cls"], cls, 3, meta["cfg_c"])]:
            for s in sd_:
                m = net(n_out, cfg)
                m.load_state_dict(s)
                out.append(m)
        return {"kind": "mlp", "reg": reg, "cls": cls, "mean": np.array(meta["mean"]), "sd": np.array(meta["sd"]),
                "cfg_r": meta["cfg_r"], "cfg_c": meta["cfg_c"],
                "q": None if meta.get("q") is None else np.array(meta["q"])}
    from .cvae import CVAE, Generator
    g = Generator()
    g.cfg, g.epochs, g.score, g.seed = meta["cfg"], meta["epochs"], meta["score"], meta["seed"]
    g.mean, g.sd = np.array(meta["mean"]), np.array(meta["sd"])
    g.model = CVAE(g.cfg)
    g.model.load_state_dict(state)
    return g
