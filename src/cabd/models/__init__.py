"""Forward models (GP, MLP ensemble, ROM + GP, referees) and the cVAE generator, with one API.

Model kinds and modules
  gp            cabd.models.gp       3 GP regressors + two-stage GP classifier, inputs on [0, 1]
  mlp           cabd.models.mlp      5-member deep ensemble (mean + log-variance) + 5-member classifier, inputs on [-1, 1]
  rom_gp        cabd.models.rom_gp   GP on design variables + ROM features (cabd.models.rom), two-stage GP classifier
  referee       cabd.models.referee  stacked poly + LightGBM + GP(RBF + linear); referee_nogp: poly + LightGBM only
  cvae          cabd.models.cvae     conditional VAE generator (not a forward model)
  rom_theta     cabd.models.rom      the 5 fitted ROM factors of one training set

A fitted forward model is a dict with "kind" (and the parts listed in each module); a cVAE is a
cabd.models.cvae.Generator.

Public API
  fit(kind, d, seed, theta=None) -> model
      d: training rows (cabd.data.load(), training_data('all' | 'fold0' | 'n512' | 'n1024' | 'n1536'), ...).
      rom_gp needs theta = the ROM factors fitted on the same rows (cabd.models.rom.fit_theta(d)["theta"]).
      cvae trains on the partial-bi rows of d.
  predict(model, X) -> (mu (n, 3), sd (n, 3) or None, P (n, 3))
      X: (n, 8) physical design variables in config.X_COLS order (array or DataFrame).
      mu, sd: theta2 [deg], u2 [mm], log10 dE_r [mJ] (config.Y_COLS); sd is None for the referees.
      P: class probabilities over config.CLASSES = ["bi>bi", "bi>mono", "mono>mono"].
  predict_unit(model, U) -> same, for U (n, 8) on [0, 1] (the inverse-design space, cabd.data.scale_x / to_phys).
      Bit-identical to the legacy inverse.predictor path (ROM features from to_phys(U)).
  predictor(model) -> f(U) -> (mu, sd, P(bi>mono))   exactly legacy inverse.predictor: sd = zeros for the referees.
  load(name) / save(model, name, root)   (cabd.models.io; name e.g. "mlp_n1536_s0", "gp_all_s3", "rom_theta_n1536")
  Generator.sample(target, n=200, seed=0) -> (n, 8) on [0, 1];  Generator.candidates(target, feasible, seed) -> (k, 8)
      target = (theta2, u2, logE); feasible(u) -> bool for one design on [0, 1].
"""
import numpy as np

from .io import exists, load, locate, model_name, save  # noqa: F401

FORWARD = ("gp", "mlp", "rom_gp", "referee", "referee_nogp")
KINDS = FORWARD + ("cvae",)


def module(kind):
    from . import cvae, gp, mlp, referee, rom_gp
    return {"gp": gp, "mlp": mlp, "rom_gp": rom_gp, "referee": referee, "referee_nogp": referee, "cvae": cvae}[kind]


def kind_of(model):
    return model["kind"] if isinstance(model, dict) else model.kind


def fit(kind, d, seed, theta=None):
    """Fit a model of the given kind on every row of d."""
    if kind == "rom_gp":
        if theta is None:
            raise ValueError("rom_gp needs the ROM factors theta fitted on the same rows (rom.fit_theta)")
        return module(kind).fit(d, seed, theta)
    if kind in ("referee", "referee_nogp"):
        from .referee import MEMBERS
        return module(kind).fit(d, seed, MEMBERS[kind])
    return module(kind).fit(d, seed)


def predict_unit(model, U, X=None):
    """U: (n, 8) on [0, 1] -> (mu, sd or None, P). X: the same points in physical units, if already known."""
    return module(kind_of(model)).predict_unit(model, np.asarray(U, float), X)


def predict(model, X):
    """X: (n, 8) physical design variables -> (mu, sd or None, P)."""
    from ..data import scale_x
    import pandas as pd
    from ..config import X_COLS
    X = X[X_COLS].to_numpy(float) if isinstance(X, pd.DataFrame) else np.asarray(X, float)
    return predict_unit(model, scale_x(X), X)


def predictor(model):
    """Inverse-design predictor: f(U) -> (mu, sd, P(bi>mono)) with sd = 0 where the model has none (referees)."""
    def f(U):
        mu, sd, P = predict_unit(model, U)
        return mu, np.zeros_like(mu) if sd is None else sd, P[:, 1]
    return f
