"""GP forward model.

  regression: 3 independent GPs (constant x ARD Matern 5/2 + white noise), outputs theta2, u2, log10 dE_r
  classification: 2 binary GP classifiers (Laplace), P1 = P(partial bi), P2 = P(full mono | partial bi)
      P(bi>mono) = P1 P2, P(bi>bi) = P1 (1 - P2), P(mono>mono) = 1 - P1
  tuning budget: 20 marginal-likelihood optimizations per model (1 + 19 restarts), the same 20 trials as the other methods
  intervals (CV only, cabd.evaluate.cv): normalized split-conformal |y - mu| / sd, calibrated by an inner 5-fold inside
      the training folds (kernel hyperparameters fixed from the outer fit), so the outer test fold is never used.
Inputs: design variables on [0, 1].
"""
import warnings

import numpy as np

from .. import data as D
from ..config import Y_COLS
from ..metrics import conformal_q

RESTARTS = 19


def reg_model(seed, n_dim=8):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(n_dim), (1e-2, 1e3), nu=2.5) + WhiteKernel(1e-3, (1e-8, 1e0))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=RESTARTS, random_state=seed)


def cls_model(seed, n_dim=8):
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    k = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(np.ones(n_dim), (1e-2, 1e3), nu=2.5)
    return GaussianProcessClassifier(k, n_restarts_optimizer=RESTARTS, random_state=seed)


def conformal_scale(X, y, kernel, seed):
    """Inner 5-fold normalized residuals with the outer fit's kernel held fixed -> conformal quantile q."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.model_selection import KFold
    s = np.empty(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        g = GaussianProcessRegressor(kernel, normalize_y=True, optimizer=None).fit(X[tr], y[tr])
        m, sd = g.predict(X[te], return_std=True)
        s[te] = np.abs(y[te] - m) / sd
    return conformal_q(s)


def p_true(c, X):
    """P(class True) of a binary classifier."""
    return c.predict_proba(X)[:, list(c.classes_).index(True)]


def three_class(p1, p2):
    """Two-stage probabilities -> (n, 3) over CLASSES (bi>bi, bi>mono, mono>mono)."""
    return np.column_stack([p1 * (1 - p2), p1 * p2, 1 - p1])


def fit(d, seed):
    """Fit on every row of d (regression on d.is_reg rows, classifiers on labelled rows)."""
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    X = D.scale_x(d)
    r = d.is_reg.to_numpy()
    reg = [reg_model(seed).fit(X[r], d.loc[r, c].to_numpy(float)) for c in Y_COLS]
    ok, is_bi, y1, y2 = D.class_targets(d)
    return {"kind": "gp", "reg": reg, "c1": cls_model(seed).fit(X[ok], y1), "c2": cls_model(seed).fit(X[is_bi], y2)}


def predict_unit(m, U, X=None):
    """U: (n, 8) inputs on [0, 1] -> mu (n, 3), sd (n, 3), P (n, 3). X (physical) is not needed."""
    ms = [g.predict(U, return_std=True) for g in m["reg"]]
    mu, sd = np.column_stack([a for a, _ in ms]), np.column_stack([b for _, b in ms])
    return mu, sd, three_class(p_true(m["c1"], U), p_true(m["c2"], U))
