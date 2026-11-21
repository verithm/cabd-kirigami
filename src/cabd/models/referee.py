"""Referee surrogate: scores every inverse-design candidate so that no method grades itself.

Same budget as the four methods: the same 512 training geometries, the same 5 folds, 20 tuning trials.
Members (different families from the methods):
  poly  degree-3 polynomial on [-1, 1] inputs with LassoCV (sparse polynomial surrogate)
  gbm   LightGBM with shallow trees, 20 random configurations scored on an inner hold-out
  gpx   GP with an additive "RBF (ARD) + linear" kernel (the GP method uses Matern 5/2), 1 + 19 restarts
Stacking: non-negative weights from inner 5-fold out-of-fold predictions (NNLS for regression,
  log-loss over a weighted mean of probabilities for each classification stage).
Classification is two-stage like the GP method: P1 = P(partial bi), P2 = P(full mono | partial bi).
Intervals: absolute-residual split-conformal on the stacked inner out-of-fold predictions (Stacked.q).
The "no_gp" variant (referee_nogp) drops the gpx member (used to check whether GP-like methods are favored).
The referee gives no predictive sd: predict() returns sd = None (the inverse objective uses sd = 0 for it).
Inputs: design variables on [-1, 1] (float64).
"""
import warnings

import numpy as np

from .. import data as D
from ..config import Y_COLS
from ..metrics import conformal_q
from .gp import three_class

TRIALS = 20
MEMBERS = {"referee": ["poly", "gbm", "gpx"], "referee_nogp": ["poly", "gbm"]}
GBM_SPACE = {"num_leaves": [4, 7, 15], "max_depth": [2, 3, 4], "learning_rate": [0.02, 0.05, 0.1],
             "min_child_samples": [5, 10, 20], "subsample": [0.7, 0.9, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
             "reg_lambda": [0.0, 1.0, 10.0]}


# ------------------------------------------------------------------ members
def gbm_configs(seed):
    rng = np.random.default_rng(seed)
    return [{k: v[rng.integers(len(v))] for k, v in GBM_SPACE.items()} for _ in range(TRIALS)]


def fit_gbm(X, y, seed, kind):
    """Pick the best of 20 configurations on an inner 20 % hold-out (early stopping), refit on all."""
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    va, tr = perm[: len(X) // 5], perm[len(X) // 5:]
    Model = lgb.LGBMRegressor if kind == "reg" else lgb.LGBMClassifier
    best = None
    for cfg in gbm_configs(seed):
        m = Model(n_estimators=2000, subsample_freq=1, random_state=seed, verbose=-1, **cfg)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        score = m.best_score_["valid_0"]["l2" if kind == "reg" else "binary_logloss"]
        if best is None or score < best[0]:
            best = (score, cfg, m.best_iteration_ or 2000)
    _, cfg, n = best
    return Model(n_estimators=n, subsample_freq=1, random_state=seed, verbose=-1, **cfg).fit(X, y)


def fit_poly(X, y, seed, kind):
    from sklearn.linear_model import LassoCV, LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    if kind == "reg":
        return make_pipeline(PolynomialFeatures(3), StandardScaler(), LassoCV(cv=5, max_iter=20000, random_state=seed)).fit(X, y)
    return make_pipeline(PolynomialFeatures(3), StandardScaler(),
                         LogisticRegressionCV(Cs=10, cv=5, max_iter=5000, scoring="neg_log_loss")).fit(X, y)


def fit_gpx(X, y, seed, kind):
    from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel
    if kind == "reg":
        k = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.ones(8), (1e-2, 1e3)) + ConstantKernel(0.1, (1e-4, 1e2)) * DotProduct(1.0, (1e-3, 1e2))
             + WhiteKernel(1e-3, (1e-8, 1e0)))
        return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=19, random_state=seed).fit(X, y)
    k = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(np.ones(8), (1e-2, 1e3)) + ConstantKernel(0.1, (1e-4, 1e2)) * DotProduct(1.0, (1e-3, 1e2))
    return GaussianProcessClassifier(k, n_restarts_optimizer=19, random_state=seed).fit(X, y)


FIT = {"poly": fit_poly, "gbm": fit_gbm, "gpx": fit_gpx}


def member_predict(m, X, kind):
    return m.predict(X) if kind == "reg" else m.predict_proba(X)[:, 1]


# ------------------------------------------------------------------ stacking
def stack_weights(P, y, kind):
    """P: (n, k) member out-of-fold predictions -> non-negative weights summing to 1."""
    from scipy.optimize import minimize, nnls
    if kind == "reg":
        w, _ = nnls(P, y)
        return w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])
    P = np.clip(P, 1e-6, 1 - 1e-6)

    def ll(z):
        w = np.exp(z) / np.exp(z).sum()
        p = np.clip(P @ w, 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    z = minimize(ll, np.zeros(P.shape[1]), method="Nelder-Mead").x
    return np.exp(z) / np.exp(z).sum()


class Stacked:
    """Stacked members for one target. kind 'reg' (one output) or 'cls' (one binary stage)."""

    def __init__(self, members, kind, seed):
        self.members, self.kind, self.seed = members, kind, seed

    def fit(self, X, y):
        from sklearn.model_selection import KFold, StratifiedKFold
        n = len(X)
        oof = np.zeros((n, len(self.members)))
        split = (KFold(5, shuffle=True, random_state=self.seed).split(X) if self.kind == "reg"
                 else StratifiedKFold(5, shuffle=True, random_state=self.seed).split(X, y))
        for tr, te in split:
            for j, name in enumerate(self.members):
                oof[te, j] = member_predict(FIT[name](X[tr], y[tr], self.seed, self.kind), X[te], self.kind)
        self.w = stack_weights(oof, y.astype(float), self.kind)
        stacked = oof @ self.w
        self.q = conformal_q(np.abs(y - stacked)) if self.kind == "reg" else None
        self.models = [FIT[name](X, y, self.seed, self.kind) for name in self.members]
        return self

    def predict(self, X):
        return np.column_stack([member_predict(m, X, self.kind) for m in self.models]) @ self.w


# ------------------------------------------------------------------ model API
def fit(d, seed, members=("poly", "gbm", "gpx")):
    """Fit on every row of d. members: MEMBERS['referee'] (default) or MEMBERS['referee_nogp']."""
    warnings.filterwarnings("ignore")
    members = list(members)
    X = D.scale_x(d) * 2 - 1
    r = d.is_reg.to_numpy()
    reg = [Stacked(members, "reg", seed).fit(X[r], d.loc[r, c].to_numpy(float)) for c in Y_COLS]
    ok, is_bi, y1, y2 = D.class_targets(d)
    return {"kind": "referee" if "gpx" in members else "referee_nogp", "reg": reg,
            "c1": Stacked(members, "cls", seed).fit(X[ok], y1.astype(int)),
            "c2": Stacked(members, "cls", seed).fit(X[is_bi], y2.astype(int))}


def predict_unit(m, U, X=None):
    """U: (n, 8) on [0, 1] -> mu (n, 3), sd None, P (n, 3)."""
    Xs = U * 2 - 1
    mu = np.column_stack([s.predict(Xs) for s in m["reg"]])
    return mu, None, three_class(m["c1"].predict(Xs), m["c2"].predict(Xs))
