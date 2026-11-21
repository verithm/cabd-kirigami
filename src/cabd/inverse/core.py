"""Inverse-design rules (fixed 2026-11-16, see the training plan doc): objective, optimizer, candidate selection.

GP, MLP, ROM+GP (the same optimizer for all three):
  objective  f(x) = smooth max_j e_j,  e_j = sqrt((mu_j - t_j)^2 + sd_j^2) / tau_j   (j = theta2, u2, logE)
             smooth max = log-sum-exp with sharpness K (assumption: K = 20, error of the max <= ln 3 / K = 0.055)
             tau = max(Ansys 90 % repeatability, referee CV RMSE) per output (config.TAU)
  validity   P(bi>mono)(x) >= P_MIN = 0.9, as an inequality constraint (no penalty weight)
  optimizer  SLSQP (maxiter 200, ftol 1e-6) in the [0, 1]^8 box from the same N_START = 20 scrambled Sobol points
             (seed SOBOL_SEED) for every query and method; gradients by forward differences (step H, backward at the
             upper bound) evaluated as one batch of 9 points; each start runs until convergence, the whole query is
             capped at TIME_CAP seconds (starts not begun by then are skipped)
cVAE: N_GEN samples for (target, full_mono = 1), clustered, medoids of the largest clusters (cabd.models.cvae).
All methods: candidates outside the geometry constraints (feasible) are dropped; up to N_CAND candidates at least
  MIN_DIST apart ([0, 1] units), ranked by the method's own objective (cVAE: cluster size). Rank 1 is the answer.

Known weakness of the time cap: it is wall-clock time (time.time()), checked only before each start. A start that
begins before the cap runs to convergence, so a query can take longer than TIME_CAP, and anything that stretches wall
time (CPU contention from parallel jobs, the machine sleeping: on 2026-11-19 Windows Modern Standby with the lid
closed was counted as query time) cuts the number of starts and can change the candidates. The result files record
starts_done per query; a query with starts_done < N_START was cut. cabd.inverse.patch reruns such queries.
"""
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from .. import config
from ..data import to_phys

TAU = np.array(config.TAU)          # tolerance per output (theta2, u2, logE)
P_MIN = config.P_MIN
K = 20.0                            # smooth-max sharpness
N_START = 20                        # SLSQP starts per query
N_CAND = 5                          # candidates kept per query
MIN_DIST = 0.1                      # minimum distance between kept candidates ([0, 1] units)
TIME_CAP = 300.0                    # wall-clock seconds per query, checked before each start
SOBOL_SEED = 20261116
H = 1e-4                            # finite-difference step ([0, 1] units)
SLSQP_OPTIONS = {"maxiter": 200, "ftol": 1e-6}


def feasible(u):
    """True if u (8,) on [0, 1] is inside the box and the geometry satisfies every constraint (cabd.geometry.params)."""
    from ..geometry import params as G
    if np.any(u < -1e-9) or np.any(u > 1 + 1e-9):
        return False
    p = dict(zip(config.X_COLS, to_phys(u)))
    p["t_w"] = G.T_W
    try:
        return all(G.check_constraints(p).values())
    except Exception:
        return False


def objective_values(mu, sd, t):
    """Smooth max over the outputs of the tau-normalized errors, one value per row of mu / sd."""
    e = np.sqrt((mu - t) ** 2 + sd ** 2) / TAU
    return np.log(np.exp(K * e).sum(1)) / K


def objective_masked(mu, sd, t, mask):
    """Objective of the demo app, where only the targeted outputs (mask) count; the log-sum-exp is shifted by its
    maximum (numerically stable). With every output targeted it equals objective_values up to rounding."""
    e = np.sqrt((mu - t) ** 2 + sd ** 2) / TAU
    ez = np.where(mask, K * e, -np.inf)
    return np.log(np.exp(ez - ez.max(1, keepdims=True)).sum(1)) / K + ez.max(1) / K


def starts(n=N_START):
    """The n scrambled Sobol start points (the same for every query and method)."""
    return qmc.Sobol(8, scramble=True, seed=SOBOL_SEED).random(n)


def make_batch(pred, t, objective=None):
    """Objective, gradient, constraint and constraint gradient at x from one batch prediction of 9 points.

    pred(U) -> (mu, sd, P(bi>mono)); objective(mu, sd) -> values (default objective_values with target t).
    Forward differences with step H (backward at the upper bound). The last x is cached, so SLSQP's separate calls
    for fun / jac / constraint at the same x cost one prediction."""
    obj = objective or (lambda mu, sd: objective_values(mu, sd, t))
    cache = {}

    def batch(x):
        key = x.tobytes()
        if key not in cache:
            step = np.where(x + H <= 1, H, -H)          # backward difference at the upper bound
            P = np.vstack([x, x + np.diag(step)])
            mu, sd, pbm = pred(P)
            fv = obj(mu, sd)
            cache.clear()
            cache[key] = (fv[0], (fv[1:] - fv[0]) / step, pbm[0] - P_MIN, (pbm[1:] - pbm[0]) / step)
        return cache[key]
    return batch


def slsqp(batch, x0):
    """One SLSQP run from x0 in the [0, 1]^8 box with the P(bi>mono) >= P_MIN constraint."""
    return minimize(lambda x: batch(x)[0], x0, jac=lambda x: batch(x)[1], method="SLSQP", bounds=[(0, 1)] * 8,
                    constraints=[{"type": "ineq", "fun": lambda x: batch(x)[2], "jac": lambda x: batch(x)[3]}],
                    options=dict(SLSQP_OPTIONS))


def select(X, score):
    """Feasible candidates by ascending score, at least MIN_DIST apart, at most N_CAND. Returns indices into X."""
    out = []
    for i in np.argsort(score):
        if len(out) == N_CAND:
            break
        if all(np.linalg.norm(X[i] - X[j]) >= MIN_DIST for j in out) and feasible(X[i]):
            out.append(i)
    return out


def optimize_query(pred, t, n_start=N_START, time_cap=TIME_CAP):
    """All starts for one target t (3,) -> ([(x, objective, P(bi>mono)) per candidate], info).

    info: time (s), starts_done, starts_valid (ends with P >= P_MIN), best_any (lowest objective of any end).
    The wall-clock cap is checked before each start (see the module docstring for its weakness)."""
    t0 = time.time()
    batch = make_batch(pred, t)
    ends, vals, pbms, n_done = [], [], [], 0
    for x0 in starts(n_start):
        if time.time() - t0 > time_cap:
            break
        r = slsqp(batch, x0)
        x = np.clip(r.x, 0, 1)
        fv, _, g, _ = batch(x)
        ends.append(x), vals.append(fv), pbms.append(g + P_MIN)
        n_done += 1
    ends, vals, pbms = np.array(ends), np.array(vals), np.array(pbms)
    valid = pbms >= P_MIN - 1e-6
    idx = np.where(valid)[0]
    chosen = idx[select(ends[idx], vals[idx])] if len(idx) else []
    return [(ends[i], vals[i], pbms[i]) for i in chosen], dict(time=time.time() - t0, starts_done=n_done,
                                                              starts_valid=int(valid.sum()), best_any=float(vals.min()))


def cvae_query(gen, t, seed):
    """cVAE candidates for target t, in the same (candidates, info) form as optimize_query (no objective)."""
    t0 = time.time()
    cand = gen.candidates(t, feasible, seed=seed)
    res = [(x, np.nan, np.nan) for x in cand]
    return res, dict(time=time.time() - t0, starts_done=0, starts_valid=len(cand), best_any=np.nan)
