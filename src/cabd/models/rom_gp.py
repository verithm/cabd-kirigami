"""ROM + GP forward model.

  ROM: cabd.models.rom (rigid arms + springs), its 5 factors fitted to the force curves of the training rows only
  regression (revised 2026-11-16, "A"): GP (same kernel family and budget as the GP model) on y directly, with the
      ROM's partial-state predictions of theta2, u2, log10 dE_r as 3 extra inputs (11 inputs, ARD), standardized on
      the training rows. Where the ROM finds no second stable point, its value at the force minimum is used
      (logE floored at LOGE_FLOOR). The earlier residual version (GP on y - ROM, original project
      results/rom_gp_v1_residual/, not carried over) did much worse: 5-fold CV combined nRMSE 0.674 vs 0.196 for this
      version (mean of 5 seeds).
  classification: the two GP classifiers of the GP model with one more input each,
      the ROM's minimum force (partial for stage 1, full for stage 2), standardized on the training rows
  intervals (CV only): same inner-5-fold normalized conformal as the GP model
Inputs: design variables on [0, 1] (+ ROM features computed from the physical design variables).
"""
import warnings

import numpy as np

from .. import data as D
from ..config import Y_COLS
from . import rom
from .gp import cls_model, p_true, reg_model, three_class

LOGE_FLOOR = -4.2


def rom_inputs(f):
    """(n, 3) ROM partial-state theta2, u2, logE (floored) from rom.features()."""
    return np.column_stack([f.rom_partial_theta2, f.rom_partial_u2, np.maximum(f.rom_partial_logE, LOGE_FLOOR)])


def fit(d, seed, theta):
    """Fit on every row of d with ROM factors theta (fitted beforehand on the same rows, rom.fit_theta)."""
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    theta = np.asarray(theta, float)
    f = rom.features(d, theta)
    X = D.scale_x(d)
    r = d.is_reg.to_numpy()
    B = rom_inputs(f)
    b_mean, b_sd = B[r].mean(0), B[r].std(0)
    XR = np.column_stack([X, (B - b_mean) / b_sd])
    reg = [reg_model(seed, XR.shape[1]).fit(XR[r], d.loc[r, c].to_numpy(float)) for c in Y_COLS]
    ok, is_bi, y1, y2 = D.class_targets(d)
    fp, ff = f.rom_partial_Fmin.to_numpy(float), f.rom_full_Fmin.to_numpy(float)
    zp = (fp[ok].mean(), fp[ok].std())
    zf = (ff[is_bi].mean(), ff[is_bi].std())
    c1 = cls_model(seed, 9).fit(np.column_stack([X, (fp - zp[0]) / zp[1]])[ok], y1)
    c2 = cls_model(seed, 9).fit(np.column_stack([X, (ff - zf[0]) / zf[1]])[is_bi], y2)
    return {"kind": "rom_gp", "theta": theta, "reg": reg, "b_mean": b_mean, "b_sd": b_sd, "c1": c1, "c2": c2,
            "zp": zp, "zf": zf}


def inputs(m, U, X=None, f=None):
    """GP inputs (regression XR, stage-1 X1, stage-2 X2) for unit inputs U and the same points' ROM features f
    (default: rom.features of the physical design variables X)."""
    r = rom.features_x(X, m["theta"]) if f is None else f
    XR = np.column_stack([U, (rom_inputs(r) - m["b_mean"]) / m["b_sd"]])
    X1 = np.column_stack([U, (r.rom_partial_Fmin.to_numpy(float) - m["zp"][0]) / m["zp"][1]])
    X2 = np.column_stack([U, (r.rom_full_Fmin.to_numpy(float) - m["zf"][0]) / m["zf"][1]])
    return XR, X1, X2


def predict_unit(m, U, X=None):
    """U: (n, 8) on [0, 1]; X: the same points in physical units (default to_phys(U)) -> mu, sd, P (n, 3) each."""
    XR, X1, X2 = inputs(m, U, D.to_phys(U) if X is None else X)
    ms = [g.predict(XR, return_std=True) for g in m["reg"]]
    mu, sd = np.column_stack([a for a, _ in ms]), np.column_stack([b for _, b in ms])
    return mu, sd, three_class(p_true(m["c1"], X1), p_true(m["c2"], X2))
