"""Model engine of the CABD demo app: geometry, forward prediction, deformation frames and inverse design.

Models (fitted on all 1536 training geometries, base 512 + ext1 + ext2, by cabd.evaluate.learning_curve; bundled by
cabd.app.bundle; nothing is retrained here). Per output the model with the lowest test error in the learning curve:
  theta2, u2     : MLP deep ensemble, 5 seeds pooled (25 members)
  log10 dE_r     : ROM + GP regressor
  stability      : GP two-stage classifier, P(partial bi) and P(full mono | partial bi)
  inverse        : the same GP as surrogate + SLSQP with the project objective (cabd.inverse.core), here restricted
                   to the targets the user gave; no time cap, the 20 starts run in parallel
Deformation frames are a kinematic illustration, not a simulation: rigid arms (the ROM kinematics of
cabd.models.rom) whose rotation is scaled so that at the predicted u2 the arm has turned by the predicted theta2; the
flanges bend as cantilevers following the tip pivot; hinge outlines blend between the bodies they join.
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .. import config
from ..data import LO, HI, to_phys
from ..geometry import params as G
from ..inverse import core as INV
from ..models import predictor
from ..models import rom
from ..models.gp import p_true
from ..models.mlp import ensemble
from ..models.rom_gp import rom_inputs
from .bundle import BUNDLE, load as load_bundle

X = config.X_COLS
TAU = INV.TAU
P_MIN, K = INV.P_MIN, INV.K
U_MAX = 15.0
N_JOBS = 10                          # parallel SLSQP starts of one inverse request
LABELS = {"t_h": ("Tip hinge thickness", "mm"), "L_h": ("Tip hinge length", "mm"), "t_a": ("Apex hinge thickness", "mm"),
          "L_a": ("Apex hinge length", "mm"), "L": ("Arm length", "mm"), "t_f": ("Flange thickness", "mm"),
          "c": ("Apex-to-wall gap", "mm"), "rho": ("Fillet ratio", "-")}


def params(x):
    p = dict(zip(X, map(float, x)))
    p["t_w"] = G.T_W
    return p


def check(x):
    """(ok, list of violated constraints / bounds as user-facing text)."""
    try:
        cons = G.check_constraints(params(x))
    except Exception as e:  # degenerate input
        return False, [f"invalid geometry ({e})"]
    bad = [k.replace(" (assumption)", "").replace("MIN_FEATURE", "0.1 mm") for k, v in cons.items() if not v]
    for c, v in zip(X, x):
        lo, hi = config.BOUNDS[c]
        if not lo - 1e-9 <= v <= hi + 1e-9:
            bad.insert(0, f"{c} outside {lo}-{hi}")
    return not bad, bad


def densify(P, step=0.35):
    out = []
    n = len(P)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * t for t in np.arange(k) / k]
    return np.array(out)


def bodies(x, n=12, dense=False):
    """Closed outlines of the bodies, duplicate points removed (dense: resampled every <= 0.35 mm for bending)."""
    out = {}
    for name, (kind, P) in G.bodies(params(x), n=n).items():
        keep = [P[0]]
        for q in P[1:]:
            if np.linalg.norm(q - keep[-1]) > 1e-9:
                keep.append(q)
        if np.linalg.norm(keep[-1] - keep[0]) < 1e-9:
            keep.pop()
        P = np.array(keep)
        out[name] = (kind, densify(P) if dense else P)
    return out


def geometry_payload(x):
    ok, bad = check(x)
    out = {"ok": ok, "violations": bad}
    try:
        b = bodies(x)
        d = G.derived(params(x))
        out["bodies"] = [{"name": k, "kind": kind, "pts": np.round(P, 4).tolist()} for k, (kind, P) in b.items()]
        out["derived"] = {"x_W": d["x_W"], "x_i": d["x_i"], "x_D": d["x_D"], "Z_o": d["Z_o"], "H": d["H"], "z_F": d["z_F"]}
    except Exception:
        out["bodies"] = []
    return out


# ------------------------------------------------------------------ deformation frames (kinematic illustration)
def frames(x, u2, theta2, theta_rom, us):
    p = params(x)
    d = G.derived(p)
    geo = rom.geometry(pd.Series(p))
    _, _, ang = rom.curve(geo, "partial", theta_rom)              # ROM arm rotation (deg) on rom.U_GRID
    rot_rom = np.interp(us, rom.U_GRID, ang)
    at_u2 = float(np.interp(u2, rom.U_GRID, ang))
    s = float(np.clip(theta2 / at_u2, 0.5, 2.0)) if at_u2 > 1e-6 else 1.0
    A0 = np.array([-(p["L"] - p["t_a"] / 2), 0.0])                  # apex pivot at the apex-hinge centre (drawing)
    T0 = np.array([0.0, d["H"] + p["L_h"] / 2])                     # tip pivot (ROM)
    R = np.linalg.norm(T0 - A0)
    phi0 = np.arctan2(T0[1] - A0[1], T0[0] - A0[0])
    B = bodies(x, n=12, dense=True)
    x_i, z_F, H, L_a = d["x_i"], d["z_F"], d["H"], p["L_a"]

    out = []
    for u, r in zip(us, rot_rom):
        dphi = np.radians(s * r)
        A = A0 + [u, 0.0]
        T = A + R * np.array([np.cos(phi0 + dphi), np.sin(phi0 + dphi)])
        delta = T[1] - T0[1]                                         # flange tip deflection
        cR, sR = np.cos(dphi), np.sin(dphi)

        def arm_up(P):
            Q = P - A0
            return A + np.column_stack([cR * Q[:, 0] - sR * Q[:, 1], sR * Q[:, 0] + cR * Q[:, 1]])

        def arm_lo(P):
            return arm_up(P * [1, -1]) * [1, -1]

        def flange(P):
            sx = np.clip((P[:, 0] - x_i) / (0 - x_i), 0, 1)
            w = delta * sx ** 2 * (3 - sx) / 2
            return P + np.column_stack([np.zeros(len(P)), np.sign(P[:, 1]) * w])

        def blend(P, f0, f1, w):
            return (1 - w)[:, None] * f0(P) + w[:, None] * f1(P)

        frame = []
        for name, (kind, P) in B.items():
            if name == "frame":
                Q = flange(P)
            elif name == "arm_upper":
                Q = arm_up(P)
            elif name == "arm_lower":
                Q = arm_lo(P)
            elif name == "hinge_tip_upper":
                Q = blend(P, arm_up, flange, np.clip((P[:, 1] - H) / (z_F - H), 0, 1))
            elif name == "hinge_tip_lower":
                Q = blend(P, arm_lo, flange, np.clip((-P[:, 1] - H) / (z_F - H), 0, 1))
            else:  # apex hinge
                Q = blend(P, arm_lo, arm_up, np.clip((P[:, 1] + L_a / 2) / L_a, 0, 1))
            frame.append({"name": name, "kind": kind, "pts": np.round(Q, 3).tolist()})
        out.append({"u": round(float(u), 4), "rot": round(float(s * r), 3), "bodies": frame,
                    "probe": np.round([A.tolist(), T.tolist()], 3).tolist()})
    return out, s


# ------------------------------------------------------------------ engine
class Engine:
    def __init__(self, path=BUNDLE):
        t0 = time.time()
        b = load_bundle(path)
        self.mlps, self.rl, self.gp, self.n_train = b["mlp"], b["rom_logE"], b["gp"], b["n_train"]
        self.theta_rom = self.rl["theta"]
        self.load_s = time.time() - t0

    # forward ------------------------------------------------------
    def classify(self, U):
        """P(partial bi) and P(full mono | partial bi) from the GP classifier, U in [0, 1]^8."""
        return p_true(self.gp["c1"], U), p_true(self.gp["c2"], U)

    def regress(self, U):
        """mean and sd of (theta2, u2, log10 dE_r): MLP pool for the first two, ROM+GP for log dE."""
        Xs = (U * 2 - 1).astype(np.float32)
        mus, vs = [], []
        for m in self.mlps:
            mu, sd = ensemble("reg", m["reg"], Xs)
            mus.append(mu * m["sd"] + m["mean"])
            vs.append((sd * m["sd"]) ** 2)
        mus, vs = np.array(mus), np.array(vs)
        mean = mus.mean(0)
        sd = np.sqrt((vs + mus ** 2).mean(0) - mean ** 2)
        r = self.rl
        B = rom_inputs(rom.features_x(to_phys(U), r["theta"]))
        mE, sE = r["reg"].predict(np.column_stack([U, (B - r["b_mean"]) / r["b_sd"]]), return_std=True)
        mean[:, 2], sd[:, 2] = mE, sE
        return mean, sd

    def forward(self, x, with_frames=True):
        x = np.asarray(x, float)
        ok, bad = check(x)
        if not ok:
            return {"ok": False, "violations": bad}
        U = ((x - LO) / (HI - LO))[None]
        mu, sd = self.regress(U)
        p1, p2 = self.classify(U)
        p1, p2 = float(p1[0]), float(p2[0])
        partial_bi, full_mono = p1 >= 0.5, p2 >= 0.5
        res = {"ok": True,
               "partial": {"bistable": bool(partial_bi), "p_bistable": p1},
               "full": {"bistable": bool(partial_bi and not full_mono), "p_monostable_given_partial_bi": p2,
                        "p_bistable": p1 * (1 - p2)},
               "p_target_combination": p1 * p2,
               "combination_ok": bool(partial_bi and full_mono),
               "theta2": {"mean": float(mu[0, 0]), "sd": float(sd[0, 0])},
               "u2": {"mean": float(mu[0, 1]), "sd": float(sd[0, 1])},
               "dE": {"mean": float(10 ** mu[0, 2]), "lo": float(10 ** (mu[0, 2] - sd[0, 2])),
                      "hi": float(10 ** (mu[0, 2] + sd[0, 2])), "log10_mean": float(mu[0, 2]), "log10_sd": float(sd[0, 2])}}
        if with_frames and res["combination_ok"]:
            u2, th2 = res["u2"]["mean"], res["theta2"]["mean"]
            us = np.unique(np.r_[np.linspace(0, U_MAX, 61), u2])
            fr, s = frames(x, u2, th2, self.theta_rom, us)
            res["frames"] = fr
            res["stable_index"] = int(np.argmin([abs(f["u"] - u2) for f in fr]))
            res["kinematic_scale"] = s
            res["x_D"] = float(G.derived(params(x))["x_D"])           # apex-hinge inner face, where u is applied
        return res

    # inverse ------------------------------------------------------
    def inverse(self, targets, n_start=INV.N_START, jobs=N_JOBS):
        """targets: dict theta2 (deg), u2 (mm), dE (mJ); None = not constrained."""
        t0 = time.time()
        keys = ["theta2", "u2", "dE"]
        mask = np.array([targets.get(k) is not None for k in keys])
        if not mask.any():
            return {"ok": False, "message": "Enter at least one target value."}
        t = np.array([targets.get("theta2") or 0.0, targets.get("u2") or 0.0,
                      np.log10(targets["dE"]) if targets.get("dE") else 0.0])
        starts = INV.starts(n_start)
        runs = Parallel(n_jobs=jobs, backend="loky")(delayed(_solve)(self.gp, t, mask, x0) for x0 in starts)
        ends = np.array([r[0] for r in runs])
        mu, sd, pbm = predictor(self.gp)(ends)
        e = np.sqrt((mu - t) ** 2 + sd ** 2) / TAU
        emax = np.where(mask, e, 0).max(1)
        feas = np.array([INV.feasible(x) for x in ends])
        valid = feas & (pbm >= P_MIN - 1e-6)
        out = {"ok": True, "time_s": round(time.time() - t0, 1), "n_start": n_start, "n_valid": int(valid.sum()),
               "targets": {k: targets.get(k) for k in keys}}
        if not valid.any():
            out.update(achievable=False, message="No start reached a manufacturable geometry with P(bi→mono) ≥ 0.9.")
            return out
        idx = np.where(valid)[0]
        best = idx[np.argmin(emax[idx])]
        x = to_phys(ends[best])
        acceptable = idx[emax[idx] <= 1.0]
        out.update(achievable=bool(emax[best] <= 1.0), n_acceptable=int(len(acceptable)), max_error_tau=float(emax[best]),
                   p_target_combination=float(pbm[best]),
                   x=x.tolist(),
                   gp={k: {"mean": float(mu[best, j]), "sd": float(sd[best, j])} for j, k in enumerate(["theta2", "u2", "logE"])},
                   tolerance=self.tolerance(ends[best], t, mask))
        if len(acceptable) > 1:
            out["spread"] = to_phys(ends[acceptable]).std(0).tolist()
        if not out["achievable"]:
            out["message"] = (f"Not achievable within tolerance: the closest design misses a target by "
                              f"{emax[best]:.2f} τ (needs ≤ 1).")
        out["forward"] = self.forward(x)
        return out

    def tolerance(self, u_best, t, mask, n=81):
        """Per dimension, how far it can move (others fixed) while every targeted output's GP mean stays within tau.

        The stability probability is reported separately: the optimum usually sits on the P = 0.9 constraint, so
        including it would make most tolerances one-sided zero."""
        pred = predictor(self.gp)
        res = []
        for i in range(8):
            grid = np.unique(np.r_[np.linspace(0, 1, n), u_best[i]])
            Ug = np.repeat(u_best[None], len(grid), 0)
            Ug[:, i] = grid
            mu, _, _ = pred(Ug)
            okm = np.where(mask, np.abs(mu - t) / TAU, 0).max(1) <= 1
            okm &= np.array([INV.feasible(u) if o else False for u, o in zip(Ug, okm)])   # manufacturable shapes only
            j0 = int(np.argmin(np.abs(grid - u_best[i])))
            lo = hi = j0
            if okm[j0]:
                while lo > 0 and okm[lo - 1]:
                    lo -= 1
                while hi < len(grid) - 1 and okm[hi + 1]:
                    hi += 1
            span = HI[i] - LO[i]
            res.append({"minus": float((grid[j0] - grid[lo]) * span), "plus": float((grid[hi] - grid[j0]) * span),
                        "ok_at_best": bool(okm[j0])})
        return res


def _solve(gp_model, t, mask, x0):
    """One SLSQP start of an app inverse request (objective over the targeted outputs only)."""
    import warnings
    warnings.filterwarnings("ignore")
    batch = INV.make_batch(predictor(gp_model), t, lambda mu, sd: INV.objective_masked(mu, sd, t, mask))
    r = INV.slsqp(batch, x0)
    return np.clip(r.x, 0, 1), float(r.fun)


# Default shape: training geometry TR0051 rounded to 0.01. Among the Ansys bi->mono geometries it is one of the closest
# to the centre of the design space (the exact centre is bistable in both states), and the models agree with Ansys there
# (1536-geometry models: P(partial bi) 0.98, P(full mono | bi) 1.00; theta2 71.0 vs 71.4 deg, u2 13.17 vs 13.21 mm).
DEFAULT = {"t_h": 0.29, "L_h": 3.27, "t_a": 0.38, "L_a": 2.57, "L": 6.94, "t_f": 1.29, "c": 2.92, "rho": 0.50}


def meta():
    return {"vars": [{"key": c, "label": LABELS[c][0], "unit": LABELS[c][1], "min": config.BOUNDS[c][0],
                      "max": config.BOUNDS[c][1], "default": DEFAULT[c]} for c in X],
            "tau": {"theta2": float(TAU[0]), "u2": float(TAU[1]), "log10_dE": float(TAU[2])},
            "u_max": U_MAX,
            "models": {"regression": "θ₂, u₂: MLP deep ensemble (25) · ΔE: ROM + GP", "classification": "GP two-stage classifier",
                       "inverse": "GP surrogate + SLSQP (20 Sobol starts)", "n_train": 1536}}
