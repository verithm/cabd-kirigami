"""Reduced-order model: rigid arms + rotational hinge springs + flange bending spring (one degree of freedom).

Half model (upper arm, the section is symmetric about z = 0):
  apex pivot  A = (x_a, 0),  x_a = x_a0 + u      (the apex is pushed in +x by the applied displacement u)
  tip pivot   T = (0, z_t)                      (end of the flange cantilever; it can move only in z)
  rigid arm   |T - A| = R  ->  z_t = sqrt(R^2 - x_a^2)
  energy U(u) = 2 * [ 1/2 k_f (z_t - z_t0)^2 + 1/2 k_tip dphi^2 ] + 1/2 k_apex (2 dphi)^2
      dphi = arm rotation from the initial angle; the apex hinge bends by the rotation of both arms
  force F = dU/du (numerical), the same landmark rules as the Ansys curves give bi/mono, u2, theta2, dE_r

Stiffness (beam formulas, b = 6 mm extrusion depth):
  k_tip  = c_tip  * E_hinge b t_h^3 / (12 l_tip)      l_tip  = straight tip-hinge ligament  = L_h (1 - rho)
  k_apex = c_apex * E_hinge b t_a^3 / (12 l_apex)     l_apex = straight apex-hinge ligament
  k_f    = c_f    * 3 E_frame (b t_f^3 / 12) / L_fl^3  L_fl   = flange cantilever length = L + c
  partial: E_hinge = E_wet, E_frame = E_dry ;  full: both E_wet   (E_dry 369.5, E_wet 12.7 MPa, Ansys template)
  geometry: pivots at the hinge centres, x_a0 = -(L - t_a/2) * g, z_t0 = H + L_h/2
  lateral path of the tip pivot: flange k_f in series with the tip hinge's lateral spring k_hl (see curve())
Five fitted factors theta = (log c_tip, log c_apex, log c_f, log c_hl, g) are shared by both states and fitted to the
force-displacement curves of the training geometries (fit()). The fitted factors are saved as rom_theta_<train>
({"kind": "rom_theta", "theta": [5 floats], "rms": float}).
"""
import numpy as np
import pandas as pd

from ..config import X_COLS
from ..geometry import params as G

E_DRY, E_WET, B = 369.5, 12.7, 6.0
U_GRID = np.linspace(0, 15, 301)


def geometry(row):
    p = {k: float(row[k]) for k in X_COLS}
    p["t_w"] = G.T_W
    d = G.derived(p)
    return dict(t_h=p["t_h"], t_a=p["t_a"], t_f=p["t_f"], L=p["L"], L_h=p["L_h"],
                l_tip=d["l_tip"], l_apex=d["l_apex"], L_fl=d["L_fl"], z_t0=d["H"] + p["L_h"] / 2)


def curve(geo, state, theta):
    """U(u), F(u) and the arm rotation |dphi| [deg] on U_GRID. theta = (log c_tip, log c_apex, log c_f, log c_hl, g).

    The tip pivot moves outward against the flange bending spring in series with the tip hinge's own
    lateral spring k_hl = c_hl E_wet b t_h / l_tip (the hinge is always wet). Added 2026-11-16 after the
    first fit: without it every full-state curve came out monostable."""
    c_tip, c_apex, c_f, c_hl = np.exp(theta[:4])
    g = theta[4]
    e_h = E_WET
    e_f = E_DRY if state == "partial" else E_WET
    k_tip = c_tip * e_h * B * geo["t_h"] ** 3 / (12 * geo["l_tip"])
    k_apex = c_apex * e_h * B * geo["t_a"] ** 3 / (12 * geo["l_apex"])
    k_f = c_f * 3 * e_f * (B * geo["t_f"] ** 3 / 12) / geo["L_fl"] ** 3
    k_hl = c_hl * e_h * B * geo["t_h"] / geo["l_tip"]
    k_lat = 1 / (1 / k_f + 1 / k_hl)
    x0 = -(geo["L"] - geo["t_a"] / 2) * g
    z0 = geo["z_t0"]
    R = np.hypot(x0, z0)
    xa = x0 + U_GRID
    zt = np.sqrt(np.maximum(R ** 2 - xa ** 2, 0.0))
    phi = np.arctan2(zt, -xa)                      # arm angle measured from -x (initial value phi0 < 90 deg)
    dphi = phi - phi[0]
    U = 2 * (0.5 * k_lat * (zt - z0) ** 2 + 0.5 * k_tip * dphi ** 2) + 0.5 * k_apex * (2 * dphi) ** 2
    F = np.gradient(U, U_GRID)
    return U, F, np.degrees(np.abs(dphi))


def landmarks(U, F, ang, u=U_GRID):
    """Same rules as the Ansys curves (class from the sign of F, u2 where F comes back above 0).

    No negative force: mono, u2 / theta2 at the force minimum, logE = -6. Negative to the end: neg_end."""
    neg = np.where(F[1:] < 0)[0] + 1
    if not len(neg):
        i = int(np.argmin(F[1:])) + 1
        return dict(cls="mono", u2=u[i], theta2=ang[i], logE=-6.0, Fmin=F[1:].min())
    a = neg[0]
    u1 = u[a - 1] + (u[a] - u[a - 1]) * F[a - 1] / (F[a - 1] - F[a])
    back = next((k for k in range(a, len(F)) if F[k] > 0), None)
    if back is None:
        return dict(cls="neg_end", u2=u[-1], theta2=ang[-1], logE=np.log10(max(np.interp(u1, u, U) - U[-1], 1e-6)), Fmin=F.min())
    w = F[back - 1] / (F[back - 1] - F[back])
    u2 = u[back - 1] + w * (u[back] - u[back - 1])
    dE = np.interp(u1, u, U) - np.interp(u2, u, U)
    return dict(cls="bi", u2=u2, theta2=np.interp(u2, u, ang), logE=np.log10(max(dE, 1e-6)), Fmin=F.min())


def fit(d, ids, curves, theta0=(0.0, 0.0, 0.0, -5.0, 1.0)):
    """Least squares on the force curves of the given geometries, each curve scaled by its own max |F|.

    curves: {geom_id: {state: (u, F)}} (cabd.data.load_curves). Returns (theta (5,), rms of the scaled residuals)."""
    from scipy.optimize import least_squares
    geos = {g: geometry(d.set_index("geom_id").loc[g]) for g in ids}

    def resid(theta):
        r = []
        for g in ids:
            for s in ("partial", "full"):
                u, f = curves[g][s]
                _, F, _ = curve(geos[g], s, theta)
                r.append((np.interp(u, U_GRID, F) - f) / max(np.abs(f).max(), 1e-3))
        return np.concatenate(r)
    res = least_squares(resid, np.array(theta0, float), bounds=([-8, -8, -8, -8, 0.5], [8, 8, 8, 8, 1.5]))
    return res.x, float(np.sqrt(np.mean(res.fun ** 2)))


def fit_theta(d):
    """ROM factors fitted on every geometry of d (curves from data/curves) -> {"kind": "rom_theta", "theta", "rms"}."""
    from ..data import load_curves
    theta, rms = fit(d, list(d.geom_id), load_curves(d.geom_id))
    return {"kind": "rom_theta", "theta": theta, "rms": rms}


def features(d, theta):
    """ROM predictions for every row of d (needs X_COLS and geom_id): class, u2, theta2, logE and min force, both states."""
    rows = []
    for _, r in d.iterrows():
        geo, out = geometry(r), {"geom_id": r.geom_id}
        for s in ("partial", "full"):
            for k, v in landmarks(*curve(geo, s, theta)).items():
                out[f"rom_{s}_{k}"] = v
        rows.append(out)
    return pd.DataFrame(rows)


def features_x(X, theta):
    """features() for an (n, 8) array of physical design variables (geom_id q0, q1, ...)."""
    df = pd.DataFrame(X, columns=X_COLS)
    df["geom_id"] = [f"q{i}" for i in range(len(df))]
    return features(df, theta)


def kinematic_theta2(d, u2, theta):
    """Arm rotation [deg] of the partial-state ROM at the displacement u2, per row of d (joint_output 'kin' variant)."""
    return np.array([np.interp(u, U_GRID, curve(geometry(r), "partial", theta)[2]) for (_, r), u in zip(d.iterrows(), u2)])
