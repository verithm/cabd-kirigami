"""Parametric cross-section of the kirigami bistable particle, v3 (8 independent variables).
Ported unchanged from Research/sampling/param_geom_v3.py (the STEP round-trip check of its __main__ is in cabd.geometry.check_step).
v3 vs v2: hinge thicknesses 0.1-0.4, L 5-8, c 2-4, t_w fixed at 3 mm, common fillet ratio rho is a variable again.

Datum: x = 0 is the outer face of the tip hinges (right end of the sheet), z = 0 the symmetry plane.
The base (back-wall outer face, where the fixed support acts) is FIXED at full length 2*Z_o = 25.4 mm.

Independent variables (box-samplable):
  t_h  tip-hinge ligament thickness               [mm]
  L_h  tip-hinge length (= slot width)            [mm]
  t_a  apex-hinge ligament thickness              [mm]
  L_a  apex-hinge length                          [mm]
  L    arm span: datum x = 0 -> apex outer face   [mm]
  t_f  flange thickness                           [mm]
  rho  common fillet ratio (all three fillet families) [-]
  c    clearance apex outer face -> back-wall inner face [mm]
Constants:
  Z_o  = 12.7 mm  (half base length)          b = 6 mm (extrusion depth)
  t_w  = 3 mm (back-wall thickness, fixed)
  fillet radii from the variable rho:
         R_slot = rho*L_h/2  (tip-hinge slot fillets, both ends)
         R_up   = rho*L_h/2  (arm-tip rounding C2, tangent to x=0 and to the hypotenuse)
         R_low  = rho*L_a/2  (apex rounding C1, tangent to the apex-hinge inner face and the hypotenuse)
Derived:
  tan(alpha) = (Z_o - t_f - L_h - t_h - L_a/2) / (L - t_a)       alpha = arm inner-edge angle
"""
import numpy as np

Z_O = 12.7
T_W = 3.0
MIN_FEATURE = 0.1  # mm, printable minimum (assumption, not measured)
B_DEPTH = 6.0
NODES_PER_MM2 = 524.0
NODE_LIMIT = 128_000

VAR_NAMES = ["t_h", "L_h", "t_a", "L_a", "L", "t_f", "c", "rho"]
REF = dict(t_h=0.3, L_h=3.0, t_a=0.3, L_a=3.0, L=7.0, t_f=1.2, c=3.0, rho=0.8 / 1.5, t_w=T_W)
RANGES = {
    "t_h": (0.1, 0.4), "L_h": (2.0, 4.0), "t_a": (0.1, 0.4), "L_a": (2.0, 4.0),
    "L": (5.0, 8.0), "t_f": (0.8, 1.8), "c": (2.0, 4.0), "rho": (0.3, 0.8),
}
VARIANTS = {  # STEP files that have Z_o = 12.7 and are therefore reproducible
    "bistable_ref_1.2mm.stp":       dict(REF),
    "bistable_ref_1.2mm_0.2mm.stp": dict(REF, t_h=0.2, t_a=0.2),
}
NOT_REPRODUCIBLE = {  # base length differs (Z_o = 12.3 / 13.0) -> excluded by design
    "bistable_ref_0.8mm.stp": 12.3, "bistable_ref_1.5mm.stp": 13.0, "bistable_ref.stp": 13.0,
}


def derived(p):
    d = {}
    num = Z_O - p["t_f"] - p["L_h"] - p["t_h"] - p["L_a"] / 2      # = (L - t_a) * tan(alpha)
    den = p["L"] - p["t_a"]
    d["tan_alpha"] = num / den
    d["alpha"] = np.degrees(np.arctan2(num, den))
    a = np.radians(d["alpha"])
    d["k"] = np.tan((np.pi / 2 - a) / 2)                          # tangent-length / radius in a (90+alpha) deg corner
    d["R_slot"] = p["rho"] * p["L_h"] / 2
    d["R_up"] = p["rho"] * p["L_h"] / 2
    d["R_low"] = p["rho"] * p["L_a"] / 2
    d["d_up"] = d["R_up"] * d["k"]                                 # tangent offset of C2 along x = 0 and the hypotenuse
    d["d_low"] = d["R_low"] * d["k"]                               # tangent offset of C1 along x = -L+t_a and the hypotenuse
    d["z_i"] = p["L_a"] / 2 + den * d["tan_alpha"]                 # hypotenuse intercept on x = 0
    d["H"] = d["z_i"] + p["t_h"]                                   # arm top face = tip-hinge bottom
    d["z_F"] = d["H"] + p["L_h"]                                   # flange inner face
    d["Z_o"] = d["z_F"] + p["t_f"]                                 # == Z_O by construction
    d["x_i"] = -(p["L"] + p["c"])                                  # back-wall inner face
    d["x_W"] = d["x_i"] - p["t_w"]                                 # back-wall outer face (fixed support)
    d["x_D"] = -p["L"] + p["t_a"]                                  # apex-hinge inner face (displaced face)
    d["W"] = -d["x_W"]                                             # overall width
    d["L_fl"] = p["L"] + p["c"]                                    # flange cantilever length
    d["l_tip"] = p["L_h"] - 2 * d["R_slot"]                        # straight ligament, tip hinges  = L_h (1-rho)
    d["l_apex"] = p["L_a"] - 2 * d["d_low"]                        # straight ligament, apex hinge  = L_a (1-rho k)
    d["l_hyp"] = den / np.cos(a) - d["d_low"] - d["d_up"]          # straight part of the hypotenuse
    d["h_arm"] = d["H"] - p["L_a"] / 2                             # arm root height at x = -L
    d["z_c2"] = d["z_i"] + d["d_up"]                               # C2 tangent point on x = 0
    # Is the C2 tangent point on x = 0 above the arm/hinge split line z = H ?  (d_up >= t_h)
    #   yes: arm top face ends where circle C2 crosses z = H (x_t <= 0), hinge bottom follows the arc up to (0, z_c2)
    #   no : outline leaves the circle at (0, z_c2) and runs straight up x = 0 -> arm tip has a vertical flat
    d["tip_cross"] = d["d_up"] >= p["t_h"]
    if d["tip_cross"]:
        d["x_t"] = -d["R_up"] + np.sqrt(max(d["R_up"] ** 2 - (d["z_c2"] - d["H"]) ** 2, 0.0))
    else:
        d["x_t"] = 0.0
    d["tip_flat"] = 0.0 if d["tip_cross"] else p["t_h"] - d["d_up"]
    d["lig_margin"] = p["t_h"] + d["R_slot"] - d["d_up"]           # >0: C2 tangent point below start of straight ligament
    return d


def arc(c, R, a0, a1, n=16):
    t = np.radians(np.linspace(a0, a1, n))
    return np.c_[c[0] + R * np.cos(t), c[1] + R * np.sin(t)]


def bodies(p, n=16):
    """dict name -> (material kind, Nx2 polygon, CCW not guaranteed) for the 6 bodies."""
    d = derived(p); a = d["alpha"]; ar = np.radians(a)
    t_h, L_h, t_a, L_a, L = p["t_h"], p["L_h"], p["t_a"], p["L_a"], p["L"]
    Rs, Ru, Rl, du, dl = d["R_slot"], d["R_up"], d["R_low"], d["d_up"], d["d_low"]
    z_i, H, z_F, Z_o, x_i, x_W = (d[k] for k in ["z_i", "H", "z_F", "Z_o", "x_i", "x_W"])

    frame = np.array([[x_W, -Z_o], [0, -Z_o], [0, -z_F], [x_i, -z_F], [x_i, z_F], [0, z_F], [0, Z_o], [x_W, Z_o]])

    # C2: arm-tip rounding, tangent to x = 0 (at z_c2) and to the hypotenuse (at T2)
    C2 = np.array([-Ru, d["z_c2"]])
    ang_T2 = -(90 - a)
    # C1: apex rounding, tangent to x = -L + t_a (at z = L_a/2 - d_low) and to the hypotenuse (at T1)
    C1 = np.array([-L + t_a + Rl, L_a / 2 - dl])
    x_a = C1[0] - np.sqrt(max(Rl ** 2 - dl ** 2, 0.0))             # where C1 crosses z = L_a/2
    ang_a = np.degrees(np.arctan2(L_a / 2 - C1[1], x_a - C1[0]))
    ang_T1 = 90 + a

    if d["tip_cross"]:
        ang_t = np.degrees(np.arctan2(H - C2[1], d["x_t"] - C2[0]))
        tip_arm = [arc(C2, Ru, ang_T2, ang_t, n), [[-L, H]]]
        tip_hinge = [[[d["x_t"], H]], arc(C2, Ru, ang_t, 0, n)]
    else:  # small rounding: arm keeps a vertical flat on x = 0 between z_c2 and H
        tip_arm = [arc(C2, Ru, ang_T2, 0, n), [[0, H]], [[-L, H]]]
        tip_hinge = [[[0, H]]]

    arm_u = np.vstack([[[-L, L_a / 2]], arc(C1, Rl, ang_a, ang_T1, n), *tip_arm])
    hinge_tip_u = np.vstack([
        [[0, z_F], [-(t_h + Rs), z_F]],
        arc([-(t_h + Rs), z_F - Rs], Rs, 90, 0, n),
        arc([-(t_h + Rs), H + Rs], Rs, 0, -90, n),
        *tip_hinge,
    ])
    hinge_apex = np.vstack([
        [[-L, -L_a / 2]],
        arc(C1 * [1, -1], Rl, -ang_a, -180, n),
        arc(C1, Rl, 180, ang_a, n),
        [[-L, L_a / 2]],
    ])
    mirror = lambda P: (P * [1, -1])[::-1]
    return {
        "frame": ("frame", frame),
        "arm_upper": ("frame", arm_u),
        "arm_lower": ("frame", mirror(arm_u)),
        "hinge_tip_upper": ("hinge", hinge_tip_u),
        "hinge_tip_lower": ("hinge", mirror(hinge_tip_u)),
        "hinge_apex": ("hinge", hinge_apex),
    }


def poly_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def total_area(p):
    return sum(poly_area(P) for _, P in bodies(p).values())


def est_nodes(p):
    return total_area(p) * NODES_PER_MM2


# ---------------------------------------------------------------- geometric validity (no shapely needed)
def _dedup(P, tol=1e-9):
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > tol:
            keep.append(i)
    if np.linalg.norm(P[keep[-1]] - P[keep[0]]) <= tol and len(keep) > 1:
        keep.pop()
    return P[keep]


def _segments(P):
    return P, np.roll(P, -1, axis=0)


def _proper_crossings(A0, A1, B0, B1, eps=1e-9):
    """count proper (interior-interior) crossings between segment sets A and B (vectorised)."""
    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    a0, a1 = A0[:, None, :], A1[:, None, :]
    b0, b1 = B0[None, :, :], B1[None, :, :]
    o1, o2 = orient(a0, a1, b0), orient(a0, a1, b1)
    o3, o4 = orient(b0, b1, a0), orient(b0, b1, a1)
    cross = (o1 * o2 < -eps) & (o3 * o4 < -eps)
    return cross


def self_intersects(P):
    P = _dedup(P)
    A0, A1 = _segments(P)
    cross = _proper_crossings(A0, A1, A0, A1)
    return bool(cross.any())


def _inside_strict(pts, P, eps=1e-7):
    from matplotlib.path import Path
    return Path(P).contains_points(pts, radius=-eps)


def bodies_overlap(polys):
    """True if any two polygons overlap with positive area (proper edge crossing or a vertex strictly inside)."""
    names = list(polys)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            P, Q = _dedup(polys[names[i]]), _dedup(polys[names[j]])
            if (P[:, 0].max() < Q[:, 0].min() or Q[:, 0].max() < P[:, 0].min() or
                    P[:, 1].max() < Q[:, 1].min() or Q[:, 1].max() < P[:, 1].min()):
                continue
            if _proper_crossings(*_segments(P), *_segments(Q)).any():
                return True, (names[i], names[j], "edge crossing")
            if _inside_strict(P, Q).any() or _inside_strict(Q, P).any():
                return True, (names[i], names[j], "vertex inside")
    return False, None


def check_constraints(p, n=24):
    """dict of name -> bool. All must be True for a valid, printable, meshable section."""
    d = derived(p)
    bd = bodies(p, n=n)
    polys = {k: v[1] for k, v in bd.items()}
    ov, _ = bodies_overlap(polys)
    cons = {
        "0<alpha<90 (numerator>0)": 0 < d["alpha"] < 90,
        "l_tip>0": d["l_tip"] > 0,
        "l_apex>0": d["l_apex"] > 0,
        "l_hyp>0 (straight hypotenuse)": d["l_hyp"] > 0,
        "x_t>=-t_h (C2 stays right of ligament inner face)": (not d["tip_cross"]) or d["x_t"] >= -p["t_h"] - 1e-9,
        "lig_margin>=0 (C2 tangent below straight ligament)": d["lig_margin"] >= -1e-9,
        "t_h+R_slot<L": p["t_h"] + d["R_slot"] < p["L"],
        "c>0": p["c"] > 0,
        "h_arm>0": d["h_arm"] > 0,
        "min feature>=MIN_FEATURE (assumption)": min_feature(p, d) >= MIN_FEATURE - 1e-9,
        "nodes<=128k": est_nodes(p) <= NODE_LIMIT,
        "no self-intersection": not any(self_intersects(P) for P in polys.values()),
        "no body overlap": not ov,
    }
    return cons


def min_feature(p, d=None):
    d = d or derived(p)
    return min(p["t_h"], p["t_a"], p["t_f"], p["t_w"], p["c"], p["L_h"], p["L_a"], d["l_tip"], d["l_apex"])
