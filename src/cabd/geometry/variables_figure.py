"""One-page summary of the v3 design variables: dimensioned reference section, two hinge insets and the range table.
Labels are Korean (team figure). Ranges and the realised alpha / W / node ranges come from the base train + test tables.

usage: python -m cabd.geometry.variables_figure [--out results/figures/geometry_design_variables_v3.png]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon, Rectangle
import pandas as pd

from cabd.config import RESULTS, SAMPLING
from cabd.geometry.params import RANGES, REF, T_W, bodies, derived

FRAME, HINGE, VAR, DER, FIX = "#c9d3df", "#f2b48a", "#d62728", "#7a7a7a", "#111111"


def fresh_style(fn):
    """Draw with matplotlib's default style (as the original one-figure scripts did in a fresh process), whatever
    another figure of the same process changed; the global style is restored afterwards."""
    import functools

    @functools.wraps(fn)
    def run(*a, **kw):
        import matplotlib
        with matplotlib.rc_context():
            matplotlib.rcdefaults()
            return fn(*a, **kw)
    return run


def dim(ax, p0, p1, text, color, off=(0, 0), inward=False, fs=15, gap=1.0):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    if inward:
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        for a, s in ((p0, -1), (p1, 1)):
            ax.annotate("", xy=a, xytext=a + s * gap * u, arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
    else:
        ax.annotate("", xy=p0, xytext=p1, arrowprops=dict(arrowstyle="<|-|>", color=color, lw=2, shrinkA=0, shrinkB=0))
    if text:
        ax.text(*((p0 + p1) / 2 + off), text, color=color, fontsize=fs, fontweight="bold", ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9))


def ext(ax, a, b, color):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=0.8, ls=":")


@fresh_style
def draw(out):
    tables = pd.concat([pd.read_csv(SAMPLING / f"params_{r}.csv") for r in ("train", "test")])
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    p = dict(REF)
    d = derived(p)
    L, t_h, L_h, t_a, L_a = p["L"], p["t_h"], p["L_h"], p["t_a"], p["L_a"]
    H, z_F, Z_o, x_i, x_W = d["H"], d["z_F"], d["Z_o"], d["x_i"], d["x_W"]

    def shape(ax):
        for kind, P in bodies(p, n=60).values():
            ax.add_patch(Polygon(P, closed=True, fc=FRAME if kind == "frame" else HINGE, ec="#55606e", lw=0.8))

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 0.75, 1.2], height_ratios=[1, 1], wspace=0.08, hspace=0.18)

    # ---- main view
    ax = fig.add_subplot(gs[:, 0])
    shape(ax)
    dim(ax, (x_W - 1.6, -Z_o), (x_W - 1.6, Z_o), "밑변 25.4\n(고정)", FIX, off=(-1.6, 0), fs=13)
    ext(ax, (x_W, Z_o), (x_W - 2.0, Z_o), FIX); ext(ax, (x_W, -Z_o), (x_W - 2.0, -Z_o), FIX)
    dim(ax, (x_W, Z_o + 1.4), (x_i, Z_o + 1.4), "t_w 3 (고정)", FIX, off=(0, 0.9), fs=12)
    ext(ax, (x_W, Z_o), (x_W, Z_o + 1.8), FIX); ext(ax, (x_i, z_F), (x_i, Z_o + 1.8), FIX)
    dim(ax, (x_i, -4.5), (-L, -4.5), "c", VAR, off=(0, 0.9))
    dim(ax, (-L, -Z_o - 1.4), (0, -Z_o - 1.4), "L", VAR, off=(0, -0.9))
    ext(ax, (-L, -L_a / 2), (-L, -Z_o - 1.8), VAR); ext(ax, (0, -Z_o), (0, -Z_o - 1.8), VAR)
    dim(ax, (x_W, -Z_o - 3.6), (0, -Z_o - 3.6), "W = L + c + t_w  (유도)", DER, off=(0, -0.9), fs=12)
    ext(ax, (x_W, -Z_o), (x_W, -Z_o - 4.0), DER); ext(ax, (0, -Z_o - 1.8), (0, -Z_o - 4.0), DER)
    dim(ax, (1.6, z_F), (1.6, Z_o), "", VAR, inward=True, gap=1.2)
    ext(ax, (0, z_F), (2.0, z_F), VAR); ext(ax, (0, Z_o), (2.0, Z_o), VAR)
    ax.text(2.4, (z_F + Z_o) / 2, "t_f", color=VAR, fontsize=15, fontweight="bold", va="center")
    c0 = np.array([-L + t_a, L_a / 2])
    ax.plot([c0[0], c0[0] + 4.2], [c0[1]] * 2, color=DER, lw=0.9, ls=":")
    ax.add_patch(Arc(c0, 5.6, 5.6, theta1=0, theta2=d["alpha"], color=DER, lw=2))
    ax.text(c0[0] + 3.6, c0[1] + 1.6, "α (유도)", color=DER, fontsize=13, fontweight="bold")
    for (x0, z0, w, h), lab in [((-2.2, 7.6, 2.8, 4.6), "A"), ((-9.0, -2.4, 3.2, 4.8), "B")]:
        ax.add_patch(Rectangle((x0, z0), w, h, fill=False, ec="#1f5fbf", lw=1.5, ls="--"))
        ax.text(x0 + 0.15, z0 + h - 0.15, lab, color="#1f5fbf", fontsize=14, fontweight="bold", va="top")
    ax.set_xlim(-19.5, 4.2); ax.set_ylim(-18.2, 15.8); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("기준 형상 (bistable_ref_1.2mm) 단면, 단위 mm", fontsize=14, loc="left")

    # ---- inset A: tip hinge
    axA = fig.add_subplot(gs[0, 1])
    shape(axA)
    dim(axA, (-t_h, 10.0), (0, 10.0), "", VAR, inward=True, gap=0.55)
    axA.text(-1.05, 10.0, "t_h", color=VAR, fontsize=15, fontweight="bold", ha="center", va="center")
    dim(axA, (-1.55, H), (-1.55, z_F), "L_h", VAR, off=(-0.42, 0))
    R = d["R_slot"]; cR = np.array([-(t_h + R), z_F - R])
    axA.add_patch(Arc(cR, 2 * R, 2 * R, theta1=0, theta2=90, color=VAR, lw=3))
    axA.annotate("", xy=cR + R * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)]), xytext=cR,
                 arrowprops=dict(arrowstyle="-|>", color=VAR, lw=1.8))
    axA.plot(*cR, "o", color=VAR, ms=3)
    axA.text(cR[0] + 0.32, cR[1] - 0.38, "R = ρ·L_h/2", color=VAR, fontsize=12, fontweight="bold", ha="center")
    axA.set_xlim(-2.4, 0.8); axA.set_ylim(7.4, 12.4); axA.set_aspect("equal"); axA.axis("off")
    axA.set_title("A  바깥 힌지 (위·아래 2개 동일)", fontsize=13, loc="left", color="#1f5fbf")

    # ---- inset B: apex hinge
    axB = fig.add_subplot(gs[1, 1])
    shape(axB)
    dim(axB, (-L, 0.0), (-L + t_a, 0.0), "", VAR, inward=True, gap=0.55)
    axB.text(-L + t_a + 0.95, 0.0, "t_a", color=VAR, fontsize=15, fontweight="bold", ha="center", va="center")
    dim(axB, (-8.0, -L_a / 2), (-8.0, L_a / 2), "L_a", VAR, off=(-0.45, 0))
    ext(axB, (-L, L_a / 2), (-8.3, L_a / 2), VAR); ext(axB, (-L, -L_a / 2), (-8.3, -L_a / 2), VAR)
    axB.set_xlim(-9.2, -5.2); axB.set_ylim(-2.6, 2.6); axB.set_aspect("equal"); axB.axis("off")
    axB.set_title("B  가운데 힌지", fontsize=13, loc="left", color="#1f5fbf")

    # ---- table
    axT = fig.add_subplot(gs[:, 2]); axT.axis("off")
    rows = [("t_h", "바깥 힌지 두께", REF["t_h"]), ("L_h", "바깥 힌지 길이", REF["L_h"]),
            ("t_a", "가운데 힌지 두께", REF["t_a"]), ("L_a", "가운데 힌지 길이", REF["L_a"]),
            ("L", "팔 길이 (x=0 → 꼭짓점)", REF["L"]), ("t_f", "플랜지 두께", REF["t_f"]),
            ("c", "꼭짓점–뒷벽 간격", REF["c"]), ("rho", "필렛 비율 ρ (공통, 무차원)", REF["rho"])]
    y = 0.97
    axT.text(0.0, y, "설계 변수 8개 (독립, 각자 범위 안에서 균일 추출)", fontsize=15, fontweight="bold", transform=axT.transAxes)
    y -= 0.06
    for x, h in zip([0.0, 0.13, 0.62, 0.78], ["기호", "의미", "기준", "범위 [mm]"]):
        axT.text(x, y, h, fontsize=12.5, fontweight="bold", color="#444", transform=axT.transAxes)
    axT.plot([0, 1], [y - 0.015] * 2, color="#999", lw=0.8, transform=axT.transAxes)
    for sym, name, ref in rows:
        y -= 0.055
        lo, hi = RANGES[sym]
        axT.text(0.0, y, "ρ" if sym == "rho" else sym, fontsize=14, fontweight="bold", color=VAR, transform=axT.transAxes)
        axT.text(0.13, y, name, fontsize=13, transform=axT.transAxes)
        axT.text(0.62, y, f"{ref:.3g}", fontsize=13, transform=axT.transAxes)
        axT.text(0.78, y, f"{lo:g} – {hi:g}", fontsize=13, fontweight="bold", transform=axT.transAxes)
    y -= 0.085
    axT.text(0.0, y, "고정값", fontsize=15, fontweight="bold", transform=axT.transAxes)
    fixed = [("밑변 (뒷벽 바깥면, 고정 지지면) 길이", "25.4 mm"), ("뒷벽 두께 t_w", f"{T_W:g} mm"),
             ("압출 깊이", "6 mm")]
    for name, val in fixed:
        y -= 0.05
        axT.text(0.0, y, name, fontsize=13, color=FIX, transform=axT.transAxes)
        axT.text(0.62, y, val, fontsize=13, fontweight="bold", color=FIX, transform=axT.transAxes)
    y -= 0.085
    axT.text(0.0, y, "유도량 (변수에서 자동 계산)", fontsize=15, fontweight="bold", transform=axT.transAxes)
    derivs = [("α 팔 경사각", "atan[(12.7 - t_f - L_h - t_h - L_a/2) / (L - t_a)]",
               f"가능 29.7–62.4°, 실제 768개 {tables.alpha_deg.min():.1f}–{tables.alpha_deg.max():.1f}°"),
              ("필렛 반경", "바깥 힌지 쪽 ρ·L_h/2,  가운데 힌지 쪽 ρ·L_a/2", "0.30–1.60 mm (기준 0.80)"),
              ("W 전체 가로 길이", "L + c + 3", f"10.0–15.0 mm, 실제 {tables.W.min():.1f}–{tables.W.max():.1f} (기준 13.0)")]
    for name, f, rng in derivs:
        y -= 0.05
        axT.text(0.0, y, name, fontsize=13, color=DER, fontweight="bold", transform=axT.transAxes)
        axT.text(0.30, y, f, fontsize=11.5, color=DER, transform=axT.transAxes)
        y -= 0.04
        axT.text(0.30, y, rng, fontsize=11.5, color=DER, transform=axT.transAxes)
    y -= 0.075
    axT.text(0.0, y, "무효 형상: Sobol 4,096점 중 4개 (0.1%, 바깥 힌지 0.1 mm + 큰 필렛 모서리) → 생성 시 건너뜀.\n"
             f"실제 768개에서는 무효 0.  예상 노드 최대 {tables.est_nodes.max() / 1e3:.0f}k (Student 한도 128k).",
             fontsize=11.5, color="#444", transform=axT.transAxes, va="top")

    fig.suptitle("설계 변수 정리 (v3)  —  빨강: 독립 변수 8개,  검정: 고정,  회색: 유도량", fontsize=19,
                 fontweight="bold", x=0.01, ha="left", y=0.995)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=RESULTS / "figures" / "geometry_design_variables_v3.png")
    draw(ap.parse_args().out)


if __name__ == "__main__":
    main()
