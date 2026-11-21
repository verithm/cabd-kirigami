"""Distribution of every input and output variable, training (512) vs test (256), one image.

Rows 1-2: the 8 design variables over their sampling range. Row 3: theta2, u2, log10 dE_r (geometries with a
real second stable point in the partial state) and the stability classes.
usage: python -m cabd.plots.distributions   ->  results/figures/distributions.{png,svg}
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import data as D
from ..config import BOUNDS, RESULTS, X_COLS

# applied at the start of main() on top of matplotlib defaults, so the figure does not depend on rcParams changed
# by other modules imported in the same process (e.g. the viewer)
STYLE = {"font.family": "Malgun Gothic", "axes.unicode_minus": False, "svg.fonttype": "none"}
INK, QUIET, GRID = "#1f2933", "#5b6670", "#e6e6e3"
TRAIN, TEST = "#2a78d6", "#eb6834"

SYM = {"t_h": "$t_h$", "L_h": "$L_h$", "t_a": "$t_a$", "L_a": "$L_a$", "L": "$L$", "t_f": "$t_f$", "c": "$c$", "rho": "$\\rho$"}
NAME = {"t_h": "끝 힌지 두께", "L_h": "끝 힌지 길이", "t_a": "apex 힌지 두께", "L_a": "apex 힌지 길이",
        "L": "암 길이", "t_f": "플랜지 두께", "c": "apex–뒷벽 간격", "rho": "필렛 비율"}
UNIT = {"t_h": "mm", "L_h": "mm", "t_a": "mm", "L_a": "mm", "L": "mm", "t_f": "mm", "c": "mm", "rho": "-"}


def frames():
    d = pd.read_csv(D.DATASET)
    # the shared dataset keeps test outputs sealed; they live in CABD_test_results.csv
    t = pd.read_csv(D.TEST_RESULTS)
    outs = ["partial_class", "full_class", "partial_theta2_deg", "partial_u2_mm", "partial_dE_r_mJ"]
    test = d[d.group == "test"].drop(columns=[c for c in outs if c in d.columns]).merge(t[["geom_id"] + outs], on="geom_id", how="left")
    out = {}
    for g in ("train_base", "test"):
        x = (d[d.group == g] if g == "train_base" else test).reset_index(drop=True)
        p, f = x.partial_class.map(D.state_label), x.full_class.map(D.state_label)
        x["label"] = np.where(p.isna() | f.isna(), "unlabelled", p + ">" + f)
        reg = x.partial_class == "bi"
        x["theta2"] = x.partial_theta2_deg.where(reg)
        x["u2"] = x.partial_u2_mm.where(reg)
        x["logE"] = np.log10(x.partial_dE_r_mJ.where(reg & (x.partial_dE_r_mJ > 0)))
        out[g] = x
    return out["train_base"], out["test"]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(QUIET)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=QUIET, labelsize=8, length=2)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def hist(ax, tr, te, bins, title, xlabel):
    tr, te = tr.dropna(), te.dropna()
    w = lambda v: np.full(len(v), 100 / len(v))          # percent of each set, so 512 and 256 compare directly
    ax.hist(tr, bins=bins, weights=w(tr), color=TRAIN, alpha=0.85, rwidth=0.9, label=f"학습 ({len(tr)})")
    ax.hist(te, bins=bins, weights=w(te), histtype="step", color=TEST, lw=1.6, label=f"시험 ({len(te)})")
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8.5, color=QUIET)
    style(ax)


def main(out=RESULTS / "figures"):
    matplotlib.rcdefaults()
    matplotlib.rcParams.update(STYLE)
    tr, te = frames()
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.6), facecolor="white")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.83, bottom=0.07, hspace=0.62, wspace=0.28)
    for ax, c in zip(axes[:2].ravel(), X_COLS):
        lo, hi = BOUNDS[c]
        hist(ax, tr[c], te[c], np.linspace(lo, hi, 11), f"{SYM[c]}  {NAME[c]}", f"{c} ({UNIT[c]}), 샘플링 범위 {lo}–{hi}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 13)
    for ax, (c, title, xl) in zip(axes[2, :3], [("theta2", "$\\theta_2$  두 번째 안정점 각도", "θ₂ (°)"),
                                                ("u2", "$u_2$  두 번째 안정점 변위", "u₂ (mm)"),
                                                ("logE", "$\\log_{10}\\Delta E$  에너지 장벽", r"$\log_{10}\Delta E$ (ΔE: mJ)")]):
        both = pd.concat([tr[c], te[c]]).dropna()
        hist(ax, tr[c], te[c], np.linspace(both.min(), both.max(), 16), title, xl)
    ax = axes[2, 3]
    order = ["bi>mono", "bi>bi", "mono>mono", "unlabelled"]
    lab = {"bi>mono": "bi→mono", "bi>bi": "bi→bi", "mono>mono": "mono→mono", "unlabelled": "판정 불가"}
    ptr = tr.label.value_counts(normalize=True).reindex(order, fill_value=0) * 100
    pte = te.label.value_counts(normalize=True).reindex(order, fill_value=0) * 100
    y = np.arange(len(order))
    ax.barh(y - 0.19, ptr, height=0.36, color=TRAIN)
    ax.barh(y + 0.19, pte, height=0.36, color="white", edgecolor=TEST, lw=1.4)
    for k in range(len(order)):
        ax.text(ptr.iloc[k] + 1, y[k] - 0.19, f"{ptr.iloc[k]:.0f}%", va="center", fontsize=7.5, color=INK)
        ax.text(pte.iloc[k] + 1, y[k] + 0.19, f"{pte.iloc[k]:.0f}%", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [lab[o] for o in order], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 60)
    ax.set_title("안정성 (partial → full)", loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel("비율 (%)", fontsize=8.5, color=QUIET)
    style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    for a in axes[:, 0]:
        a.set_ylabel("비율 (%)", fontsize=8.5, color=QUIET)
    axes[2, 0].set_ylabel("비율 (%)", fontsize=8.5, color=QUIET)

    fig.suptitle("변수별 분포 (학습 vs 시험)", x=0.06, ha="left", y=0.975, fontsize=15, fontweight="bold", color=INK)
    fig.text(0.06, 0.925, "입력 8개: Scrambled Sobol, 형상 제약 통과분.  출력 3개: partial 상태에서 두 번째 안정점이 있는 형상만 (학습 408, 시험 207).  "
             "세로축은 각 세트 안의 비율(%)이라 개수가 다른 두 세트를 바로 비교할 수 있음.",
             fontsize=9, color=QUIET)
    h = [plt.Rectangle((0, 0), 1, 1, color=TRAIN, alpha=0.85), plt.Line2D([], [], color=TEST, lw=1.6)]
    fig.legend(h, [f"학습 {len(tr)}개", f"시험 {len(te)}개"], loc="upper right", bbox_to_anchor=(0.985, 0.985),
               ncol=2, frameon=False, fontsize=9.5)
    for row, label in ((0, "입력 (설계 변수)"), (2, "출력")):
        y0 = axes[row, 0].get_position().y1 + 0.045
        fig.text(0.06, y0, label, fontsize=10.5, color=INK, fontweight="bold")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"distributions.{ext}", dpi=200, facecolor="white")
    print({g: len(x) for g, x in (("train", tr), ("test", te))}, "reg", tr.theta2.notna().sum(), te.theta2.notna().sum())


if __name__ == "__main__":
    main()
