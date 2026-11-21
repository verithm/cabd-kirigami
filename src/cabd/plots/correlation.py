"""Pearson R between design inputs and outputs of the 512 training geometries (test set not used), one image.

Panels: A inputs x outputs, B outputs x outputs, C inputs x inputs (sampling check).
Outputs: theta2, u2, log10 dE_r (partial bi geometries), partial bi (1/0, labeled geometries),
         full mono given partial bi (1/0). R with a 0/1 variable is the point-biserial correlation.
Each pair uses the geometries where both values exist; cells with p >= 0.05 are drawn in muted ink.
usage: python -m cabd.plots.correlation   ->  results/figures/correlation_R.{png,svg}, correlation_R_*.csv
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from .. import data as D
from ..config import RESULTS, X_COLS

# applied at the start of main() on top of matplotlib defaults, so the figure does not depend on rcParams changed
# by other modules imported in the same process (e.g. the viewer)
STYLE = {"font.family": "Malgun Gothic", "axes.unicode_minus": False, "svg.fonttype": "none"}
INK, QUIET, MUTED, GAP = "#1f2933", "#5b6670", "#9aa0a6", "white"
CMAP = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

SYM = {"t_h": "$t_h$", "L_h": "$L_h$", "t_a": "$t_a$", "L_a": "$L_a$", "L": "$L$", "t_f": "$t_f$", "c": "$c$", "rho": "$\\rho$"}
NAME = {"t_h": "끝 힌지 두께", "L_h": "끝 힌지 길이", "t_a": "apex 힌지 두께", "L_a": "apex 힌지 길이",
        "L": "암 길이", "t_f": "플랜지 두께", "c": "apex–뒷벽 간격", "rho": "필렛 비율"}
OUT = {"theta2": "$\\theta_2$", "u2": "$u_2$", "logE": "$\\log_{10}\\Delta E$", "bi": "partial\nbi", "mono": "full\nmono"}


def data():
    d = D.load()
    p, f = d.partial_class.map(D.state_label), d.full_class.map(D.state_label)
    d["bi"] = np.where(p.isna(), np.nan, (p == "bi").astype(float))
    d["mono"] = np.where((p == "bi") & f.notna(), (f == "mono").astype(float), np.nan)
    for c in ("theta2", "u2", "logE"):
        d.loc[~d.is_reg, c] = np.nan
    return d


def corr(d, rows, cols):
    R, P, N = (np.full((len(rows), len(cols)), np.nan) for _ in range(3))
    for i, a in enumerate(rows):
        for j, b in enumerate(cols):
            m = d[a].notna() & d[b].notna()
            N[i, j] = m.sum()
            if a != b and d.loc[m, a].nunique() > 1 and d.loc[m, b].nunique() > 1:
                R[i, j], P[i, j] = stats.pearsonr(d.loc[m, a], d.loc[m, b])
    return R, P, N


def heat(ax, R, P, xlabels, ylabels, title, mask=None):
    M = R.copy()
    if mask is not None:
        M[mask] = np.nan
    im = ax.imshow(M, cmap=CMAP, vmin=-1, vmax=1, aspect="equal")
    ax.set_anchor("N")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            v = 0.0 if abs(M[i, j]) < 0.005 else M[i, j]
            strong = abs(v) >= 0.6
            ax.text(j, i, f"{v:.2f}".replace("0.", ".", 1), ha="center", va="center", fontsize=10,
                    fontweight="bold" if strong else "normal",
                    color="white" if strong else (INK if P[i, j] < 0.05 else MUTED))
    ax.set_xticks(range(M.shape[1]), xlabels, fontsize=10)
    ax.set_yticks(range(M.shape[0]), ylabels, fontsize=10)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(M.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(M.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=GAP, linewidth=2.5)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=34)
    return im


def main(out=RESULTS / "figures"):
    matplotlib.rcdefaults()
    matplotlib.rcParams.update(STYLE)
    d = data()
    X, Y = X_COLS, ["theta2", "u2", "logE", "bi", "mono"]
    fig = plt.figure(figsize=(14, 6.6), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[5, 3, 7], wspace=0.42, left=0.115, right=0.985, top=0.70, bottom=0.13)

    R, P, N = corr(d, X, Y)
    im = heat(fig.add_subplot(gs[0]), R, P, [OUT[c] for c in Y], [f"{SYM[c]}  {NAME[c]}" for c in X], "A  입력 → 출력")

    YB = ["theta2", "u2", "logE", "mono"]      # partial bi left out: theta2, u2, logE exist only when it is 1
    R2, P2, _ = corr(d, YB, YB)
    heat(fig.add_subplot(gs[1]), R2[1:, :-1], P2[1:, :-1], [OUT[c] for c in YB[:-1]],
         [OUT[c].replace("\n", " ") for c in YB[1:]], "B  출력 ↔ 출력", mask=np.triu(np.ones((3, 3), bool), 1))

    R3, P3, _ = corr(d, X, X)
    heat(fig.add_subplot(gs[2]), R3[1:, :-1], P3[1:, :-1], [SYM[c] for c in X[:-1]], [SYM[c] for c in X[1:]],
         "C  입력 ↔ 입력 (샘플링 확인)", mask=np.triu(np.ones((7, 7), bool), 1))

    cax = fig.add_axes([0.40, 0.055, 0.30, 0.022])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, -0.5, 0, 0.5, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, length=0, colors=QUIET)
    cb.set_label("Pearson R", fontsize=9.5, color=QUIET)

    n_reg, n_bi, n_mono = int(N[0, 0]), int(N[0, 3]), int(N[0, 4])
    fig.suptitle("입력·출력 상관관계 (Pearson R, 학습 데이터)", x=0.115, ha="left", y=0.985, fontsize=16, fontweight="bold", color=INK)
    fig.text(0.115, 0.915,
             f"학습 형상 512개, 시험 세트 제외.  θ₂ · u₂ · log ΔE: partial bi {n_reg}개  |  partial bi (1/0): 라벨 있는 {n_bi}개  |  "
             f"full mono (1/0): partial bi 중 {n_mono}개\n"
             "0/1 출력은 point-biserial R.  회색 글자는 p ≥ 0.05.  θ₂ · u₂ · ΔE는 partial bi일 때만 정의되므로 B에서 partial bi는 뺐음",
             fontsize=9.5, color=QUIET, va="top", linespacing=1.6)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"correlation_R.{ext}", dpi=200, facecolor="white")
    pd.DataFrame(R, index=X, columns=Y).round(3).to_csv(out / "correlation_R_inputs_outputs.csv")
    pd.DataFrame(R2, index=YB, columns=YB).round(3).to_csv(out / "correlation_R_outputs.csv")
    print("max |R| between inputs:", np.round(np.nanmax(np.abs(R3)), 3))


if __name__ == "__main__":
    main()
