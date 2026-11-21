"""Learning-curve figure from results/learning_curve/metrics.csv (test set 256; mean ± sd over seeds).

usage: python -m cabd.plots.learning_curve  ->  results/figures/learning_curve.{png,svg}
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from ..config import RESULTS, TAU

# applied at the start of main() on top of matplotlib defaults, so the figure does not depend on rcParams changed
# by other modules imported in the same process (e.g. the viewer)
STYLE = {"font.family": "Malgun Gothic", "axes.unicode_minus": False, "svg.fonttype": "none"}
INK, QUIET, GRID = "#1f2933", "#5b6670", "#e6e6e3"
COL = {"mlp": "#2a78d6", "gp": "#eb6834", "rom_gp": "#1baf7a"}
NAME = {"mlp": "MLP 앙상블", "gp": "GP", "rom_gp": "ROM+GP"}


def main(out=RESULTS / "figures"):
    matplotlib.rcdefaults()
    matplotlib.rcParams.update(STYLE)
    d = pd.read_csv(RESULTS / "learning_curve" / "metrics.csv")
    panels = [("rmse_theta2", "θ₂ 시험 RMSE (°)", TAU[0]), ("rmse_u2", "u₂ 시험 RMSE (mm)", TAU[1]),
              ("rmse_logE", r"$\log_{10}\Delta E$ 시험 RMSE", TAU[2]), ("bal_acc_3class", "안정성 3클래스 balanced accuracy", None)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9), facecolor="white")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.78, bottom=0.16, wspace=0.32)
    order = [m for m in ("mlp", "gp", "rom_gp") if m in set(d.model)]
    for k, (ax, (c, title, tau)) in enumerate(zip(axes, panels)):
        ends = []
        for m in order:
            s = d[d.model == m].groupby("size")[c].agg(["mean", "std"])
            ax.errorbar(s.index, s["mean"], yerr=s["std"].fillna(0), color=COL[m], marker="o", ms=5, lw=1.8, capsize=3,
                        label=NAME[m])
            ends.append((s["mean"].iloc[-1], m))
        if tau is not None:
            ax.axhline(tau, color=QUIET, lw=0.9, ls="--")
            ax.text(1536, tau, "허용오차 τ ", va="bottom", ha="right", fontsize=8, color=QUIET)
            ax.set_ylim(0, tau * 1.25)
        else:
            ax.set_ylim(0.85, 1.0)
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.065                    # end labels only, spread so they never overlap
        ys = []
        for y, m in sorted(ends):
            ys.append(max(y, ys[-1] + gap) if ys else y)
        shift = max(0.0, ys[-1] - (hi - gap / 2)) if ys else 0
        for (y, m), yl in zip(sorted(ends), ys):
            ax.text(1585, yl - shift, f"{y:.3f}", va="center", ha="left", fontsize=8.5, color=COL[m], fontweight="bold")
        ax.set_xticks([512, 1024, 1536])
        ax.set_xlim(380, 1760)
        ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.set_xlabel("학습 형상 수", fontsize=9, color=QUIET)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8.5, colors=QUIET)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Learning curve: 학습 데이터 512 → 1,024 → 1,536", x=0.05, ha="left", y=0.97, fontsize=14, fontweight="bold", color=INK)
    nseed = d.groupby("model").seed.nunique()
    seeds = ", ".join(f"{NAME[m]} 시드 {nseed[m]}개" for m in order)
    fig.text(0.05, 0.86, f"시험 세트 256개(회귀 207, 클래스 241) 기준. {seeds} (여러 시드는 평균 ± 표준편차). "
             "크기마다 처음부터 학습(같은 튜닝 예산). 학습 세트는 Sobol 순서대로 기존 → +ext1 → +ext2.", fontsize=9, color=QUIET)
    if len(order) > 1:
        axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"learning_curve.{ext}", dpi=200, facecolor="white")


if __name__ == "__main__":
    main()
