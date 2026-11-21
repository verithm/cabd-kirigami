"""Figures of the Ansys data collection (Korean labels: team figures). Outputs go to results/figures/ansys_*.png.

  theta        deformed shape and arm rotation of collected geometries: the rotation of the upper arm's A->B probe
               vector from its undeformed direction, taken where the force returns to zero (second stable point);
               a curve without that point is drawn at its last converged substep. Of the final and first-attempt
               (autofailed) curves, the one that reached further is used.     -> ansys_theta_<geom_id>.png
  classes      force-displacement curves of each class (bi, mono, neg_end, unknown) of the 512 training geometries,
               6 random ones per class (numpy default_rng(0)), to show what the undecided classes look like; reads only
               the shared files (data/CABD_dataset.csv, data/curves/curves_base.json.gz).
                                                                                -> ansys_class_examples_<state>.png
  kit-check    one-page figure of the collect-v2 kit verification (2026-11-03, for the team chat): TR0000 solved twice
               (development run and the distributed kit) gives the same curve, and three geometries show the snap-
               through region. Inputs: the three development-run curves copied to data/kit_check/ (from the
               development folder of 2026-11-02) and TR0000 of PC1's raw results.   -> ansys_kit_verification.png

usage: python -m cabd.ansys.figures theta TR0293 [TE0011 ...] [--state partial full]
       python -m cabd.ansys.figures classes [--state partial]
       python -m cabd.ansys.figures kit-check
       (all: [--out DIR], default results/figures)
"""
import argparse
import collections
import csv
import gzip
import io
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon
import numpy as np
import pandas as pd

from cabd.ansys.results import Results, angle, load_state
from cabd.config import DATA, RESULTS

FIGURES = RESULTS / "figures"
KIT_CHECK = DATA / "kit_check"
INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
FILL = {"frame": "#d9d8d4", "arm": "#cde2fb", "hinge": "#f6c3ad"}


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


# ----------------------------------------------------------------------------------------------- theta
def _bodies(points, which):
    out = collections.defaultdict(list)
    for name, xy in points.items():
        m = re.match(r"(.+)_v\d+$", name)
        if m:
            out[m.group(1)].append(xy[which])
    return out


def draw_shape(ax, points, which):
    for name, poly in _bodies(points, which).items():
        ax.add_patch(Polygon(np.array(poly), closed=True, ec=MUTED, lw=.8, fc=FILL["frame" if name == "frame" else "arm" if name.startswith("arm") else "hinge"]))
    for side in ("upper", "lower"):
        ax.annotate("", xy=points[f"B_{side}"][which], xytext=points[f"A_{side}"][which],
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2, shrinkA=0, shrinkB=0))
    ax.set_aspect("equal")
    ax.tick_params(colors=MUTED, length=2)
    ax.set_xlabel("x (mm)", color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_angle(ax, points):
    a0, b0, a1, b1 = (points[k][j] for k, j in (("A_upper", 0), ("B_upper", 0), ("A_upper", 1), ("B_upper", 1)))
    v0, v1 = b0 - a0, b1 - a1
    t0, t1 = angle(v0), angle(v1)
    theta = t1 - t0
    end = a1 + 1.25 * np.linalg.norm(v1) * v0 / np.linalg.norm(v0)
    ax.plot([a1[0], end[0]], [a1[1], end[1]], ls="--", color=ORANGE, lw=1.6)
    r = .55 * np.linalg.norm(v1)
    ax.add_patch(Arc(a1, 2 * r, 2 * r, theta1=min(t0, t1), theta2=max(t0, t1), color=ORANGE, lw=2.2))
    return abs(theta)


TITLE = {"bi": "두 번째 안정점", "mono": "단안정, 마지막 해석 시점", "neg_end": "끝까지 음수, 마지막 해석 시점", "unknown": "조기 중단, 마지막 해석 시점"}


@fresh_style
def theta(gid, states, out_dir, raw=None):
    from cabd.ansys.dataset import assignment
    plt.rcParams.update({"font.family": "Malgun Gothic", "font.size": 10, "axes.unicode_minus": False})
    assign = assignment("base")
    if gid not in assign.index:
        raise SystemExit(f"unknown geometry: {gid}")
    res = Results(int(assign.pc[gid]), raw)
    found = {s: d for s in states if (d := load_state(res, gid, s)) is not None}
    if not found:
        raise SystemExit(f"{gid}: no curves in {res.name}")
    fig, axs = plt.subplots(1, len(found) + 1, figsize=(5.2 * (len(found) + 1), 6), constrained_layout=True, sharey=True)
    first = next(iter(found.values()))
    draw_shape(axs[0], first["points"], 0)
    axs[0].set_title("변형 전 (u = 0)", loc="left", fontweight="bold")
    axs[0].set_ylabel("z (mm)", color=MUTED)
    extents = []
    for ax, (state, d) in zip(axs[1:], found.items()):
        draw_shape(ax, d["points"], 1)
        th = draw_angle(ax, d["points"])
        note = " · 첫 해석 곡선" if d["tag"] else ""
        ax.set_title(f"{state}: {TITLE[d['kind']]}\nu = {d['u']:.2f} mm, 회전각 = {th:.1f}°{note}", loc="left", fontweight="bold", fontsize=10.5)
        print(f"{gid} {state}: {d['kind']}, u = {d['u']:.2f} mm, rotation = {th:.1f} deg, F_min = {d['F_min']:.3f} N, curve end = {d['u_end']:.2f} mm{note}")
    for d in found.values():
        extents += [xy[j] for xy in d["points"].values() for j in (0, 1)]
    e = np.array(extents)
    for ax in axs:
        ax.set_xlim(e[:, 0].min() - 1.2, e[:, 0].max() + 5)
        ax.set_ylim(e[:, 1].min() - 1.2, e[:, 1].max() + 1.2)
    p = assign.loc[gid]
    fig.suptitle(f"{gid}   t_h {p.t_h:.2f} · L_h {p.L_h:.2f} · t_a {p.t_a:.2f} · L_a {p.L_a:.2f} · L {p.L:.2f} · t_f {p.t_f:.2f} · c {p.c:.2f} · rho {p.rho:.2f}", fontsize=11, color=INK)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ansys_theta_{gid}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("saved", path)


# ----------------------------------------------------------------------------------------------- classes
ORDER = [("bi", "쌍안정 (bi)", "0 아래로 내려갔다가 다시 0 위로 올라옴"),
         ("mono", "단안정 (mono)", "0 아래로 내려가지 않고 12 mm 이상 진행"),
         ("neg_end", "판정 불가 (neg_end)", "0 아래로 내려간 채 곡선이 끝남 (대부분 15 mm까지 수렴)"),
         ("unknown", "판정 불가 (unknown)", "0 아래로 내려가지 않았지만 12 mm 전에 멈춤")]


@fresh_style
def classes(s, out_dir):
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
    df = pd.read_csv(DATA / "CABD_dataset.csv")
    df = df[df.group == "train_base"]
    curves = json.load(gzip.open(DATA / "curves" / "curves_base.json.gz", "rt"))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), constrained_layout=True)
    for ax, (cls, title, rule) in zip(axes, ORDER):
        ids = df.geom_id[df[f"{s}_class"] == cls].tolist()
        ax.axhline(0, color="#888", lw=.8)
        ax.axvline(12, color="#c33", lw=.8, ls="--")
        ax.text(12.15, .97, "12 mm", transform=ax.get_xaxis_transform(), color="#c33", fontsize=8, va="top")
        lo, hi = 0, 0
        for g in rng.choice(ids, size=min(6, len(ids)), replace=False) if ids else []:
            c = curves[g][s]
            u, f = np.r_[0, c["u"]], np.r_[0, c["F"]]
            keep = u <= 12.5
            ax.plot(u, f, lw=1.4, label=f"{g} (끝 {u[-1]:.1f} mm)")
            ax.plot(u[-1], f[-1], "o", ms=4, color=ax.lines[-1].get_color())
            lo, hi = min(lo, f[keep].min()), max(hi, f[keep].max())
        ax.set_ylim(lo - .2 * (hi - lo), hi + .2 * (hi - lo))
        ax.set_xlim(0, 15.5)
        ends = df.loc[df[f"{s}_class"] == cls, f"{s}_u_end_mm"]
        ax.set_title(f"{title}: {len(ids)}개\n{rule}\n곡선 끝 변위 {ends.min():.1f}~{ends.max():.1f} mm (중앙값 {ends.median():.1f})", loc="left", fontsize=10)
        ax.set_xlabel("가한 변위 u (mm)")
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[0].set_ylabel("반력 F (N)")
    fig.suptitle(f"{s} 상태 학습용 512개의 판정별 힘-변위 곡선 (판정별 무작위 6개, 점 = 해석이 멈춘 곳, y축은 12.5 mm까지에 맞춤)", fontsize=11)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"ansys_class_examples_{s}.png"
    fig.savefig(out, dpi=140)
    print("saved", out)


# ----------------------------------------------------------------------------------------------- kit check
@fresh_style
def kit_check(out_dir, raw=None):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    SURFACE, INK2, GRID = "#fcfcfb", "#52514e", "#e4e3df"
    pc1 = Results(1, raw, overlay=False).own   # the distributed kit's own run (PC1)

    def curve(key):
        text = pc1.bytes(key[4:]).decode() if key.startswith("KIT:") else (KIT_CHECK / key).read_text()
        rows = list(csv.DictReader(io.StringIO(text)))
        return (np.r_[0, [float(r["u_applied_mm"]) for r in rows]], np.r_[0, [float(r["fx_disp_N"]) for r in rows]])

    def style(ax):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=11, length=0)

    fig = plt.figure(figsize=(9, 12.6), facecolor=SURFACE)
    fig.text(0.06, 0.965, "데이터 수집 키트 검증 통과", fontsize=26, fontweight="bold", color=INK)
    fig.text(0.06, 0.937, "PC 1대에서 형상 단위로 끝까지 자동 실행을 확인했습니다. 768개 수집은 이제 시작입니다.", fontsize=13, color=INK2)

    # 1. same geometry twice
    ax = fig.add_axes([0.09, 0.685, 0.86, 0.195])
    style(ax)
    u0, f0 = curve("TR0000_h10_8_g60_partial_curve.csv")
    u1, f1 = curve("KIT:TR0000/partial_curve.csv")
    ax.plot(u0, f0, color=BLUE, lw=2, label="이전 실행")
    ax.plot(u1, f1, ls="none", marker="o", ms=8, mfc="none", mec=ORANGE, mew=1.8, label="최종 키트 zip으로 다시 실행")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("변위 (mm)", fontsize=12, color=INK2)
    ax.set_ylabel("반력 (N)", fontsize=12, color=INK2)
    ax.legend(frameon=False, fontsize=12, loc="upper left", labelcolor=INK)
    fig.text(0.06, 0.898, "같은 형상을 다시 돌려도 곡선이 일치합니다", fontsize=16, fontweight="bold", color=INK)
    fig.text(0.60, 0.898, "TR0000, 살짝 젖은 상태, 같은 PC", fontsize=11.5, color=INK2)

    # 2. three geometries: the snap-through region
    fig.text(0.06, 0.627, "서로 다른 형상에서 쌍안정 구간이 나옵니다", fontsize=16, fontweight="bold", color=INK)
    fig.text(0.06, 0.604, "살짝 젖은 상태. 형상마다 힘 크기가 달라 세로축이 서로 다르고, 큰 힘이 나는 곡선 끝부분은 잘라 냈습니다.", fontsize=11.5, color=INK2)
    cases = [("TR0000", "힌지 두꺼움", "KIT:TR0000/partial_curve.csv"),
             ("TR0362", "힌지 가장 얇음", "TR0362_h10_8_g60_partial_curve.csv"),
             ("TE0110", "프레임 가장 얇음", "TE0110_g60b_partial_autofailed_curve.csv")]
    for i, (gid, note, key) in enumerate(cases):
        ax = fig.add_axes([0.09 + i * 0.305, 0.39, 0.25, 0.17])
        style(ax)
        u, f = curve(key)
        k = int(np.argmin(f))
        lim = max(-f[k], f[:k + 1].max())
        ax.plot(u, f, color=BLUE, lw=2)
        ax.plot(u[k], f[k], marker="o", ms=8, color=BLUE, mec=SURFACE, mew=2)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_ylim(-lim * 1.55, lim * 1.2)
        ax.set_xlim(0, 15)
        ax.set_xticks([0, 5, 10, 15])
        ax.set_title(f"{gid}  {note}", fontsize=11.5, color=INK, loc="left")
        ax.annotate(f"최소 {f[k]:.3f} N", (u[k], f[k]), xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=11, color=INK)
        ax.set_xlabel("변위 (mm)", fontsize=11, color=INK2)
        if i == 0:
            ax.set_ylabel("반력 (N)", fontsize=12, color=INK2)

    # 3. checklist
    y = 0.337
    fig.text(0.06, y, "무엇을 확인했고, 무엇이 남았나", fontsize=16, fontweight="bold", color=INK)
    rows = [("통과", "새 zip을 풀어 실행", "입력 없이 해석, 추출, 결과 zip 생성까지 완료 (TR0000)"),
            ("통과", "수렴 실패 처리", "실패 곡선을 저장하고 굵은 메시로 재해석 (일부러 실패시켜 시험)"),
            ("통과", "해석 시작 실패 처리", "3번 시도 후 start_error로 기록 (일부러 실패시켜 시험)"),
            ("통과", "극단 형상 16개 메시", "모두 생성. 2개는 예비 메시 설정으로 넘어감"),
            ("통과", "힘 평형", "양쪽 반력 차이 최대 0.002 N (TR0000, 최대 힘 3.7 N)"),
            ("미확인", "768개 전체의 실패율", "해석해 본 3개 중 1개가 14.3 mm에서 멈춰 재해석됨"),
            ("미확인", "PC 3대 사이 재현성", "첫 결과 zip이 올라오면 확인")]
    y -= 0.034
    for status, what, detail in rows:
        done = status == "통과"
        fig.text(0.06, y, status, fontsize=12, fontweight="bold", color=INK if done else ORANGE)
        fig.text(0.135, y, what, fontsize=12.5, fontweight="bold", color=INK)
        fig.text(0.135, y - 0.021, detail, fontsize=11.5, color=INK2)
        y -= 0.0375
    fig.text(0.06, 0.026, "TR0000은 배포용 키트로, TR0362와 TE0110은 같은 메시 설정의 개발용 스크립트로 해석했습니다.", fontsize=10, color=INK2)
    fig.text(0.06, 0.010, "TE0110 곡선은 멈추기 전(14.3 mm)까지입니다.", fontsize=10, color=INK2)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ansys_kit_verification.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["theta", "classes", "kit-check"])
    ap.add_argument("geom_ids", nargs="*")
    ap.add_argument("--state", nargs="+", default=None, choices=["partial", "full"])
    ap.add_argument("--out", type=Path, default=FIGURES)
    ap.add_argument("--raw", type=Path, help="raw result zips (default ANSYS_RAW)")
    a = ap.parse_args()
    if a.what == "theta":
        for g in a.geom_ids:
            theta(g, a.state or ["partial", "full"], a.out, a.raw)
    elif a.what == "classes":
        classes((a.state or ["partial"])[0], a.out)
    else:
        kit_check(a.out, a.raw)


if __name__ == "__main__":
    main()
