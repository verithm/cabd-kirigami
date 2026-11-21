"""Score the inverse-design candidates with their Ansys results (verification kits 1 and 2).

Ansys success of a candidate (same rule as the referee, with the Ansys classes in place of P >= 0.9):
  partial state bistable AND full state monostable AND max_j |y_j - t_j| / tau_j <= 1,
  y = (theta2_deg, u2_mm, log10 dE_r) of the partial state, tau = config.TAU.
Scalars come from cabd.ansys.results.scalars (the curve that reached further), the same as the datasets.
A candidate whose either state failed or has not been run yet is "unsolved"; rates are given over the solved ones,
and the count of unsolved ones is listed next to them (they are not silently counted as failures).
Rates come with Wilson 95% intervals.

Result zips: the newest collect_result_PC<label>_*.zip per label in ANSYS_RAW; kit 1 labels ALL / V1-V3, kit 2 ALL /
P1-Pn. Both kits use the label ALL; the geometry ids (V..., W..., X...) keep them apart, so every zip is read for
both kits (later labels in sorted order win).

Candidates: results/inverse/ansys_verify_candidates.csv (kit 1, train_size 512) and, if present,
results/inverse/v2/ansys_verify2_candidates.csv (kit 2). The evidence copies in results/verify/ were scored on
2026-11-18, before kit 2 existed: `--kits 1` reproduces them (with kit 2 present, its 320 candidates are added as
unsolved until its results arrive).

usage: python -m cabd.verify.score [--kits 1 2] [--out DIR]   writes ansys_verify_scored.csv, ansys_verify_summary.csv,
                                                            ansys_verify.png / .svg (default out: results/verify)
       python -m cabd.verify.score --self-test    check the pipeline on the 40 ansys40 test geometries themselves
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cabd.ansys.figures import fresh_style
from cabd.ansys.results import Results, latest, scalars, zips
from cabd.config import DATA, RESULTS, TAU as _TAU
from cabd.verify.candidates import FILES, INVERSE

TAU = np.array(_TAU)
NAME = {"gp": "GP", "mlp": "MLP", "rom_gp": "ROM+GP", "cvae": "cVAE"}
OUT = RESULTS / "verify"
CRLF = "\r\n"


def candidates(kits=(1, 2), inverse=INVERSE):
    parts = []
    if 1 in kits:
        parts.append(pd.read_csv(inverse / FILES[1]).assign(train_size=512, kit=1))
    if 2 in kits and (inverse / FILES[2]).exists():
        parts.append(pd.read_csv(inverse / FILES[2]).assign(kit=2))
    return pd.concat(parts, ignore_index=True)


def result_sources(raw=None):
    return [Results(lb, raw) for lb in sorted(zips(raw)) if re.fullmatch(r"ALL|V\d+|P\d+", lb)]


def ansys_outputs(sources, ids):
    """solve_id -> dict of partial/full status, classes and the partial-state targets; later sources win."""
    ids, out = set(ids), {}
    for res in sources:
        p = res.read("progress.csv")
        if p is None:
            continue
        for _, x in latest(p).iterrows():
            if x.geom_id not in ids:
                continue
            r = out.setdefault(x.geom_id, {"source": res.name})
            r[f"{x.state}_status"] = x.status
            if x.status in ("ok", "not_converged"):
                s = scalars(res, x.geom_id, x.state)
                r[f"{x.state}_class"] = s.get("class")
                if x.state == "partial":
                    r.update(theta2=s.get("theta2_deg", np.nan), u2=s.get("u2_mm", np.nan), dE_r=s.get("dE_r_mJ", np.nan))
    return out


def score(c, out):
    rows = []
    for _, r in c.iterrows():
        a = out.get(r.solve_id, {})
        done = all(a.get(f"{s}_status") in ("ok", "not_converged") for s in ("partial", "full"))
        d = dict(r)
        d.update(ansys_source=a.get("source"), ansys_partial_status=a.get("partial_status"), ansys_full_status=a.get("full_status"),
                 solved=done, ansys_partial_class=a.get("partial_class"), ansys_full_class=a.get("full_class"),
                 ansys_theta2=a.get("theta2", np.nan), ansys_u2=a.get("u2", np.nan), ansys_dE_r=a.get("dE_r", np.nan))
        logE = np.log10(d["ansys_dE_r"]) if d["ansys_dE_r"] and d["ansys_dE_r"] > 0 else np.nan
        d["ansys_logE"] = logE
        err = np.abs(np.array([d["ansys_theta2"], d["ansys_u2"], logE]) - [r.t_theta2, r.t_u2, r.t_logE]) / TAU
        for j, k in enumerate(("theta2", "u2", "logE")):
            d[f"err_{k}_tau"] = err[j]
        d["ansys_maxerr_tau"] = np.nan if np.isnan(err).any() else err.max()
        d["ansys_bimono"] = d["ansys_partial_class"] == "bi" and d["ansys_full_class"] == "mono"
        d["ansys_ok"] = bool(done and d["ansys_bimono"] and d["ansys_maxerr_tau"] <= 1)
        rows.append(d)
    return pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    return centre - m, centre + m


def summary(s):
    rows = []
    for (n, m), g in s.groupby(["train_size", "method"]):
        sv = g[g.solved]
        k = int(sv.ansys_ok.sum())
        lo, hi = wilson(k, len(sv))
        r = dict(train_size=n, method=m, candidates=len(g), solved=len(sv), unsolved=len(g) - len(sv), success=k,
                 success_rate=k / len(sv) if len(sv) else np.nan, ci95_lo=lo, ci95_hi=hi,
                 bimono_rate=sv.ansys_bimono.mean() if len(sv) else np.nan,
                 median_maxerr_tau=sv.ansys_maxerr_tau.median() if len(sv) else np.nan)
        for k2 in ("theta2", "u2", "logE"):
            r[f"within_tau_{k2}"] = (sv[f"err_{k2}_tau"] <= 1).mean() if len(sv) else np.nan
        if "referee_ok" in g and len(sv):
            ref = sv.referee_ok.astype(bool)
            r["referee_success_rate"] = ref.mean()
            r["referee_agreement"] = (ref == sv.ansys_ok).mean()
        rows.append(r)
    return pd.DataFrame(rows)


@fresh_style
def figure(sm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "svg.fonttype": "none"})
    INK, QUIET, GRID = "#1f2933", "#5b6670", "#e6e6e3"
    shade = {512: "#9cc3ef", 1024: "#4f93dd", 1536: "#1d5fae"}
    methods = [m for m in NAME if m in set(sm.method)]
    sizes = sorted(sm.train_size.unique())
    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.12)
    w = 0.8 / len(sizes)
    for i, n in enumerate(sizes):
        for j, m in enumerate(methods):
            r = sm[(sm.train_size == n) & (sm.method == m)]
            if not len(r) or not r.solved.iloc[0]:
                continue
            r = r.iloc[0]
            x = j + (i - (len(sizes) - 1) / 2) * w
            ax.bar(x, r.success_rate, w * 0.9, color=shade[n], label=f"학습 {n:,}개" if j == 0 else None)
            ax.errorbar(x, r.success_rate, yerr=[[r.success_rate - r.ci95_lo], [r.ci95_hi - r.success_rate]], color=INK, lw=0.9, capsize=2.5)
            ax.text(x, r.ci95_hi + 0.02, f"{r.success}/{r.solved}", ha="center", va="bottom", fontsize=7.5, color=QUIET)
    ax.set_xticks(range(len(methods)), [NAME[m] for m in methods], fontsize=10, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Ansys 성공률", fontsize=9.5, color=QUIET)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5, colors=QUIET)
    ax.legend(frameon=False, fontsize=8.5, ncol=len(sizes), loc="upper right")
    fig.suptitle("역설계 후보의 Ansys 검증 결과", x=0.08, ha="left", y=0.97, fontsize=13, fontweight="bold", color=INK)
    fig.text(0.08, 0.83, "성공 = 부분 젖음 쌍안정, 완전 젖음 단안정, 세 출력 모두 |오차| ≤ τ. 목표 40개, 방법별 시드 0의 1순위 후보.\n"
             "막대 위 숫자는 성공/해석 완료, 오차 막대는 Wilson 95% 구간.", fontsize=8, color=QUIET)
    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, facecolor="white")


def self_test(raw=None):
    """The 40 ansys40 test geometries as their own candidates: every bi>mono one must succeed with zero error."""
    q = pd.read_csv(DATA / "queries.csv")
    q = q[(q.set == "final") & q.ansys40]
    c = pd.DataFrame({"solve_id": q.geom_id, "geom_id": q.geom_id, "method": "gp", "train_size": 0,
                      "t_theta2": q.theta2, "t_u2": q.u2, "t_logE": q.logE})
    srcs = [Results(pc, raw) for pc in (1, 2, 3)]
    s = score(c, ansys_outputs(srcs, c.solve_id))
    print(s[["solve_id", "solved", "ansys_partial_class", "ansys_full_class", "ansys_maxerr_tau", "ansys_ok"]].to_string())
    print(f"solved {int(s.solved.sum())}/40, ok {int(s.ansys_ok.sum())}, max error {s.ansys_maxerr_tau.max():.2e} tau")
    return s


def run(out=OUT, kits=(1, 2), raw=None, inverse=INVERSE, fig=True):
    c = candidates(kits, inverse)
    srcs = result_sources(raw)
    print("result zips:", [r.name for r in srcs] or "none yet")
    s = score(c, ansys_outputs(srcs, c.solve_id))
    out.mkdir(parents=True, exist_ok=True)
    s.to_csv(out / "ansys_verify_scored.csv", index=False, float_format="%.6g", lineterminator=CRLF)
    sm = summary(s)
    sm.to_csv(out / "ansys_verify_summary.csv", index=False, float_format="%.4f", lineterminator=CRLF)
    with pd.option_context("display.width", 200):
        print(sm.round(3).to_string(index=False))
    if fig and sm.solved.sum():
        figure(sm, out / "ansys_verify.png")
    return s, sm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--kits", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--raw", type=Path, help="raw result zips (default ANSYS_RAW)")
    a = ap.parse_args()
    if a.self_test:
        self_test(a.raw)
    else:
        run(a.out, a.kits, a.raw)


if __name__ == "__main__":
    main()
