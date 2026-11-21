"""Datasets from the raw Ansys results: one row per geometry, ready for pandas.

Outputs (default folder build/dataset; `--out data` refreshes the committed copies in data/):
  base     CABD_dataset.csv (512 train + 256 test + placeholder rows; test results blank), CABD_test_results.csv
           (the 256 test rows with results, same columns), CABD_dataset_columns.csv (meaning, type, unit of every
           column), curves/curves_base.json.gz (curves and deformed shapes of the training states, what the shared
           viewer draws), curves/curves_test.json.gz (the same for the test states, kept apart: sealed);
           with --xlsx also CABD_dataset.xlsx (sheets data, test_results, summary, notes, columns; needs openpyxl;
           text in sheet "notes" is kept when the file is rebuilt).
  ext      CABD_dataset_ext.csv: the extra training geometries (ext1 = PC4-6, ext2 = PC7-9), same columns and rules;
           curves/curves_ext.json.gz: their force-displacement curves for the ROM fit.
  table    geometry_table.csv: design values and collected results of every base geometry (lookup table). The
           committed file is the snapshot of 2026-11-08 23:55, right after the final kit was built (--final built,
           default, reproduces it); --final run gives the up-to-date table with what the final kit solved.
  share    build/share/CABD_viewer/ and build/share/CABD_viewer.zip: the team's viewer folder (viewer, dataset,
           training curves, never test results).
  show     print one or more geometries from geometry_table.csv.

Rules:
  * latest record of every (geometry, state) in each kit's progress.csv; the final kit replaces what PC1-3 recorded,
    and a state queued in the final kit but not run there drops an older failure of PC1-3 (it is waiting again);
    for the ext data, collect_result_PCretry1 replaces the PC4-6 records of the geometries it re-solved.
    (The final kit's records are read through PC1's reader, as the original did: the final kit's files, then PC1's.)
  * scalars: cabd.ansys.results.scalars (the curve that reached further). When the first-attempt curve is used, the
    mesh rule and node count are those of that first solve, from wb_log.txt (cabd.ansys.results.first_solve_mesh).
  * run_status: finished (status ok -> converged 1, not_converged -> 0), not_run, planned, failed (anything else).
  * placeholders: 192 adaptive (AD) and 40 inverse-validation (IV) rows with run_status planned.

usage: python -m cabd.ansys.dataset base|ext|table|share|all [--out DIR] [--raw DIR] [--xlsx] [--final built|run]
       python -m cabd.ansys.dataset show TR0293 [TE0011 ...]
"""
import argparse
import gzip
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from cabd.ansys import kit as K
from cabd.ansys.results import MemoryArchive, Results, best_curve, first_solve_mesh, latest, load_state, scalars
from cabd.config import BUILD, DATA
from cabd.geometry.params import VAR_NAMES as DESIGN

STATES = ("partial", "full")
SEEDS = {"train": 20261101, "test": 20261102}
GROUP = {"train": "train_base", "test": "test"}
N_ADAPTIVE, N_INVERSE = 192, 40
EXT_STAGES = {"ext1": (4, 5, 6), "ext2": (7, 8, 9)}
EXT_RETRY = ["retry1"]
CRLF = "\r\n"

# order: run check -> label -> targets along the curve -> where the numbers came from
FIELDS = ["run_status", "converged", "class", "F_max_N", "u1_mm", "F_min_N", "u2_mm", "dE_f_mJ", "dE_r_mJ", "theta2_deg",
          "u_end_mm", "mesh_rule", "nodes"]
NUMERIC = ["F_max_N", "u1_mm", "F_min_N", "u2_mm", "dE_f_mJ", "dE_r_mJ", "theta2_deg", "u_end_mm"]
HIDDEN_FOR_TEST = ["class", *NUMERIC]
COLUMNS = ["geom_id", "group", "sobol_seed", *DESIGN, *[f"{s}_{f}" for s in STATES for f in FIELDS]]

DESCRIPTIONS = {
    "run_status": ("text", "-", "planned = geometry not chosen yet; not_run = waiting; finished = solved; failed = solver did not give a result"),
    "converged": ("0/1", "-", "1 = reached 15 mm; 0 = stopped earlier; blank = not finished"),
    "class": ("text", "-", "bi = force goes below zero and returns through zero; mono = never below zero, curve reaches 12 mm or more; neg_end = still below zero at the end of the curve; unknown = never below zero, curve ends before 12 mm"),
    "F_max_N": ("float", "N", "Largest force before u1; largest force of the whole curve if the force never goes below zero"),
    "u1_mm": ("float", "mm", "Displacement where the force first crosses zero downward"),
    "F_min_N": ("float", "N", "Lowest force valley, excluding the origin; blank if the curve has no valley"),
    "u2_mm": ("float", "mm", "Second stable point: displacement where the force returns through zero"),
    "dE_f_mJ": ("float", "mJ", "Forward barrier, U(u1) - U(0)"),
    "dE_r_mJ": ("float", "mJ", "Reverse barrier, U(u1) - U(u2)"),
    "theta2_deg": ("float", "deg", "Rotation of the upper arm A-to-B vector at u2"),
    "u_end_mm": ("float", "mm", "Last displacement reached by the curve used"),
    "mesh_rule": ("text", "-", "Hinge mesh of the solve that produced the values: in-plane element size and layers along the depth"),
    "nodes": ("int", "count", "Mesh nodes of the solve that produced the values"),
}
BASE_DESCRIPTIONS = {
    "geom_id": ("text", "-", "Geometry id. TR = training, TE = test, AD = adaptive (placeholder), IV = inverse validation (placeholder)"),
    "group": ("text", "-", "train_base, test, adaptive, inverse_validation"),
    "sobol_seed": ("int", "-", "Scrambled Sobol seed of the point; blank for placeholders"),
    "t_h": ("float", "mm", "Tip hinge thickness"), "L_h": ("float", "mm", "Tip hinge length"),
    "t_a": ("float", "mm", "Apex hinge thickness"), "L_a": ("float", "mm", "Apex hinge length"),
    "L": ("float", "mm", "Arm length, x = 0 to apex outer face"), "t_f": ("float", "mm", "Flange thickness"),
    "c": ("float", "mm", "Gap between apex outer face and back wall inner face"), "rho": ("float", "-", "Common fillet ratio"),
}
# geometry_table.csv
TABLE_VARS = [("t_h", "끝 힌지 두께", "mm"), ("L_h", "끝 힌지 길이", "mm"), ("t_a", "가운데(apex) 힌지 두께", "mm"),
              ("L_a", "가운데(apex) 힌지 길이", "mm"), ("L", "암 길이", "mm"), ("t_f", "플랜지 두께", "mm"),
              ("c", "apex-뒷벽 간격", "mm"), ("rho", "필렛 비율", "-"), ("alpha_deg", "암 안쪽 빗변 각도 α", "deg")]
TABLE_CLS = {"bi": "쌍안정", "mono": "단안정", "neg_end": "미확인(끝까지 음수)", "unknown": "미확인(12 mm 전 중단)"}
TABLE_RESULTS = ["status", "mode", "mesh_rule", "curve", "u_end_mm", "class", "F_min_N", "dE_f_mJ", "dE_r_mJ", "u2_mm", "theta2_deg"]


def assignment(group):
    """assignment_*.csv of a collection group, read back from its %.6g text exactly as the original read it."""
    return pd.read_csv(io.StringIO(K._csv(K.assignment(group)[K.JOB_COLS], float_format="%.6g"))).set_index("geom_id")


def write_csv(df, path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator=CRLF, **kw)


def split_status(status):
    if status == "ok":
        return "finished", 1
    if status == "not_converged":
        return "finished", 0
    if status in ("planned", "not_run"):
        return status, np.nan
    return "failed", np.nan


# ----------------------------------------------------------------------------------------------- base dataset
def collect_base(raw=None):
    """{(geom_id, state): record} for the base geometries; the final kit replaces what PC1-3 recorded."""
    assign = assignment("base")
    rows = {}
    sources = {pc: Results(pc, raw) for pc in (1, 2, 3)}

    def add(res, progress):
        for _, x in latest(progress).iterrows():
            r = {"status": x.status, "mesh_rule": x.mesh_rule, "nodes": x.nodes}
            if x.status in ("ok", "not_converged"):
                r.update(scalars(res, x.geom_id, x.state))
                if r.get("curve") == "first":
                    src = sources[int(assign.loc[x.geom_id, "pc"])]
                    found = first_solve_mesh(src.texts(f"{x.geom_id}/wb_log.txt"), x.state)
                    if found is None:
                        raise SystemExit(f"first-solve mesh not found: {x.geom_id} {x.state}")
                    r["mesh_rule"], r["nodes"] = found
            rows[x.geom_id, x.state] = r

    for res in sources.values():
        add(res, res.read("progress.csv"))
    final = Results("final", raw, overlay=False).read("progress.csv")
    if final is not None:
        queued, _, _, _ = K.final_plan(raw)
        ran = set(zip(final.geom_id, final.state))
        for g, states in zip(queued.geom_id, queued.states):
            for s in states.split(";"):
                if (g, s) not in ran:
                    rows.pop((g, s), None)  # waiting in the final kit, so an older failure no longer applies
        add(sources[1], final)
    return rows


def _frame(out):
    T = pd.DataFrame(out).reindex(columns=COLUMNS)
    for c in ("sobol_seed", *[f"{s}_{f}" for s in STATES for f in ("converged", "nodes")]):
        T[c] = pd.to_numeric(T[c]).astype("Int64")  # whole numbers stay whole in the CSV files
    for c in T.columns:
        if T[c].dtype == object:
            T[c] = T[c].where(T[c].notna(), None)
    return T


def _row(r, group, seed, rows):
    d = {"geom_id": r.geom_id, "group": group, "sobol_seed": seed, **{v: r[v] for v in DESIGN}}
    for s in STATES:
        x = rows.get((r.geom_id, s), {"status": "not_run"})
        d[f"{s}_run_status"], d[f"{s}_converged"] = split_status(x["status"])
        for f in FIELDS[2:]:
            d[f"{s}_{f}"] = x.get(f, np.nan)
    return d


def build_table(rows):
    a = assignment("base").reset_index()
    out = [_row(r, GROUP[r.role], SEEDS[r.role], rows) for _, r in a.iterrows()]
    for prefix, group, n in (("AD", "adaptive", N_ADAPTIVE), ("IV", "inverse_validation", N_INVERSE)):
        for i in range(n):
            d = {"geom_id": f"{prefix}{i:04d}", "group": group}
            for s in STATES:
                d[f"{s}_run_status"] = "planned"
            out.append(d)
    return _frame(out)


def columns_table():
    rows = []
    for c in COLUMNS:
        per_state = c.startswith(("partial_", "full_"))
        kind, unit, text = DESCRIPTIONS[c.split("_", 1)[1]] if per_state else BASE_DESCRIPTIONS[c]
        rows.append(dict(column=c, state=c.split("_", 1)[0] if per_state else "", type=kind, unit=unit, description=text))
    return pd.DataFrame(rows)


def split_test(T):
    """(data with test results hidden, test rows with results, hidden columns)."""
    is_test = T.group.eq("test")
    test = T[is_test].copy()
    data = T.copy()
    hide = [f"{s}_{f}" for s in STATES for f in HIDDEN_FOR_TEST]
    data.loc[is_test, hide] = np.nan
    return data, test, hide


def summary_table(data):
    rows = []
    for g, q in data.groupby("group", sort=False):
        for s in STATES:
            for v, n in q[f"{s}_run_status"].value_counts().items():
                rows.append(dict(group=g, state=s, metric="run_status", value=v, count=int(n)))
            for v, n in q[f"{s}_converged"].dropna().astype(int).value_counts().items():
                rows.append(dict(group=g, state=s, metric="converged", value=str(v), count=int(n)))
            if g != "test":  # test labels stay sealed
                for v, n in q[f"{s}_class"].value_counts().items():
                    rows.append(dict(group=g, state=s, metric="class", value=v, count=int(n)))
    return pd.DataFrame(rows)


def write_xlsx(T, data, test, hide, path):
    """CABD_dataset.xlsx (needs openpyxl): data, test_results, summary, notes (kept), columns; grey = planned/waiting."""
    from openpyxl.styles import Font, PatternFill
    notes = pd.read_excel(path, sheet_name="notes", dtype=str) if path.exists() else pd.DataFrame(columns=["geom_id", "note"])
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        data.to_excel(w, sheet_name="data", index=False)
        test.to_excel(w, sheet_name="test_results", index=False)
        summary_table(T.assign(**{c: data[c] for c in hide})).to_excel(w, sheet_name="summary", index=False)
        notes.to_excel(w, sheet_name="notes", index=False)
        columns_table().to_excel(w, sheet_name="columns", index=False)
        book = w.book
        planned, waiting = PatternFill("solid", fgColor="BFBFBF"), PatternFill("solid", fgColor="EDEDED")
        for ws in book.worksheets:
            for cell in ws[1]:
                cell.font = Font(bold=True)
            ws.freeze_panes = "B2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(44, max(10, len(str(col[0].value)) + 3))
        for name, frame in (("data", data), ("test_results", test)):
            ws = book[name]
            for s in STATES:
                first = COLUMNS.index(f"{s}_converged") + 1
                for i, v in enumerate(frame[f"{s}_run_status"].to_numpy()):
                    fill = planned if v == "planned" else waiting if v == "not_run" else None
                    if fill:
                        for col in range(first, first + len(FIELDS) - 1):
                            ws.cell(row=i + 2, column=col + 1).fill = fill


def viewer_data(T, raw=None):
    """Curve, energy, stress and landmark positions of the shown step, per finished state; test states kept apart."""
    assign = assignment("base")
    sources = {pc: Results(pc, raw) for pc in (1, 2, 3)}
    shared, sealed = {}, {}
    r6 = lambda values: [round(float(x), 6) for x in values]
    for _, r in T.iterrows():
        for s in STATES:
            if r[f"{s}_run_status"] != "finished":
                continue
            d = load_state(sources[int(assign.loc[r.geom_id, "pc"])], r.geom_id, s)
            if d is None:
                continue
            c = d["curve"]
            peak = int(c.seqv_max_MPa.to_numpy(float).argmax())
            item = dict(at=round(float(d["u"]), 6), kind=d["kind"], u=r6(c.u_applied_mm), F=r6(c.fx_disp_N), U=r6(c.U_total_mJ),
                        S=r6(c.seqv_max_MPa), S_body=str(c.seqv_max_body.iloc[peak]),
                        points={k: r6([*a, *b]) for k, (a, b) in d["points"].items()})
            (sealed if r.group == "test" else shared).setdefault(r.geom_id, {})[s] = item
    return shared, sealed


def write_json_gz(obj, path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="ascii") as f:
        json.dump(obj, f, **kw)


def base(out, raw=None, xlsx=False):
    T = build_table(collect_base(raw))
    data, test, hide = split_test(T)
    write_csv(data, out / "CABD_dataset.csv", encoding="ascii")
    write_csv(test, out / "CABD_test_results.csv", encoding="ascii")
    write_csv(columns_table(), out / "CABD_dataset_columns.csv", encoding="ascii")
    if xlsx:
        write_xlsx(T, data, test, hide, out / "CABD_dataset.xlsx")
    shared, sealed = viewer_data(T, raw)
    write_json_gz(shared, out / "curves" / "curves_base.json.gz", separators=(",", ":"))
    write_json_gz(sealed, out / "curves" / "curves_test.json.gz", separators=(",", ":"))
    print(f"base: {data.shape} | curves: {len(shared)} shared, {len(sealed)} sealed -> {out}")
    return T


# ----------------------------------------------------------------------------------------------- ext dataset
def ext_sources(raw=None):
    """Results of PC4-9, then retry1 last (its records replace the failed PC4-6 runs)."""
    from cabd.ansys.results import zips
    labels = [pc for pcs in EXT_STAGES.values() for pc in pcs] + [lb for lb in EXT_RETRY if lb in zips(raw)]
    return [Results(lb, raw) for lb in labels]


def collect_ext(raw=None):
    rows = {}
    for res in ext_sources(raw):
        for _, x in latest(res.read("progress.csv")).iterrows():
            r = {"status": x.status, "mesh_rule": x.mesh_rule, "nodes": x.nodes, "source": res.name}
            if x.status in ("ok", "not_converged"):
                r.update(scalars(res, x.geom_id, x.state))
                if r.get("curve") == "first":
                    found = first_solve_mesh([res.own.bytes(f"{x.geom_id}/wb_log.txt").decode("utf-8-sig", "replace")], x.state)
                    if found is None:
                        raise SystemExit(f"first-solve mesh not found: {x.geom_id} {x.state}")
                    r["mesh_rule"], r["nodes"] = found
            rows[x.geom_id, x.state] = r
    return rows


def ext_curves(ids, raw=None):
    """Force-displacement curves of the extra geometries for the ROM fit, same rule as the shared viewer: of the final
    and first-attempt curves the one that reached further, u_applied_mm and fx_disp_N, no origin."""
    out = {}
    for res in ext_sources(raw):
        for _, x in latest(res.read("progress.csv")).iterrows():
            if x.status not in ("ok", "not_converged"):
                continue
            best = best_curve(res, x.geom_id, x.state)
            if best is not None:
                c = best[1]
                out.setdefault(x.geom_id, {})[x.state] = {"u": c.u_applied_mm.round(6).tolist(), "F": c.fx_disp_N.round(6).tolist()}
    missing = [g for g in ids if set(out.get(g, {})) != {"partial", "full"}]
    assert not missing, missing
    return {g: out[g] for g in ids}


def ext(out, raw=None):
    rows = collect_ext(raw)
    recs = []
    for stage in EXT_STAGES:
        a = assignment(stage).reset_index()
        recs += [_row(r, f"train_{stage}", SEEDS["train"], rows) for _, r in a.iterrows()]
    T = _frame(recs)
    write_csv(T, out / "CABD_dataset_ext.csv", encoding="ascii")
    for g, q in T.groupby("group"):
        both = (q.partial_run_status == "finished") & (q.full_run_status == "finished")
        print(f"{g}: {len(q)} geometries, both states finished {int(both.sum())}, "
              f"partial class {q.partial_class.value_counts().to_dict()}, failed/not_run {sorted(q.geom_id[~both])}")
    write_json_gz(ext_curves(list(T.geom_id), raw), out / "curves" / "curves_ext.json.gz")
    return T


# ----------------------------------------------------------------------------------------------- geometry table
def geometry_table(out, raw=None, final="built"):
    """geometry_table.csv: design values, status and scalars of every base geometry (no queued-state rule).

    final = "built": the final kit's results as they were right after the kit was built (2026-11-08 23:53: only the
    12 seeded not_converged states). This reproduces data/geometry_table.csv, which was written at 23:55 that day and
    never rebuilt, so it does not contain what the final kit solved later (CABD_dataset.csv does).
    final = "run": the final kit's results zip (up to date)."""
    rows, sources = {}, []

    def add(res, p):
        for _, x in latest(p).iterrows():
            r = {"status": x.status, "mode": x["mode"], "mesh_rule": x.mesh_rule}
            if x.status in ("ok", "not_converged"):
                r.update(scalars(res, x.geom_id, x.state))
            rows[x.geom_id, x.state] = r

    if final == "built":
        _, files, _, _ = K.final_plan(raw)
        overlay = MemoryArchive({k[len("results/"):]: v for k, v in files.items()}, "final kit as built")
    else:
        overlay = True
    for pc in (1, 2, 3):
        res = Results(pc, raw, overlay)
        p = res.read("progress.csv")
        sources.append(f"PC{pc}: {res.name} (last {p.finished.iloc[-1]})")
        add(res, p)
    p = Results(1, raw, overlay).final
    p = None if p is None else p.bytes("progress.csv")
    if p is not None:
        # states finished by the final kit replace what the three PCs recorded; the readers prefer its files
        p = pd.read_csv(io.BytesIO(p))
        sources.append(f"final kit ({final}): {len(p)} rows")
        add(Results(1, raw, overlay), p)
    T = assignment("base").reset_index()[["geom_id", "role", "test_stage", "pc", *[v for v, _, _ in TABLE_VARS]]].copy()
    for s in STATES:
        for k in TABLE_RESULTS:
            T[f"{s}_{k}"] = [rows.get((g, s), {}).get(k, "not_run" if k == "status" else np.nan) for g in T.geom_id]
    write_csv(T, out / "geometry_table.csv", encoding="utf-8-sig", float_format="%.6g")
    print("saved", out / "geometry_table.csv", len(T), "geometries"); print("\n".join(sources))
    return T


def show(gid, T):
    if gid not in T.index:
        raise SystemExit(f"unknown geometry: {gid}")
    r = T.loc[gid]
    print(f"\n{gid} | {'학습용' if r.role == 'train' else '시험용'} | PC{r.pc}")
    for v, ko, unit in TABLE_VARS:
        print(f"  {v:10s} {r[v]:>9.4g} {unit:4s} {ko}")
    fmt = lambda x, n=3: "-" if pd.isna(x) else f"{x:.{n}f}"
    for s in STATES:
        print(f"  [{s}] {r[s + '_status']} | {TABLE_CLS.get(r[s + '_class'], '-')} | 메시 {r[s + '_mesh_rule']} | 곡선 끝 {fmt(r[s + '_u_end_mm'], 2)} mm ({r[s + '_curve']})")
        print(f"      F_min {fmt(r[s + '_F_min_N'])} N | dE_f {fmt(r[s + '_dE_f_mJ'])} mJ | dE_r {fmt(r[s + '_dE_r_mJ'])} mJ | u2 {fmt(r[s + '_u2_mm'], 2)} mm | theta2 {fmt(r[s + '_theta2_deg'], 1)} deg")


# ----------------------------------------------------------------------------------------------- shared viewer
def share(data_dir, dest=BUILD / "share"):
    """Folder (and zip) for the team: viewer, dataset and training curves, never test results."""
    from cabd.ansys import viewer
    from cabd.geometry import params
    folder = dest / "CABD_viewer"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    shutil.copy2(data_dir / "curves" / "curves_base.json.gz", folder / "CABD_viewer_data.json.gz")
    for name in ("CABD_dataset.csv", "CABD_dataset_columns.csv"):
        shutil.copy2(data_dir / name, folder / name)
    here = Path(viewer.__file__).parent
    for src, name in ((here / "app.py", "viewer.py"), (here / "viewer.html", "viewer.html"), (here / "README.md", "README.md"),
                      (Path(params.__file__), "param_geom_v3.py")):
        shutil.copy2(src, folder / name)
    (folder / "Geometry_Viewer.bat").write_bytes(
        b'@echo off\r\nrem Double-click to open the geometry viewer in your browser. Close this window to stop it.\r\n'
        b'python -X utf8 "%~dp0viewer.py" %*\r\nif errorlevel 1 pause\r\n')
    assert not any("test_results" in p.name or "viewer_test" in p.name for p in folder.iterdir()), "test data in the share folder"
    z = shutil.make_archive(str(dest / folder.name), "zip", root_dir=dest, base_dir=folder.name)
    print(f"share folder {folder} | zip {z} ({Path(z).stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["base", "ext", "table", "share", "all", "show"])
    ap.add_argument("ids", nargs="*", help="show: geometry ids")
    ap.add_argument("--out", type=Path, default=BUILD / "dataset")
    ap.add_argument("--raw", type=Path, help="raw result zips (default ANSYS_RAW)")
    ap.add_argument("--xlsx", action="store_true", help="base: also CABD_dataset.xlsx (needs openpyxl)")
    ap.add_argument("--final", choices=["built", "run"], default="built",
                    help="table: final-kit results as built (reproduces data/geometry_table.csv) or as run (up to date)")
    a = ap.parse_args()
    if a.what == "show":
        path = a.out / "geometry_table.csv"
        table = pd.read_csv(path if path.exists() else DATA / "geometry_table.csv", encoding="utf-8-sig").set_index("geom_id")
        for g in a.ids:
            show(g, table)
        return
    if a.what in ("base", "all"):
        base(a.out, a.raw, a.xlsx)
    if a.what in ("ext", "all"):
        ext(a.out, a.raw)
    if a.what in ("table", "all"):
        geometry_table(a.out, a.raw, a.final)
    if a.what in ("share", "all"):
        share(a.out if (a.out / "CABD_dataset.csv").exists() else DATA)


if __name__ == "__main__":
    main()
