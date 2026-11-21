"""Build the Ansys kits described in cabd.ansys.kits into build/kits/<kit>/ and build/kit_zips/<kit>.zip.

Never runs Ansys: it only writes kit folders (see cabd.ansys.kits for what a kit is and for every historical kit).

Landmarks (for angles and the deformed shape): every vertex of the 6 body outlines on the y = 6 face (arcs as
start / mid / end, params.bodies(p, n=3)), plus the shared project's probe points A = arm top-face root (-L, H),
B = end of the lower slot fillet (-(t_h + R_slot), H) on the upper arm and their mirrors on the lower arm.
jobs.csv and landmarks.csv are written with %.9g, the assignment tables with %.6g.

STEP files come from build/step/<set>/ (set = base, ext1, ext2, verify, verify2); missing ones are generated with the
CAD python (cabd.geometry.step, CABD_CAD_PYTHON). --no-step skips them (kit folders without step files, for checks).

The verification kits rebuild their parameter tables (build/sampling/params_verify.csv, params_verify2.csv) from the
candidate lists in results/inverse (cabd.verify.candidates) and the STEP files from those tables.

The final kit (single PC, built 2026-11-08) finishes every state still missing on PC1-3, read from their raw
results (PC1 and the newest zips of PC2/PC3; the originals are only read). States whose first run left a partial curve
and whose coarser retry could not produce a result fail the same way every time, so they are recorded as
not_converged in the new kit's results (with their first-attempt files and the status they were reclassified from)
and are not solved again. Its jobs.csv has a `states` column; its collect.ps1 is PC1's collect.ps1
(ansys_kit/final/collect_pc1.ps1) with per-job state lists, the job's own pc in the records, the two ext fixes and an
optional -NoZip; it adds overnight.ps1 / run_overnight.bat (ansys_kit/final: run every remaining state unattended).

usage: python -m cabd.ansys.kit <group> [<group> ...] [--no-step] [--no-zip] [--out DIR]
  groups: base ext1 ext2 retry1 final verify verify_all verify2 (see cabd.ansys.kits), or all
"""
import argparse
import io
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cabd.ansys.kits import CONFIG, GROUPS, METHODS, SOLVER_FILES
from cabd.ansys.results import Results
from cabd.config import ANSYS_KIT, BUILD, SAMPLING
from cabd.geometry import params as pg
from cabd.geometry import sampling
from cabd.geometry.step import run_cad

V = pg.VAR_NAMES
CR, LF = "\r", "\n"
JOB_COLS = ["priority", "geom_id", "role", "test_stage"] + V + ["x_fixed_face", "x_displaced_face", "alpha_deg", "est_nodes", "pc"]
STATES = ("partial", "full")
FINAL_STATUS = ("ok", "not_converged")
FINAL_HEADER = "finished,pc,geom_id,state,status,mode,mesh_rule,nodes,n_sets,last_time,job_min"
STEP_TABLES = {"base": ["params_train.csv", "params_test.csv"], "ext1": ["params_train_ext1.csv"],
               "ext2": ["params_train_ext2.csv"]}


# ----------------------------------------------------------------------------------------------- kit contents
def landmarks(gid, p):
    d = pg.derived(p)
    rows = []
    for body, (_, P) in pg.bodies(p, n=3).items():
        keep = [P[0]]
        for q in P[1:]:
            if np.linalg.norm(q - keep[-1]) > 1e-9:
                keep.append(q)
        if np.linalg.norm(keep[-1] - keep[0]) < 1e-9:
            keep.pop()
        rows += [dict(name=f"{body}_v{i:02d}", x=q[0], z=q[1]) for i, q in enumerate(keep)]
    xa, xb, H = -p["L"], -(p["t_h"] + d["R_slot"]), d["H"]
    rows += [dict(name="A_upper", x=xa, z=H), dict(name="B_upper", x=xb, z=H),
             dict(name="A_lower", x=xa, z=-H), dict(name="B_lower", x=xb, z=-H)]
    return [dict(geom_id=gid, **r) for r in rows]


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def _edit_ps1(raw, subs, all_states=False):
    eol = "\r\n" if "\r\n" in raw else "\n"
    t = raw.replace("\r\n", "\n")
    if all_states:
        assert t.count("$STATES") == 2, t.count("$STATES")
        t = t.replace("$STATES", "$ALL_STATES")  # str.replace is case-sensitive, so $states of each job stays
    for old, new in subs:
        t = replace_once(t, old, new)
    return t.replace("\n", eol).encode("utf-8")


_NOT_CONVERGED_FIX = (
    'if ($c.Count) { $info.auto_failed_last_time = $c[-1].time } }\n                }\n            }\n',
    'if ($c.Count) { $info.auto_failed_last_time = $c[-1].time } }\n                }\n            }\n'
    '            # the first run left a usable curve and the coarser retry produced no result: this repeats on every run\n'
    '            if ($af -and $info.auto_failed_extract -eq "ok" -and $info.status -eq "error: no result file") { $info.status = "not_converged" }\n')
_START_FAILED_FIX = ('-Pattern "^$state solver_start_failed" -Quiet', '-Pattern "^$state solver_start_failed mode=auto$" -Quiet')


def collect_fixed(kit_dir=ANSYS_KIT):
    """collect.ps1 of the ext / retry / verification kits: ansys_kit/collect.ps1 plus the two fixes."""
    raw = (kit_dir / "collect.ps1").read_bytes().decode("utf-8")
    return _edit_ps1(raw, [_START_FAILED_FIX, _NOT_CONVERGED_FIX], all_states=True)


def collect_final(kit_dir=ANSYS_KIT):
    """collect.ps1 of the final kit: PC1's collect.ps1 with per-job states, the job's pc, the fixes and -NoZip."""
    raw = (kit_dir / "final" / "collect_pc1.ps1").read_bytes().decode("utf-8")
    subs = [
        ('[string]$PostState = "")', '[string]$PostState = "", [switch]$NoZip)'),
        ('$BODY_NAMES = @(', '$WANTED = @{}\nforeach ($j in $jobs) { $WANTED[$j.geom_id] = @($j.states -split \';\') }\n$BODY_NAMES = @('),
        ('foreach ($s in $ALL_STATES) {', 'foreach ($s in $WANTED[$id]) {'),
        ('pc = $pc; computer', 'pc = $job.pc; computer'),
        ('$info.finished, $pc, $id', '$info.finished, $job.pc, $id'),
        _START_FAILED_FIX, _NOT_CONVERGED_FIX,
        ('& (Join-Path $kit "make_result_zip.ps1")', 'if (-not $NoZip) { & (Join-Path $kit "make_result_zip.ps1") }'),
    ]
    return _edit_ps1(raw, subs)


_INTRO = "학습·시험용 형상 768개를 PC 3대에 나눠 자동으로 해석하는 키트입니다. 이 폴더에는 **이 PC가 맡은 형상(약 256개)**만 들어 있습니다."
_TIME = "따라서 256개에 **약 2~4일**이 걸립니다(추정)."
_ORDER = "- 순서: 학습용 형상을 먼저 돌리고, 그다음 시험용 형상을 돌립니다."
_FOLDER = r"예: `C:\CABD_collect_PC1`"
_TITLE = "# CABD 데이터 수집 키트"


def readme(group, label, n, total=None, first=None, last=None, n_pcs=1, kit_dir=ANSYS_KIT):
    """README.md of one kit: ansys_kit/README.md with the kit's own intro, time estimate, order and folder name.
    The kit README is the Korean guide for the person running the kit."""
    t = (kit_dir / "README.md").read_text(encoding="utf-8")
    g = GROUPS[group]
    if g["kind"] == "base":
        return None   # unchanged file
    if g["kind"] == "ext":
        pcs = [int(k.rsplit("PC", 1)[1]) for k in g["kits"]]
        t = replace_once(t, _INTRO, f"추가 학습용 형상 {total}개({first}~{last})를 PC 3대(PC{pcs[0]}~{pcs[-1]})에 나눠 자동으로 해석하는 키트입니다. "
                                    f"이 폴더에는 **이 PC가 맡은 형상({n}개)**만 들어 있습니다.")
        t = replace_once(t, _TIME, f"따라서 {n}개에 **약 1.5~3일**이 걸립니다(추정).")
        t = replace_once(t, _ORDER, "- 순서: 형상 번호 순서대로 돌립니다.")
        return replace_once(t, _FOLDER, rf"예: `C:\CABD_collect_PC{label}`")
    if g["kind"] == "retry":
        geoms = g["geoms"]
        t = replace_once(t, _TITLE, "# CABD 재해석 키트 (ext1 형상 5개)")
        t = replace_once(t, _INTRO, f"추가 학습용 형상 중 PC5·PC6에서 솔버가 시작되지 못한 {len(geoms)}개({', '.join(geoms)})를 다시 해석하는 키트입니다. "
                                    "해석 설정은 데이터 수집 키트와 같습니다.")
        t = replace_once(t, _TIME, f"따라서 {len(geoms)}개에 **약 1~2시간**이 걸립니다(추정).")
        t = replace_once(t, _ORDER, "- 순서: 형상 번호 순서대로 돌립니다.")
        return replace_once(t, _FOLDER, rf"예: `C:\CABD_collect_{label}`")
    # verification kits
    t = replace_once(t, _TITLE, "# CABD 역설계 Ansys 검증 키트")
    t = replace_once(t, _INTRO, f"역설계 방법 4개가 낸 후보 형상 {total}개(목표 40개 × 방법 4개)를 Ansys로 검증하는 키트입니다. "
                     + (f"PC {n_pcs}대에 나눴고, 이 폴더에는 **이 PC가 맡은 형상({n}개)**만 들어 있습니다." if n_pcs > 1 else "PC 1대에서 모두 돌립니다.")
                     + " 해석 방식과 실행 방법은 데이터 수집 키트와 같습니다.")
    t = replace_once(t, _TIME, f"따라서 {n}개에 **약 {n * 10 / 60:.0f}~{n * 20 / 60:.0f}시간**이 걸립니다(추정).")
    t = replace_once(t, _ORDER, "- 순서: 목표 번호 순서대로, 같은 목표의 방법 4개를 이어서 돌립니다.")
    t = replace_once(t, _FOLDER, rf"예: `C:\CABD_verify_{label}`")
    if g["kit"] == 2:
        t = t.replace("# CABD 역설계 Ansys 검증 키트", "# CABD 역설계 Ansys 검증 키트 2 (1,024·1,536개 학습 모델)")
        t = t.replace(f"역설계 방법 4개가 낸 후보 형상 {total}개(목표 40개 × 방법 4개)",
                      f"1,024개와 1,536개로 다시 학습한 모델의 역설계 후보 형상 {total}개(학습 크기 2개 × 목표 40개 × 방법 4개)")
        t = t.replace("- 순서: 목표 번호 순서대로, 같은 목표의 방법 4개를 이어서 돌립니다.",
                      "- 순서: 목표 번호 순서대로, 같은 목표의 후보 8개(크기 2개 × 방법 4개)를 이어서 돌립니다.")
        t = t.replace(rf"C:\CABD_verify_{label}", rf"C:\CABD_verify2_{label}")
    return t


def config_json(version, label, n_jobs):
    """kit_config.json text (version None: 'new (<today>)', for a kit that is not a historical one)."""
    version = version or f"new ({time.strftime('%Y-%m-%d')})"
    return crlf(json.dumps(dict(kit_version=version, **CONFIG, pc=label, n_jobs=n_jobs), indent=2))


def crlf(text):
    """Windows line ends, as every historical kit file has (the kits run on Windows only)."""
    return text.replace(CR + LF, LF).replace(LF, CR + LF)


# ----------------------------------------------------------------------------------------------- job tables
def read_table(name, sampling_dir=SAMPLING):
    return pd.read_csv(sampling_dir / name)


def assignment(group, sampling_dir=SAMPLING):
    """Job order of a collection group with priority and pc (the assignment_*.csv of the original dist folder)."""
    g = GROUPS[group]
    if g["kind"] == "base":
        tr, te = read_table("params_train.csv", sampling_dir), read_table("params_test.csv", sampling_dir)
        order = pd.concat([tr, te[te.test_stage == 1], te[te.test_stage == 2]], ignore_index=True)
        order["priority"] = np.arange(len(order))
        order["pc"] = order.priority % len(g["kits"]) + 1
        return order
    if g["kind"] == "ext":
        stage = g["stage"]
        order = read_table(f"params_train_ext{stage}.csv", sampling_dir)
        assert len(order) == 512 and order.geom_id.is_unique
        first = 768 + 512 * (stage - 1)
        pcs = [int(k.rsplit("PC", 1)[1]) for k in g["kits"]]
        order["priority"] = first + np.arange(len(order))
        order["pc"] = (order.priority - first) % len(pcs) + pcs[0]
        return order
    raise ValueError(group)


def _csv(df, **kw):
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator=CR + LF, **kw)
    return buf.getvalue()


class Kit:
    """Contents of one kit folder (written by `write`)."""

    def __init__(self, name, label, jobs, group, readme_text, collect, version, step_dir, extra=None):
        self.name, self.label, self.jobs, self.group = name, label, jobs, group
        self.readme, self.collect, self.version, self.step_dir = readme_text, collect, version, step_dir
        self.extra = extra or {}                    # relative path -> bytes (final kit: seeded results, overnight files)
        self.jobs_csv = _csv(jobs[JOB_COLS], float_format="%.9g")
        lm = []
        for _, r in jobs.iterrows():
            lm += landmarks(r.geom_id, dict(pg.REF, **{v: float(r[v]) for v in V}))
        self.landmarks_csv = _csv(pd.DataFrame(lm), float_format="%.9g")

    def files(self):
        f = {name: (ANSYS_KIT / name).read_bytes() for name in SOLVER_FILES}
        f["collect.ps1"] = self.collect
        # base kits: ansys_kit/README.md byte for byte; the others: rewritten text with CRLF line ends
        f["README.md"] = (ANSYS_KIT / "README.md").read_bytes() if self.readme is None else crlf(self.readme).encode("utf-8")
        f["kit_config.json"] = config_json(self.version, self.label, len(self.jobs)).encode("ascii")
        f["jobs.csv"] = self.jobs_csv.encode("utf-8")
        f["landmarks.csv"] = self.landmarks_csv.encode("utf-8")
        return f


def write(kit, out, zip_dir=None, step=True):
    """Write the kit folder (clearing files of an older build first) and optionally its zip."""
    folder = out / kit.name
    if folder.exists():
        for f in folder.rglob("*"):   # clear files only (folders may be held open by Explorer)
            if f.is_file():
                f.unlink()
    (folder / "step").mkdir(parents=True, exist_ok=True)
    for name, data in {**kit.files(), **kit.extra}.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)   # bytes carry their final (CRLF) line ends
    if step:
        for gid in kit.jobs.geom_id:
            shutil.copy2(kit.step_dir / f"{gid}.step", folder / "step")
    if zip_dir is not None:
        zip_dir.mkdir(parents=True, exist_ok=True)
        return Path(shutil.make_archive(str(zip_dir / kit.name), "zip", root_dir=out, base_dir=kit.name))
    return None


# ----------------------------------------------------------------------------------------------- STEP sets
def step_dir(name, build=BUILD):
    return build / "step" / name


def ensure_step(name, ids, tables, build=BUILD):
    """Make sure build/step/<name>/<id>.step exists for every id (generate the set with the CAD python if not)."""
    d = step_dir(name, build)
    if all((d / f"{g}.step").exists() for g in ids):
        return d
    print(f"building STEP set {name} ({len(ids)} needed) with the CAD python ...", flush=True)
    r = run_cad("build", d, *tables)
    print(r.stdout.strip())
    log = pd.read_csv(d / "step_build_log.csv")
    assert log.area_diff.abs().max() < 1e-4, log.area_diff.abs().max()   # earlier tables: 2e-5
    return d


# ----------------------------------------------------------------------------------------------- groups
def collection_kits(group, sampling_dir=SAMPLING, build=BUILD):
    """(kits, assignment table, assignment file name) of base / ext groups."""
    g = GROUPS[group]
    order = assignment(group, sampling_dir)
    sd = step_dir(g["step"], build)
    kits = []
    for name in g["kits"]:
        label = int(name.rsplit("PC", 1)[1])
        jobs = order[order.pc == label]
        if g["kind"] == "base":
            text, collect = readme(group, label, len(jobs)), (ANSYS_KIT / "collect.ps1").read_bytes()
        else:
            text = readme(group, label, len(jobs), len(order), order.geom_id.iloc[0], order.geom_id.iloc[-1])
            collect = collect_fixed()
        kits.append(Kit(name, label, jobs, group, text, collect, g["version"], sd))
    return kits, order


def retry_kits(group, sampling_dir=SAMPLING, build=BUILD):
    g = GROUPS[group]
    order = assignment(g["from_group"], sampling_dir)
    # the original read assignment_ext1.csv back (%.6g); the tables are %.6g already, so the values are the same
    a = order.set_index("geom_id").loc[g["geoms"]].reset_index()
    a["pc"] = g["label"]
    return [Kit(g["kits"][0], g["label"], a, group, readme(group, g["label"], len(a)), collect_fixed(), g["version"],
                step_dir(g["step"], build))], None


def verify_candidates(kit_no, inverse=None):
    """Candidates of verification kit 1 or 2.

    From the inverse-design result csvs (full precision, as the original builder) when they are in `inverse`
    (default results/inverse); otherwise from the candidate list results/inverse/ansys_verify*_candidates.csv, whose
    %.9g rounding can move the 9th digit of a few jobs.csv / landmarks.csv values (geometry unchanged)."""
    from cabd.verify import candidates as C
    inverse = Path(inverse or C.INVERSE)
    need = [inverse / f"{m}_final_n20_s{C.SEED}.csv" for m in METHODS] if kit_no == 1 else \
        [inverse / "v2" / f"{m}_final_train{n}_n20_s{C.SEED}.csv" for n in (1024, 1536) for m in METHODS]
    if all(p.exists() for p in need):
        return C.kit1(inverse) if kit_no == 1 else C.kit2(inverse, pd.read_csv(C.INVERSE / C.FILES[1]))
    print(f"inverse result csvs not found in {inverse}: using the candidate list {C.FILES[kit_no]} (9 digits)")
    return pd.read_csv(C.INVERSE / C.FILES[kit_no], keep_default_na=False, na_values=[""])


def verify_kits(group, inverse=None, build=BUILD):
    """Verification kits: parameter table (written to build/sampling), job order and per-PC blocks of whole queries."""
    g = GROUPS[group]
    c = verify_candidates(g["kit"], inverse)
    P = sampling.table_from_candidates(c)
    params_file = build / "sampling" / ("params_verify.csv" if g["kit"] == 1 else "params_verify2.csv")
    params_file.parent.mkdir(parents=True, exist_ok=True)
    params_file.write_text(_csv(P, float_format=sampling.FLOAT_FORMAT), encoding="utf-8", newline="")
    keys = ["solve_id", "query_id", "method"] + (["train_size"] if g["kit"] == 2 else [])
    order = P.merge(c.drop_duplicates("solve_id")[keys], left_on="geom_id", right_on="solve_id")
    order["m_rank"] = order.method.map(list(METHODS).index)
    order = order.sort_values(["query_id", "train_size", "m_rank"] if g["kit"] == 2 else ["query_id", "m_rank"]).reset_index(drop=True)
    order["priority"] = np.arange(len(order))
    labels = [k.rsplit("_", 1)[1] for k in g["kits"]]
    qs = sorted(order.query_id.unique())
    block = {q: labels[k] for k, part in enumerate(np.array_split(np.arange(len(qs)), len(labels))) for q in np.array(qs)[part]}
    order["pc"] = order.query_id.map(block)
    sd = step_dir(g["step"], build)
    kits = []
    for name, label in zip(g["kits"], labels):
        jobs = order[order.pc == label]
        kits.append(Kit(name, label, jobs, group, readme(group, label, len(jobs), len(order), n_pcs=len(labels)),
                        collect_fixed(), g["version"], sd))
    return kits, params_file


# ----------------------------------------------------------------------------------------------- final kit
def _status_of(src, gid, state):
    b = src.own.bytes(f"{gid}/{state}_status.txt")
    if b is None:
        return "not_run", None
    text = b.decode("ascii", "replace")
    return re.search(r"^status=(.*?)\s*$", text, re.M).group(1), text


def final_plan(raw=None, sampling_dir=SAMPLING):
    """What the final kit had to do: (jobs table with `states`, seeded result files, progress lines, landmarks csv).

    Sources: the PC1-3 kits (jobs.csv / landmarks.csv as written, read back like the original read them from the kit
    folders) and the raw results of PC1-3 without the final kit's overlay (the final kit did not exist yet)."""
    kits, _ = collection_kits("base", sampling_dir)
    jobs_out, lms, files, progress = [], [], {}, [FINAL_HEADER]
    seeded = []
    for kit in kits:
        pc = kit.label
        src = Results(pc, raw, overlay=False)
        src_jobs = pd.read_csv(io.StringIO(kit.jobs_csv))
        src_lm = pd.read_csv(io.StringIO(kit.landmarks_csv))
        for _, job in src_jobs.iterrows():
            gid, todo = job.geom_id, []
            for state in STATES:
                status, text = _status_of(src, gid, state)
                if status in FINAL_STATUS:
                    continue
                curve = src.own.bytes(f"{gid}/{state}_autofailed_curve.csv")
                repeatable = curve is not None and (status == "error: no result file" or status == "start_error"
                                                    and b"solver_start_failed mode=auto_coarse" in src.own.bytes(f"{gid}/wb_log.txt"))
                if not repeatable:
                    todo.append(state)
                    continue
                for name in src.own.listdir(gid, f"{state}_autofailed"):
                    files[f"results/{gid}/{name}"] = src.own.bytes(f"{gid}/{name}")
                note = text.replace(f"status={status}", "status=not_converged", 1).rstrip() + f"\r\nreclassified_from={status}\r\n"
                files[f"results/{gid}/{state}_status.txt"] = note.encode("ascii")
                info = dict(l.split("=", 1) for l in text.splitlines() if "=" in l)
                c = pd.read_csv(io.BytesIO(curve))
                progress.append(",".join(str(v) for v in (info.get("finished", ""), pc, gid, state, "not_converged", info.get("mode", ""),
                                                          info.get("mesh_rule", ""), info.get("nodes", ""), len(c), c.time.iloc[-1], info.get("job_min", ""))))
                seeded.append((pc, gid, state, status))
            if todo:
                row = job.to_dict()
                row["pc"], row["states"] = pc, ";".join(todo)
                jobs_out.append(row)
                lms.append(src_lm[src_lm.geom_id == gid])
    J = pd.DataFrame(jobs_out).sort_values("priority")
    files["results/progress.csv"] = crlf("\n".join(progress) + "\n").encode("ascii")
    return J, files, _csv(pd.concat(lms)), seeded


class FinalKit(Kit):
    def __init__(self, raw=None, sampling_dir=SAMPLING, build=BUILD):
        g = GROUPS["final"]
        J, files, lm_csv, self.seeded = final_plan(raw, sampling_dir)
        self.name, self.label, self.jobs, self.group = g["kits"][0], "final", J, "final"
        self.version, self.step_dir = g["version"], step_dir(g["step"], build)
        self.jobs_csv, self.landmarks_csv = _csv(J), lm_csv
        self.extra = files

    def files(self):
        f = {name: (ANSYS_KIT / name).read_bytes() for name in SOLVER_FILES if name != "run_collect.bat"}
        f["collect.ps1"] = collect_final()
        for name in ("overnight.ps1", "run_overnight.bat"):
            f[name] = (ANSYS_KIT / "final" / name).read_bytes()
        f["kit_config.json"] = config_json(self.version, "final", len(self.jobs)).encode("ascii")
        f["jobs.csv"] = self.jobs_csv.encode("utf-8")
        f["landmarks.csv"] = self.landmarks_csv.encode("utf-8")
        return f


# ----------------------------------------------------------------------------------------------- main
def build_group(group, out=None, zip_dir=None, step=True, raw=None, build=BUILD, inverse=None):
    """Build one group; returns the Kit objects (folders written to out, zips to zip_dir if given)."""
    out = out or build / "kits"
    g = GROUPS[group]
    out.mkdir(parents=True, exist_ok=True)
    if g["kind"] in ("base", "ext"):
        kits, order = collection_kits(group, build=build)
        (out / g["assignment"]).write_text(_csv(order[JOB_COLS], float_format="%.6g"), encoding="utf-8", newline="")
        tables = STEP_TABLES[g["step"]]
    elif g["kind"] == "retry":
        kits, _ = retry_kits(group, build=build)
        tables = STEP_TABLES[g["step"]]
    elif g["kind"] == "verify":
        kits, params_file = verify_kits(group, inverse, build=build)
        tables = [params_file]
    else:
        kits = [FinalKit(raw, build=build)]
        tables = STEP_TABLES[g["step"]]
    if step:
        ensure_step(g["step"], [gid for k in kits for gid in k.jobs.geom_id], tables, build)
    for kit in kits:
        z = write(kit, out, zip_dir if g["kind"] != "final" else None, step)
        print(f"{kit.name}: {len(kit.jobs)} jobs" + (f", zip {z.stat().st_size / 1e6:.1f} MB" if z else ""), flush=True)
    return kits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+", choices=[*GROUPS, "all"])
    ap.add_argument("--out", type=Path, default=BUILD / "kits")
    ap.add_argument("--zip-dir", type=Path, default=BUILD / "kit_zips")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--no-step", action="store_true")
    ap.add_argument("--inverse-dir", type=Path, help="inverse result csvs for the verification kits (default results/inverse)")
    ap.add_argument("--raw", type=Path, help="raw result zips for the final kit (default ANSYS_RAW)")
    a = ap.parse_args()
    for group in (list(GROUPS) if "all" in a.groups else a.groups):
        build_group(group, a.out, None if a.no_zip else a.zip_dir, not a.no_step, a.raw, inverse=a.inverse_dir)


if __name__ == "__main__":
    main()
