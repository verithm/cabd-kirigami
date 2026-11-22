import argparse
import csv
import io
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

GRID = np.linspace(0, 15, 61)
FEATURES = "t_h L_h t_a L_a L t_f c rho".split()
STATES = ("partial", "full")


def read_rows(archive, member):
    return list(csv.DictReader(io.StringIO(archive.read(member).decode("utf-8-sig"))))


def write_rows(path, rows, fields=None):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields if fields is not None else list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stability(force, complete, tolerance):
    positive = np.flatnonzero(force > tolerance)
    negative = np.flatnonzero(force < -tolerance)
    if not len(negative):
        return ("monostable" if complete and len(positive) else "unknown"), None
    if not len(positive) or positive[0] >= negative[0]:
        return "unknown", None
    later = positive[positive > negative[0]]
    if not len(later):
        return "unknown", None
    stop = int(later[0])
    start = int(negative[negative < stop][-1])
    for right in range(start + 1, stop + 1):
        if force[right - 1] < 0 <= force[right]:
            ratio = -force[right - 1] / (force[right] - force[right - 1])
            return "bistable", (right - 1, right, float(ratio))
    return "unknown", None


def arm_angles(archive, members, prefix, left_set, right_set, ratio):
    node_file = members.get(prefix + "_landmark_nodes.csv")
    disp_file = members.get(prefix + "_landmark_disp.csv")
    if not node_file or not disp_file:
        return None
    nodes = {
        r["name"]: (int(r["landmark"]), np.array([float(r["x0_mm"]), float(r["z0_mm"])]))
        for r in read_rows(archive, node_file)
    }
    names = ("A_upper", "B_upper", "A_lower", "B_lower")
    if not all(name in nodes for name in names):
        return None
    ids = {nodes[name][0] for name in names}
    offsets = {}
    for r in read_rows(archive, disp_file):
        step, marker = int(r["set"]), int(r["landmark"])
        if step in (left_set, right_set) and marker in ids:
            offsets[step, marker] = np.array([float(r["ux_mm"]), float(r["uz_mm"])])
    if not all((step, marker) in offsets for step in (left_set, right_set) for marker in ids):
        return None

    initial = [nodes[name][1] for name in names]
    current = [
        p + (1 - ratio) * offsets[left_set, marker] + ratio * offsets[right_set, marker]
        for marker, p in (nodes[name] for name in names)
    ]
    angles = []
    for idx in (0, 2):
        before = initial[idx + 1] - initial[idx]
        after = current[idx + 1] - current[idx]
        if not np.isfinite(after).all() or not np.linalg.norm(after):
            return None
        difference = math.degrees(math.atan2(after[1], after[0]) - math.atan2(before[1], before[0]))
        angles.append((difference + 180) % 360 - 180)
    return angles


def process(archive_path, tolerance):
    with ZipFile(archive_path) as archive:
        members = {n.replace(chr(92), "/"): n for n in archive.namelist()}
        progress = next(member for path, member in members.items() if path.split("/")[-1] == "progress.csv")
        progress_rows = read_rows(archive, progress)
        latest = {(r["geom_id"], r["state"]): r for r in progress_rows}
        for (geom, state), source in sorted(latest.items()):
            member = next((p for p in (f"{geom}/{state}_curve.csv",) if p in members), None)
            row = {
                "geom_id": geom, "state": state, "role": "train" if geom.startswith("TR") else "test",
                "pc": source["pc"], "finished": source["finished"], "status": source["status"],
                "mesh_rule": source["mesh_rule"], "archive": archive_path.name,
                "curve_file": member, "curve_quality": "missing", "last_u_mm": None,
                "force_complete": 0, "stability_label": "unknown", "u2_mm": None,
                "stable_angle_upper_deg": None, "stable_angle_lower_deg": None,
            }
            force_vector = None
            if member:
                samples = read_rows(archive, members[member])
                try:
                    u = np.array([float(r["u_applied_mm"]) for r in samples])
                    f = np.array([float(r["fx_disp_N"]) for r in samples])
                    sets = [int(r["set"]) for r in samples]
                except (KeyError, ValueError):
                    row["curve_quality"] = "invalid_values"
                    yield row, None
                    continue
                valid = (
                    len(u) >= 2 and np.isfinite(u).all() and np.isfinite(f).all()
                    and 0 <= u[0] and np.all(np.diff(u) > 0) and len(sets) == len(set(sets))
                )
                if valid:
                    complete = source["status"] == "ok" and np.isclose(u[-1], 15, atol=1e-6, rtol=0)
                    label, bracket = stability(f, complete, tolerance)
                    row.update(curve_quality="valid", last_u_mm=float(u[-1]), force_complete=int(complete),
                               stability_label=label)
                    if complete:
                        fit_u, fit_f = (np.r_[0, u], np.r_[0, f]) if u[0] > 0 else (u, f)
                        force_vector = np.interp(GRID, fit_u, fit_f)
                    if bracket:
                        left, right, ratio = bracket
                        row["u2_mm"] = float(u[left] + ratio * (u[right] - u[left]))
                        angles = arm_angles(archive, members, member.removesuffix("_curve.csv"),
                                            sets[left], sets[right], ratio)
                        if angles is not None:
                            row["stable_angle_upper_deg"], row["stable_angle_lower_deg"] = angles
                else:
                    row["curve_quality"] = "invalid_grid"
            yield row, force_vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--force-tol", type=float, default=0.01)
    args = parser.parse_args()
    if not math.isfinite(args.force_tol) or args.force_tol <= 0:
        parser.error("--force-tol must be positive and finite")
    jobs = {}
    if args.jobs:
        with args.jobs.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                geom = row["geom_id"]
                features = {key: float(row[key]) for key in FEATURES}
                if geom in jobs or not all(map(math.isfinite, features.values())):
                    parser.error(f"duplicate or invalid jobs row for {geom}")
                jobs[geom] = features
    archives = sorted(args.source.glob("collect_result_PC*.zip"))
    if not archives:
        parser.error("no collect_result_PC*.zip files found")
    records = {}
    for archive in archives:
        for row, vector in process(archive, args.force_tol):
            key = row["geom_id"], row["state"]
            if key in records:
                parser.error(f"duplicate geometry/state across archives: {key}")
            if not re.fullmatch(r"(TR|TE)\d+", key[0]) or key[1] not in STATES:
                parser.error(f"unexpected geometry/state: {key}")
            row.update({key: jobs.get(row["geom_id"], {}).get(key) for key in FEATURES})
            records[key] = row, vector

    index, force, angles = [], [], []
    pairs = defaultdict(dict)
    force_columns = [f"force_N_at_{u:05.2f}_mm" for u in GRID]
    for key, (row, vector) in sorted(records.items()):
        index.append(row)
        pairs[key[0]][key[1]] = row
        common = {key: row[key] for key in ["geom_id", "state", "role", *FEATURES]}
        if vector is not None:
            force.append({**common, **dict(zip(force_columns, vector))})
        if row["stable_angle_upper_deg"] is not None:
            angles.append({**common, **{k: row[k] for k in (
                "u2_mm", "stable_angle_upper_deg", "stable_angle_lower_deg")}})

    paired = []
    for geom, states in sorted(pairs.items()):
        partial, full = (states.get(state, {}) for state in STATES)
        a, b = partial.get("stability_label", "missing"), full.get("stability_label", "missing")
        angle = partial.get("stable_angle_upper_deg")
        paired.append(dict(geom_id=geom, role=next(iter(states.values()))["role"],
                           partial_stability=a, full_stability=b, partial_angle_deg=angle,
                           matches_required_pattern=int(a == "bistable" and b == "monostable"),
                           angle_available=int(angle is not None)))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "case_index.csv", index)
    write_rows(output / "force_curves.csv", force, ["geom_id", "state", "role", *FEATURES, *force_columns])
    write_rows(output / "stable_angles.csv", angles, ["geom_id", "state", "role", *FEATURES, "u2_mm",
                                                    "stable_angle_upper_deg", "stable_angle_lower_deg"])
    write_rows(output / "paired_stability.csv", paired)
    counts = []
    for role in ("train", "test"):
        rows = [r for r in index if r["role"] == role]
        for state in STATES:
            group = [r for r in rows if r["state"] == state]
            labels = Counter(r["stability_label"] for r in group)
            counts.append(dict(role=role, state=state, cases=len(group),
                               missing_or_invalid_curve=sum(r["curve_quality"] != "valid" for r in group),
                               complete_force_curves=sum(r["force_complete"] for r in group),
                               stable_angles=sum(r["stable_angle_upper_deg"] is not None for r in group),
                               **{label: labels[label] for label in ("bistable", "monostable", "unknown")}))
    write_rows(output / "summary.csv", counts)
    print(f"{len(index)} cases, {len(paired)} geometries, {len(force)} force curves, {len(angles)} angles")
    print(f"Matching pairs with angle: {sum(r['matches_required_pattern'] and r['angle_available'] for r in paired)}")
    print(f"Cases joined to jobs: {sum(r['t_h'] is not None for r in index)}")
    (output / "README.md").write_text(
        "# Forward 전처리 데이터\n\n"
        "case_index.csv는 형상·젖음 상태별 최종 기록과 판정입니다. force_curves.csv는 0–15 mm를 "
        "0.25 mm 간격으로 보간한 완료 곡선이며, 원점이 빠진 곡선에는 (0, 0)을 추가했습니다. "
        "stable_angles.csv는 두 번째 안정점에서 초기 자세 대비 위·아래 암의 회전각입니다. "
        "paired_stability.csv는 동일 형상에 대한 partial 쌍안정 및 full 단안정 조합을 집계합니다.\n\n"
        f"힘 판정 허용오차는 임시로 ±{args.force_tol:g} N을 적용했습니다. 원본 곡선에서 음수 구간 뒤 "
        "양수 복귀를 확인하고 그 사이의 영점을 보간했습니다. 단안정은 15 mm 해석 범위 기준입니다. "
        "각도는 같은 해석의 측정점 좌표를 영점에서 보간해 계산했습니다. "
        "첫 해석의 autofailed 곡선은 포함하지 않았습니다. TR/TE 접두어를 train/test로 표시했습니다.\n\n"
        + ("jobs.csv의 형상 변수를 geom_id로 연결했습니다. 매칭되지 않은 값은 비워뒀습니다.\n"
           if args.jobs else "수집본에 jobs.csv가 없어 형상 변수 8개는 비워뒀습니다. 치수를 연결한 뒤 "
           "학습할 수 있습니다. PCA는 이후 학습 분할에만 적합합니다.\n"), encoding="utf-8")


if __name__ == "__main__":
    main()
