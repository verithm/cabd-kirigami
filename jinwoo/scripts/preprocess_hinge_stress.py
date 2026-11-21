"""Extract recorded hinge von Mises maxima from the original collection ZIPs."""

import argparse
import csv
import math
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

from preprocess_forward import FEATURES, read_rows, write_rows
from train_forward import read_csv

HINGES = ("hinge_tip_upper", "hinge_tip_lower", "hinge_apex")
STATES = ("partial", "full")
WINDOWS = ("recorded_path", "initial_to_peak", "peak_to_second", "initial_to_second")


def key(row):
    return row["geom_id"], row["state"]


def maxima(prefix, samples):
    fields = {f"{prefix}_{name}_max_MPa": None for name in HINGES}
    fields.update({f"{prefix}_{field}": None for field in
                   ("max_MPa", "critical_hinge", "critical_set", "critical_u_mm", "n_sets")})
    if samples:
        for i, name in enumerate(HINGES):
            fields[f"{prefix}_{name}_max_MPa"] = max(s[2][i] for s in samples)
        step, u, stresses = max(samples, key=lambda s: max(s[2]))
        i = max(range(len(HINGES)), key=stresses.__getitem__)
        fields.update({f"{prefix}_max_MPa": stresses[i], f"{prefix}_critical_hinge": HINGES[i],
                       f"{prefix}_critical_set": step, f"{prefix}_critical_u_mm": u,
                       f"{prefix}_n_sets": len(samples)})
    return fields


def extract(source, jobs, index, scalar_targets):
    cases = {key(r): r for r in read_csv(index)}
    targets = {key(r): r for r in read_csv(scalar_targets)}
    with ExitStack() as stack:
        archives = {}
        for job in read_csv(jobs):
            geom = job["geom_id"]
            p, f = (targets.get((geom, state), {}).get("stability") for state in STATES)
            pattern = int((p, f) == ("bistable", "monostable")) if {p, f} <= {"bistable", "monostable"} else None
            for state in STATES:
                case = cases.get((geom, state), {})
                target = targets.get((geom, state), {})
                label = target.get("stability", "unknown")
                row = {name: job[name] for name in ("geom_id", "role", *FEATURES)}
                row.update(state=state, status=case.get("status", "missing"),
                           mesh_rule=case.get("mesh_rule"), complete_15mm=int(case.get("force_complete") == "1"),
                           energy_stability=label, confirmed_bi_mono=pattern,
                           stable_angle_deg=target.get("stable_angle_deg"),
                           deformation_applied_mm=target.get("deformation_applied_mm"),
                           reverse_barrier_mJ=target.get("reverse_barrier_mJ"),
                           archive=case.get("archive"), curve_file=case.get("curve_file"),
                           stress_data_status="missing_curve", peak_set=None, valley_set=None,
                           peak_u_mm=None, valley_u_mm=None)
                windows = {name: [] for name in WINDOWS}
                if case.get("curve_quality") == "valid":
                    name = case["archive"]
                    if name not in archives:
                        archive = stack.enter_context(ZipFile(source / name))
                        archives[name] = archive, {n.replace(chr(92), "/"): n for n in archive.namelist()}
                    archive, members = archives[name]
                    curve = read_rows(archive, members[case["curve_file"]])
                    try:
                        recorded = [(int(r["set"]), float(r["u_applied_mm"]),
                                     tuple(float(r[f"seqv_max_{h}_MPa"]) for h in HINGES))
                                    for r in curve]
                        if not recorded or any(not math.isfinite(v) or v < 0
                                               for _, _, stresses in recorded for v in stresses):
                            raise ValueError("Invalid hinge stress")
                    except (KeyError, ValueError, TypeError):
                        row["stress_data_status"] = "invalid_stress"
                    else:
                        windows["recorded_path"] = recorded
                        row["stress_data_status"] = "recorded_path_only"
                        if label == "bistable" and row["complete_15mm"]:
                            steps = {sample[0]: i for i, sample in enumerate(recorded)}
                            peak, valley = (int(target[k]) for k in ("peak_set", "valley_set"))
                            a, b = steps[peak], steps[valley]
                            if not a < b:
                                raise ValueError(f"Unexpected peak/valley order: {geom} {state}")
                            windows.update(initial_to_peak=recorded[:a+1],
                                           peak_to_second=recorded[a:b+1],
                                           initial_to_second=recorded[:b+1])
                            row.update(stress_data_status="transition_available", peak_set=peak, valley_set=valley,
                                       peak_u_mm=recorded[a][1], valley_u_mm=recorded[b][1])
                for window, samples in windows.items():
                    row.update(maxima(window, samples))
                yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "jobs", "index", "scalar_targets", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    rows = list(extract(args.source, args.jobs, args.index, args.scalar_targets))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output, rows)
    print(Counter((r["state"], r["stress_data_status"]) for r in rows))


if __name__ == "__main__":
    main()
