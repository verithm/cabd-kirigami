"""Extract four scalar forward targets from completed ANSYS runs."""
import argparse
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from preprocess_forward import FEATURES, arm_angles, read_rows, write_rows
from csv_io import read_csv


def first_energy_well(energy):
    """Return the first interior peak and the first interior valley after it."""
    changes = np.diff(energy)
    steps = np.flatnonzero(changes)
    signs = np.sign(changes[steps])
    peak = None
    for i in np.flatnonzero(signs[:-1] != signs[1:]):
        # Use the center sample if a local extremum spans a plateau.
        point = int((steps[i] + 1 + steps[i + 1]) // 2)
        if signs[i] > 0 and peak is None:
            peak = point
        elif signs[i] < 0 and peak is not None:
            return "bistable", (peak, point)
    return ("monostable" if len(signs) and np.all(signs > 0) else "unknown"), None


def prepare(source, index):
    with ExitStack() as stack:
        archives = {}
        for case in read_csv(index):
            if case["status"] != "ok" or case["force_complete"] != "1":
                continue
            name = case["archive"]
            if name not in archives:
                archive = stack.enter_context(ZipFile(source[name] if isinstance(source, dict) else source / name))
                archives[name] = archive, {n.replace(chr(92), "/"): n for n in archive.namelist()}
            archive, members = archives[name]
            member = case["curve_file"]
            samples = read_rows(archive, members[member])
            energy = np.array([float(r["U_total_mJ"]) for r in samples])
            u = np.array([float(r["u_applied_mm"]) for r in samples])
            sets = [int(r["set"]) for r in samples]
            if not np.isfinite(energy).all():
                label, extrema = "unknown", None
            else:
                label, extrema = first_energy_well(energy)
            result = {c: case[c] for c in ("geom_id", "state", "role", *FEATURES)}
            result.update(stability=label, stable_angle_deg=None, reverse_barrier_mJ=None,
                          deformation_applied_mm=None, peak_set=None, valley_set=None,
                          peak_energy_mJ=None, valley_energy_mJ=None,
                          stable_angle_lower_deg=None, status=case["status"],
                          mesh_rule=case["mesh_rule"], archive=name, curve_file=member)
            if extrema is not None:
                peak, valley = extrema
                step = sets[valley]
                angles = arm_angles(archive, members, member.removesuffix("_curve.csv"), step, step, 0)
                result.update(reverse_barrier_mJ=float(energy[peak] - energy[valley]),
                              deformation_applied_mm=float(u[valley]), peak_set=sets[peak],
                              valley_set=step, peak_energy_mJ=float(energy[peak]),
                              valley_energy_mJ=float(energy[valley]))
                if angles is not None:
                    result["stable_angle_deg"], result["stable_angle_lower_deg"] = angles
            yield result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Folder with collection ZIP files")
    parser.add_argument("index", type=Path, help="Joined case_index.csv")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = list(prepare(args.source, args.index))
    if not rows or any(not row[k] for row in rows for k in FEATURES):
        parser.error("Completed cases with joined geometry inputs are required.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output, rows)
    print(Counter((r["role"], r["state"], r["stability"]) for r in rows))
    print("Rows with all three regression targets:", sum(r["stable_angle_deg"] is not None for r in rows))
    pairs = {(r["geom_id"], r["state"]): r for r in rows}
    print("Required pairs by role:", Counter(
        r["role"] for r in rows if r["state"] == "partial" and r["stable_angle_deg"] is not None
        and pairs.get((r["geom_id"], "full"), {}).get("stability") == "monostable"
    ))


if __name__ == "__main__":
    main()
