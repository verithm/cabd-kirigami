"""Join original and supplemental ANSYS result archives for scalar forward training."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from preprocess_energy_targets import prepare
from preprocess_forward import FEATURES, process, read_rows, write_rows
from preprocess_hinge_stress import extract
from train_forward import read_csv

STRESS = "initial_to_second_max_MPa"
STATES = ("partial", "full")


def dimensions(nodes):
    xy = {r["name"]: (float(r["x0_mm"]), float(r["z0_mm"])) for r in nodes}
    x, z = xy["A_upper"]
    half_base = xy["frame_v06"][1]
    inner_top = xy["frame_v05"][1]
    if not np.isclose(half_base, 12.7, atol=1e-8, rtol=0):
        raise ValueError("Unexpected base geometry")
    tip_length = inner_top - z
    tip_thickness = -xy["hinge_tip_upper_v03"][0]
    apex = [v for n, v in xy.items() if n.startswith("hinge_apex_v")]
    return (tip_thickness, tip_length, min(apex, key=lambda v: abs(v[1]))[0] - x,
            2 * max(v[1] for v in apex), -x, half_base - inner_top,
            x - xy["frame_v04"][0], 2 * (-xy["B_upper"][0] - tip_thickness) / tip_length)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Folder with original collection ZIPs")
    parser.add_argument("jobs", type=Path, help="Original jobs.csv")
    parser.add_argument("output", type=Path, help="New output folder")
    parser.add_argument("additional", type=Path, nargs="+", help="Supplemental result ZIPs")
    parser.add_argument("--job-kits", type=Path, nargs="+", default=[],
                        help="Collection kits with original jobs.csv for supplemental results")
    args = parser.parse_args()
    original_jobs = {r["geom_id"]: r for r in read_csv(args.jobs)}
    jobs = {g: {k: r[k] for k in ("geom_id", "role", *FEATURES)} | {"feature_source": "jobs_csv"}
            for g, r in original_jobs.items()}
    kit_jobs = {}
    for path in args.job_kits:
        with ZipFile(path) as archive:
            member, = [name for name in archive.namelist()
                       if name.replace(chr(92), "/").split("/")[-1] == "jobs.csv"]
            for row in read_rows(archive, member):
                geom = row["geom_id"]
                if geom in original_jobs or geom in kit_jobs or row["role"] != "train":
                    raise ValueError(f"Unexpected kit geometry: {geom}")
                kit_jobs[geom] = row
    with TemporaryDirectory() as directory:
        paths = [*sorted(args.source.glob("collect_result_PC*.zip")), *args.additional]
        source = Path(directory)
        cases, recovered = {}, {}
        verified = 0
        for i, path in enumerate(paths):
            (source / path.name).symlink_to(path.resolve())
            for row, _ in process(path, 0.01):
                geom = row["geom_id"]
                key = (geom, row["state"])
                if key in cases:
                    raise ValueError(f"Duplicate result: {key}")
                if i >= len(paths) - len(args.additional) and (geom in original_jobs or row["role"] != "train"):
                    raise ValueError(f"Supplemental result overlaps original IDs or test data: {key}")
                cases[key] = row
            with ZipFile(path) as archive:
                for name in archive.namelist():
                    parts = name.replace(chr(92), "/").split("/")
                    if len(parts) < 2 or parts[-1] not in {f"{s}_landmark_nodes.csv" for s in STATES}:
                        continue
                    geom = parts[-2]
                    features = dimensions(read_rows(archive, name))
                    listed = original_jobs.get(geom) or kit_jobs.get(geom)
                    if listed:
                        expected = [float(listed[k]) for k in FEATURES]
                        if not np.allclose(features, expected, atol=1e-8, rtol=0):
                            raise ValueError(f"Landmark recovery disagrees with original jobs.csv: {geom}")
                        verified += 1
                    if geom in recovered and not np.allclose(features, recovered[geom], atol=1e-8, rtol=0):
                        raise ValueError(f"Inconsistent partial/full geometry: {geom}")
                    recovered[geom] = features
        if not verified:
            raise ValueError("No original geometry coordinates for cross-checking")
        for geom, _ in sorted(cases):
            if geom not in jobs:
                if geom in kit_jobs:
                    listed = kit_jobs[geom]
                    jobs[geom] = {k: listed[k] for k in ("geom_id", "role", *FEATURES)}
                    jobs[geom]["feature_source"] = "jobs_csv"
                    continue
                x = recovered.get(geom, (None,) * len(FEATURES))
                jobs[geom] = dict(geom_id=geom, role="train", **dict(zip(FEATURES, x)),
                                  feature_source="landmark_coordinates" if geom in recovered else "missing")
        for (geom, _), row in cases.items():
            if geom in kit_jobs and row["pc"] != kit_jobs[geom]["pc"]:
                raise ValueError(f"Result PC differs from kit assignment: {geom}")
            row.update({k: jobs[geom][k] for k in (*FEATURES, "feature_source")})

        index = source / "case_index.csv"
        jobs_path = source / "jobs.csv"
        targets_path = source / "scalar_forward_targets.csv"
        write_rows(index, [cases[k] for k in sorted(cases)])
        write_rows(jobs_path, [jobs[g] for g in sorted(jobs)])
        targets = list(prepare(source, index))
        write_rows(targets_path, targets)
        stress = list(extract(source, jobs_path, index, targets_path))

        output = args.output
        output.mkdir(parents=True, exist_ok=False)
        for path in (index, jobs_path):
            (output / path.name).write_bytes(path.read_bytes())
        stress_lookup = {(r["geom_id"], r["state"]): r for r in stress}
        for row in stress:
            row["feature_source"] = jobs[row["geom_id"]]["feature_source"]
        write_rows(output / "hinge_stress.csv", stress)
        model_rows = []
        for row in targets:
            if not all(row[k] for k in FEATURES):
                continue
            row["feature_source"] = jobs[row["geom_id"]]["feature_source"]
            row["hinge_transition_max_MPa"] = stress_lookup[row["geom_id"], row["state"]][STRESS]
            model_rows.append(row)
        write_rows(output / "scalar_forward_targets.csv", model_rows)
        stats = defaultdict(Counter)
        for row in model_rows:
            group = ("original" if row["geom_id"] in original_jobs else "additional", row["role"], row["state"])
            counts = stats[group]
            counts[row["stability"]] += 1
            counts["regression_complete"] += all(row[k] is not None for k in
                                                ("stable_angle_deg", "reverse_barrier_mJ",
                                                 "deformation_applied_mm", "hinge_transition_max_MPa"))
        write_rows(output / "summary.csv", [
            dict(batch=batch, role=role, state=state, **{k: counts[k] for k in
                 ("bistable", "monostable", "unknown", "regression_complete")})
            for (batch, role, state), counts in sorted(stats.items())])
        print(f"Verified {verified} coordinate exports against jobs.csv. {len(jobs)} geometries, "
              f"{len(cases)} result records, {len(model_rows)} completed cases.")
        print(f"Missing geometry IDs: {[g for g, row in jobs.items() if row['feature_source'] == 'missing']}")


if __name__ == "__main__":
    main()
