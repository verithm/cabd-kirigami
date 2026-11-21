"""Make three ANSYS-backed target sets for the fixed inverse design benchmark."""
import argparse
from pathlib import Path

import numpy as np

from evaluate_inverse_v2 import STRESS_CAP, TOL, write
from inverse_cvae import dataset, read

GOAL_COLUMNS = ("angle_deg", "deformation_mm", "barrier_mJ")


def goal_row(number, goal, train_ids, known):
    distances = np.abs((known - goal) / TOL).max(axis=1)
    nearest = distances.argmin()
    return dict(goal_id=number, **dict(zip(GOAL_COLUMNS, goal)),
                nearest_train_geometry=train_ids[nearest],
                nearest_train_distance=distances[nearest],
                train_geometries_within_tolerance=int(np.count_nonzero(distances <= 1)))


def heldout(path, test, train_ids, known):
    stress = {r["geom_id"]: float(r["hinge_transition_max_MPa"])
              for r in read(path) if r["role"] == "test" and r["state"] == "partial"
              and r["hinge_transition_max_MPa"]}
    return [dict(goal_row(i, goal, train_ids, known), source_heldout_geometry=geom,
                 actual_hinge_stress_MPa=stress.get(geom),
                 actual_hinge_stress_le_6_mpa=stress[geom] <= STRESS_CAP if geom in stress else "")
            for i, (geom, _, goal, _) in enumerate(test, 1)]


def interpolate(train_ids, known, edges, rng):
    distance = np.linalg.norm((known[:, None, :] - known) / TOL, axis=2)
    np.fill_diagonal(distance, np.inf)
    neighbors = np.argsort(distance, axis=1, kind="stable")[:, :5]
    rows = []
    for bin_no, (low, high) in enumerate(zip(edges, edges[1:]), 1):
        anchors = np.flatnonzero((known[:, 0] >= low) & (known[:, 0] <= high))
        count = 0
        for _ in range(5000):
            left = rng.choice(anchors)
            right = rng.choice(neighbors[left])
            weight = rng.uniform(.15, .85)
            goal = weight * known[left] + (1 - weight) * known[right]
            if not low <= goal[0] <= high:
                continue
            row = goal_row(len(rows) + 1, goal, train_ids, known)
            if row["train_geometries_within_tolerance"] == 0:
                continue
            rows.append(dict(row, angle_bin=bin_no, anchor_a=train_ids[left],
                             anchor_b=train_ids[right], anchor_a_weight=weight))
            count += 1
            if count == 10:
                break
        if count != 10:
            raise RuntimeError(f"Only {count} supported interpolation goals in angle bin {bin_no}")
    return rows


def angle_first(train_ids, known, edges, rng):
    rows = []
    for bin_no, (low, high) in enumerate(zip(edges, edges[1:]), 1):
        subedges = np.linspace(low, high, 11)
        for start, end in zip(subedges, subedges[1:]):
            angle = rng.uniform(start, end)
            angle_refs = known[np.abs(known[:, 0] - angle) <= TOL[0]]
            if not len(angle_refs):
                raise RuntimeError(f"No observed neighbors for angle {angle}")
            displacement_band = np.quantile(angle_refs[:, 1], [.1, .9])
            for _ in range(5000):
                displacement = rng.uniform(*displacement_band)
                local = angle_refs[np.abs(angle_refs[:, 1] - displacement) <= TOL[1]]
                if not len(local):
                    continue
                barrier = rng.uniform(*np.quantile(local[:, 2], [.1, .9]))
                row = goal_row(len(rows) + 1, np.array([angle, displacement, barrier]),
                               train_ids, known)
                if row["train_geometries_within_tolerance"]:
                    rows.append(dict(row, angle_bin=bin_no, angle_neighbor_count=len(angle_refs),
                                     angle_displacement_neighbor_count=len(local)))
                    break
            else:
                raise RuntimeError(f"No supported angle-first goal for angle {angle}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("heldout", "interpolation", "angle-first"))
    parser.add_argument("targets", type=Path)
    parser.add_argument("goals", type=Path)
    args = parser.parse_args()
    train, test = dataset(args.targets)
    train_ids = [r[0] for r in train if r[3]]
    known = np.array([r[2] for r in train if r[3]])
    if set(train_ids) & {r[0] for r in test}:
        raise ValueError("A held-out geometry appears in the training references")
    if args.method == "heldout":
        rows = heldout(args.targets, test, train_ids, known)
    else:
        generator = interpolate if args.method == "interpolation" else angle_first
        edges = np.linspace(known[:, 0].min(), known[:, 0].max(), 11)
        rows = generator(train_ids, known, edges, np.random.default_rng(42))
    write(args.goals, rows)
    print(args.method, len(rows), "goals", flush=True)


if __name__ == "__main__":
    main()
