"""Compare inverse generators on fixed goals using a forward model and hinge stress."""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from inverse_cvae import (ANGLE_BARRIER_CHECKPOINT, ANGLE_BARRIER_TARGETS, BUDGET,
                          FEATURES, TARGETS, dataset, decode, gmm_sample, read)

TOL = np.array([3., .5, .05])
STRESS_CAP = 6.
SEEDS = (42, 43, 44)
GOAL_COLUMNS = ("angle_deg", "deformation_mm", "barrier_mJ")


def response_config(args):
    if args.angle_displacement:
        return TARGETS[:2], GOAL_COLUMNS[:2], TOL[:2], "cvae_2"
    if args.angle_barrier:
        return (ANGLE_BARRIER_TARGETS, (GOAL_COLUMNS[0], GOAL_COLUMNS[2]),
                TOL[[0, 2]], ANGLE_BARRIER_CHECKPOINT)
    return TARGETS, GOAL_COLUMNS, TOL, "cvae_3"


def method_names(args):
    current = f"cvae_{args.version}"
    if args.compare_barrier_conditioning:
        return args.baseline_label, current
    if args.cvae_only:
        return (current,)
    return args.baseline_label, current, f"gmm_{args.version}"


def fit_gmm(args):
    from scipy.special import logit
    from sklearn.mixture import GaussianMixture

    train, _ = dataset(args.targets)
    designs, responses = (np.array([r[i] for r in train if r[3]]) for i in (1, 2))
    with np.load(args.output / "cvae_3_42.npz") as saved:
        lo, hi, mu, sd = (saved[k] for k in ("lo", "hi", "mu", "sd"))
    joint = np.column_stack((logit(np.clip((designs - lo) / (hi - lo), .001, .999)),
                             (responses - mu) / sd))
    gmm = GaussianMixture(n_components=4, reg_covar=.01, n_init=5, random_state=42).fit(joint)
    mixture = (gmm.weights_, gmm.means_, gmm.covariances_)
    np.savez(args.output / "conditional_gmm.npz", weights=mixture[0], means=mixture[1],
             covariances=mixture[2], lo=lo, hi=hi, mu=mu, sd=sd)
    return mixture, lo, hi, mu, sd


def predict(args):
    import joblib
    from threadpoolctl import threadpool_limits

    goals = read(args.goals)
    response_names, goal_columns, tolerances, checkpoint = response_config(args)
    count = len(response_names)
    desired = np.array([[float(row[k]) for k in goal_columns] for row in goals])
    targets = np.repeat(desired, BUDGET, axis=0)
    model = lambda name: joblib.load(args.forward / f"{name}.joblib")
    clf = model("stability")
    regressors = [model(name) for name in (*response_names, "hinge_transition_max_MPa")]
    with threadpool_limits(limits=2):
        folders = (args.output,)
        if args.compare_barrier_conditioning:
            folders = (args.previous, args.output)
        elif not args.cvae_only:
            mixture, lo, hi, mu, sd = fit_gmm(args)
            folders = (args.previous, args.output, None)
        for method, folder in zip(method_names(args), folders):
            original_conditions = args.compare_barrier_conditioning and method == args.baseline_label
            generation_columns = GOAL_COLUMNS if original_conditions else goal_columns
            generation_targets = TARGETS if original_conditions else response_names
            model_checkpoint = "cvae_3" if original_conditions else checkpoint
            generation_goals = (np.array([[float(row[k]) for k in generation_columns] for row in goals])
                                if original_conditions else desired)
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                if folder:
                    with np.load(folder / f"{model_checkpoint}_{seed}.npz") as saved:
                        generator = dict(saved)
                    if "target_names" in generator and tuple(generator["target_names"]) != generation_targets:
                        raise ValueError("CVAE target order does not match the evaluation")
                    x = np.concatenate([decode(generator, goal, rng.normal(size=(BUDGET, 2)),
                                               len(generation_columns)) for goal in generation_goals])
                else:
                    x = np.concatenate([gmm_sample(mixture, (goal - mu) / sd, rng, lo, hi)
                                        for goal in desired])
                partial = np.column_stack((x, np.zeros(len(x))))
                full = partial.copy()
                full[:, -1] = 1
                ps = clf.predict_proba(partial)[:, 1]
                fs = clf.predict_proba(full)[:, 1]
                pred = np.column_stack([r.predict(partial) for r in regressors])
                hit = (ps > .5) & (fs <= .5) & (np.abs(pred[:, :count] - targets) <= tolerances).all(axis=1)
                idx = np.flatnonzero(hit)
                np.savez_compressed(args.output / f"{method}_{seed}_shortlist.npz", indices=idx,
                                    x=x[idx], pred=pred[idx], ps=ps[idx], fs=fs[idx])
                print(method, seed, len(idx), flush=True)


def write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def geometry(args):
    from geometry_check import check

    goals = read(args.goals)
    response_names, goal_columns, tolerances, _ = response_config(args)
    count = len(response_names)
    reports, rows, choices = [], [], []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for method in method_names(args):
            for seed in SEEDS:
                with np.load(args.output / f"{method}_{seed}_shortlist.npz") as data:
                    idx, x, pred, ps, fs = (data[k] for k in ("indices", "x", "pred", "ps", "fs"))
                valid = np.concatenate(list(pool.map(check, np.array_split(x, max(1, (len(x)+255)//256)))))
                valid = np.asarray(valid, dtype=bool)
                before = np.bincount(idx[valid] // BUDGET, minlength=len(goals))
                after = valid & (pred[:, count] <= STRESS_CAP)
                passed = np.bincount(idx[after] // BUDGET, minlength=len(goals))
                summary = dict(method=method, seed=seed, goals=len(goals), candidates=len(goals)*BUDGET,
                               pre_stress_candidates=int(before.sum()), pre_stress_goals=int((before>0).sum()),
                               pass_candidates=int(passed.sum()), pass_goals=int((passed>0).sum()))
                reports.append(summary)
                print(summary, flush=True)
                for i, goal in enumerate(goals):
                    rows.append(dict(method=method, seed=seed, goal_id=goal["goal_id"],
                                     pre_stress_count=before[i], pass_count=passed[i]))
                    selected = np.flatnonzero(after & (idx // BUDGET == i))
                    if selected.size:
                        desired = np.array([float(goal[k]) for k in goal_columns])
                        j = selected[np.argmin((np.abs(pred[selected, :count] - desired) / tolerances).sum(1))]
                        choices.append(dict(method=method, seed=seed, goal_id=goal["goal_id"],
                                            **dict(zip(FEATURES, x[j])),
                                            **{f"goal_{k}": v for k, v in zip(goal_columns, desired)},
                                            **{f"predicted_{k}": v for k, v in zip(response_names, pred[j, :count])},
                                            predicted_hinge_transition_max_MPa=pred[j,count],
                                            partial_bistable_score=ps[j], full_bistable_score=fs[j]))
    write(args.output / "goal_results.csv", rows)
    if choices:
        write(args.output / "forward_screened_candidates.csv", choices)
    (args.output / "metrics.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("predict", "geometry"))
    parser.add_argument("--version", default="v2")
    parser.add_argument("--baseline-label", default="previous_cvae")
    parser.add_argument("--cvae-only", action="store_true")
    response_options = parser.add_mutually_exclusive_group()
    response_options.add_argument("--angle-barrier", action="store_true")
    response_options.add_argument("--angle-displacement", action="store_true")
    parser.add_argument("--compare-barrier-conditioning", action="store_true")
    for name in ("targets", "goals", "forward", "previous", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.compare_barrier_conditioning and not args.angle_displacement:
        parser.error("--compare-barrier-conditioning requires --angle-displacement.")
    if args.compare_barrier_conditioning and args.baseline_label == f"cvae_{args.version}":
        parser.error("The model and baseline labels must differ.")
    if args.angle_barrier or args.angle_displacement:
        args.cvae_only = True
    (predict if args.stage == "predict" else geometry)(args)
