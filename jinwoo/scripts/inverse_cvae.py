"""Train and compare conditional inverse design generators."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FEATURES = ("t_h", "L_h", "t_a", "L_a", "L", "t_f", "c", "rho")
TARGETS = ("stable_angle_deg", "deformation_applied_mm", "reverse_barrier_mJ")
ANGLE_BARRIER_TARGETS = (TARGETS[0], TARGETS[2])
ANGLE_BARRIER_CHECKPOINT = "cvae_angle_barrier"
TOL = np.array([4.0, 0.5, 0.05])
SEEDS = (42, 43, 44)
BUDGET = 512


def transform_barrier(values, targets, transform="none", scale=0.05):
    result = np.array(values, dtype=float, copy=True)
    if transform == "none":
        return result
    index = tuple(targets).index("reverse_barrier_mJ")
    barrier = result[..., index]
    if np.any(barrier < 0) or (transform == "log" and np.any(barrier == 0)):
        raise ValueError("The selected barrier transform requires nonnegative or positive values")
    result[..., index] = np.log(barrier) if transform == "log" else np.log1p(barrier / scale)
    return result


def read(path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def dataset(path, targets=TARGETS):
    pairs = defaultdict(dict)
    for row in read(path):
        pairs[row["role"], row["geom_id"]][row["state"]] = row
    train = []
    test = []
    for (role, geom), pair in sorted(pairs.items()):
        p, f = pair.get("partial", {}), pair.get("full", {})
        if (p.get("stability") != "bistable" or f.get("stability") not in ("bistable", "monostable")
                or not all(p.get(key) for key in targets)):
            continue
        switch = int(f["stability"] == "monostable")
        item = (geom, [float(p[k]) for k in FEATURES], [float(p[k]) for k in targets], switch)
        if role == "train":
            train.append(item)
        elif switch:
            test.append(item)
    return train, test


def fit(args):
    import torch
    from torch import nn

    torch.set_num_threads(2)
    angle_barrier = getattr(args, "angle_barrier", False)
    targets = ANGLE_BARRIER_TARGETS if angle_barrier else TARGETS
    train, _ = dataset(args.targets, targets)
    x = np.array([r[1] for r in train])
    y = np.array([r[2] for r in train])
    barrier_transform = getattr(args, "barrier_transform", "none")
    barrier_scale = getattr(args, "barrier_scale", 0.05)
    y = transform_barrier(y, targets, barrier_transform, barrier_scale)
    switch = np.array([r[3] for r in train])
    tr_jobs = [r for r in read(args.jobs) if r["role"] == "train"]
    all_x = np.array([[float(r[k]) for k in FEATURES] for r in tr_jobs])
    lo, hi = all_x.min(0), all_x.max(0)
    mu, sd = y.mean(0), y.std(0)
    args.output.mkdir(parents=True, exist_ok=False)
    tx = torch.tensor((x - lo) / (hi - lo), dtype=torch.float32)
    fraction = getattr(args, "learning_fraction", None)
    condition_counts = getattr(args, "condition_counts", None) or ((3,) if fraction is not None else (3, 2, 1))
    if angle_barrier:
        condition_counts = (len(targets),)
    for k in condition_counts:
        for seed in SEEDS:
            if fraction is None:
                subset = np.arange(len(x))
            else:
                rng = np.random.default_rng(seed)
                groups = [rng.permutation(np.flatnonzero(switch == label)) for label in (0, 1)]
                subset = np.sort(np.concatenate([group[:round(len(group) * fraction)] for group in groups]))
            local_mu = mu if fraction is None else y[subset].mean(0)
            local_sd = sd if fraction is None else y[subset].std(0)
            condition = np.column_stack(((y[subset, :k] - local_mu[:k]) / local_sd[:k], switch[subset]))
            tc = torch.tensor(condition, dtype=torch.float32)
            tx_subset = tx[subset]
            torch.manual_seed(seed)
            enc = nn.Sequential(nn.Linear(9 + k, 32), nn.Tanh(), nn.Linear(32, 32),
                                nn.Tanh(), nn.Linear(32, 4))
            dec = nn.Sequential(nn.Linear(3 + k, 32), nn.Tanh(), nn.Linear(32, 32),
                                nn.Tanh(), nn.Linear(32, 8), nn.Sigmoid())
            optimizer = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=0.002)
            steps = 0
            steps_per_full_epoch = (len(tx) + 31) // 32
            step_limit = 300 * steps_per_full_epoch
            while steps < step_limit:
                for ids in torch.randperm(len(tx_subset)).split(32):
                    xb, c = tx_subset[ids], tc[ids]
                    mean, logvar = enc(torch.cat((xb, c), dim=1)).chunk(2, dim=1)
                    logvar = logvar.clamp(-10, 5)
                    z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
                    reconstruction = dec(torch.cat((z, c), dim=1))
                    mse = (reconstruction - xb).square().mean()
                    kl = (logvar.exp() + mean.square() - 1 - logvar).mean() / 2
                    loss = mse + 0.005 * min(1, (steps // steps_per_full_epoch + 1) / 60) * kl
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    steps += 1
                    if steps == step_limit:
                        break
            weights = {f"w{i}": dec[j].weight.detach().numpy() for i, j in enumerate((0, 2, 4))}
            biases = {f"b{i}": dec[j].bias.detach().numpy() for i, j in enumerate((0, 2, 4))}
            checkpoint = ANGLE_BARRIER_CHECKPOINT if angle_barrier else f"cvae_{k}"
            np.savez(args.output / f"{checkpoint}_{seed}.npz", lo=lo, hi=hi, mu=local_mu, sd=local_sd,
                     target_names=targets[:k], barrier_transform=barrier_transform,
                     barrier_scale_mJ=barrier_scale, **weights, **biases)
            print(f"trained k={k}, seed={seed}, n={len(subset)}, positive={switch[subset].sum()}", flush=True)


def decode(model, target, z, k):
    target = np.array(target[:k], dtype=float)
    transform = str(model.get("barrier_transform", "none"))
    if transform != "none":
        target = transform_barrier(target, tuple(model["target_names"]), transform,
                                   float(model["barrier_scale_mJ"]))
    condition = np.r_[(target - model["mu"][:k]) / model["sd"][:k], 1]
    a = np.column_stack((z, np.broadcast_to(condition, (len(z), k + 1))))
    for i in (0, 1):
        a = np.tanh(a @ model[f"w{i}"].T + model[f"b{i}"])
    a = 1 / (1 + np.exp(-np.clip(a @ model["w2"].T + model["b2"], -60, 60)))
    return model["lo"] + a * (model["hi"] - model["lo"])



def gmm_sample(mixture, target, generator, lo, hi):
    from scipy.special import expit, softmax
    from scipy.stats import multivariate_normal

    weights, means, covs = mixture
    conditional_means = []
    conditional_covs = []
    log_weights = []
    for weight, mean, cov in zip(weights, means, covs):
        cross = np.linalg.solve(cov[8:, 8:], cov[8:, :8]).T
        conditional_means.append(mean[:8] + cross @ (target - mean[8:]))
        conditional_covs.append(cov[:8, :8] - cross @ cov[8:, :8])
        log_weights.append(np.log(weight) + multivariate_normal.logpdf(target, mean[8:], cov[8:, 8:]))
    components = generator.choice(len(weights), size=BUDGET, p=softmax(log_weights))
    values = np.empty((BUDGET, 8))
    for j in range(len(weights)):
        mask = components == j
        values[mask] = generator.multivariate_normal(conditional_means[j], conditional_covs[j], size=mask.sum())
    return lo + expit(values) * (hi - lo)


def benchmark(args):
    import joblib
    from scipy.special import logit
    from scipy.stats import qmc
    from sklearn.mixture import GaussianMixture

    train, test = dataset(args.targets)
    designs = np.array([r[1] for r in train if r[3]])
    stability = joblib.load(args.forward / "stability.joblib")
    regression = [joblib.load(args.forward / f"{key}.joblib") for key in TARGETS]
    with np.load(args.output / "cvae_3_42.npz") as data:
        lo, hi, mu, sd = (data[name] for name in ("lo", "hi", "mu", "sd"))
    targets = np.array([r[2] for r in train if r[3]])
    joint = np.column_stack((logit(np.clip((designs - lo) / (hi - lo), 0.001, 0.999)),
                             (targets - mu) / sd))
    gmm = GaussianMixture(n_components=4, reg_covar=0.01, n_init=5, random_state=42).fit(joint)
    mixture = (gmm.weights_, gmm.means_, gmm.covariances_)
    np.savez(args.output / "conditional_gmm.npz", weights=mixture[0], means=mixture[1],
             covariances=mixture[2], lo=lo, hi=hi, mu=mu, sd=sd)
    details = []
    metrics = []
    for method in ("CVAE-3", "CVAE-2", "CVAE-1", "Conditional GMM", "Sobol", "Known-TR"):
        for seed in (SEEDS if method != "Known-TR" else SEEDS[:1]):
            generator = np.random.default_rng(seed)
            with np.load(args.output / f"cvae_{method[-1] if method.startswith('CVAE') else 3}_{seed}.npz") as file:
                model = dict(file)
            totals = defaultdict(list)
            for geom, _, target, _ in test:
                target = np.array(target)
                if method.startswith("CVAE"):
                    x = decode(model, target, generator.normal(size=(BUDGET, 2)), int(method[-1]))
                elif method == "Conditional GMM":
                    x = gmm_sample(mixture, (target - mu) / sd, generator, lo, hi)
                elif method == "Sobol":
                    points = qmc.Sobol(d=8, seed=int(generator.integers(2**31))).random_base2(9)
                    x = qmc.scale(points, model["lo"], model["hi"])
                else:
                    x = designs
                x = x[np.all(np.isfinite(x) & (x >= model["lo"]) & (x <= model["hi"]), axis=1)]
                p = np.column_stack((x, np.zeros(len(x))))
                f = np.column_stack((x, np.ones(len(x))))
                probs = stability.predict_proba(np.concatenate((p, f)))[:, 1]
                p_bi, f_bi = np.split(probs, 2)
                good = (p_bi > 0.5) & (f_bi <= 0.5)
                totals["pass_fraction"].append(float(np.mean(good)))
                x, p = x[good], p[good]
                if not len(x):
                    for metric in ("top1_all", "top5_all", "top5_angle", "any_all"):
                        totals[metric].append(False)
                    continue
                responses = np.column_stack([m.predict(p) for m in regression])
                error = np.abs(responses - target)
                normalized = error / TOL
                matches = normalized <= 1
                all_match = matches.all(axis=1)
                # Rank complete matches first, then near matches using fixed normalized errors.
                rank = np.lexsort((normalized.sum(axis=1), ~all_match))
                top = []
                for i in rank:
                    xi = (x[i] - model["lo"]) / (model["hi"] - model["lo"])
                    if all(np.linalg.norm(xi - previous) >= 0.05 for previous in top):
                        top.append(xi)
                        details.append(dict(method=method, seed=seed, target_id=geom, rank=len(top),
                                            **dict(zip(FEATURES, x[i])),
                                            **{f"goal_{name}": v for name, v in zip(TARGETS, target)},
                                            **{f"predicted_{name}": v for name, v in zip(TARGETS, responses[i])},
                                            partial_bistable_score=p_bi[good][i],
                                            full_bistable_score=f_bi[good][i],
                                            predicted_all_within=int(all_match[i])))
                        if len(top) == 1:
                            totals["top1_all"].append(bool(all_match[i]))
                            top_matches = []
                        top_matches.append(matches[i])
                        if len(top) == 5:
                            break
                totals["top5_all"].append(bool(np.any(np.all(top_matches, axis=1))))
                totals["top5_angle"].append(bool(np.any(np.array(top_matches)[:, 0])))
                totals["any_all"].append(bool(all_match.any()))
            summary = dict(method=method, seed=seed, goals=len(test),
                           **{key: float(np.mean(value)) for key, value in totals.items()})
            metrics.append(summary)
            print(summary, flush=True)
    with (args.output / "top_candidates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    (args.output / "metrics.json").write_text(json.dumps(dict(
        observed_training_shapes=len(train), target_training_shapes=sum(r[3] for r in train),
        test_goal_count=len(test), forward_budget_per_goal=BUDGET,
        tolerances=dict(zip(TARGETS, TOL.tolist())), results=metrics), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "benchmark"))
    parser.add_argument("--condition-counts", nargs="+", type=int, choices=(1, 2, 3))
    parser.add_argument("--angle-barrier", action="store_true")
    parser.add_argument("--barrier-transform", choices=("none", "log", "log1p"), default="none")
    parser.add_argument("--barrier-scale", type=float, default=0.05)
    for name in ("targets", "jobs", "forward", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.barrier_scale <= 0:
        parser.error("--barrier-scale must be positive.")
    if args.barrier_transform != "none" and (
            args.stage != "fit" or (not args.angle_barrier and args.condition_counts != [3])):
        parser.error("A barrier transform requires fit with --condition-counts 3 or --angle-barrier.")
    if args.angle_barrier and (args.stage != "fit" or args.condition_counts is not None):
        parser.error("--angle-barrier is a standalone fit configuration.")
    if args.stage == "fit":
        fit(args)
    else:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=2):
            benchmark(args)
