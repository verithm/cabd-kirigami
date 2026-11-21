"""Conditional VAE that generates geometries from a target (decided 2026-11-16).

Training data: the partial-bi geometries of the training set (408 of the 512 base ones).
  x = 8 design variables on [-1, 1]
  condition = (theta2, u2, log10 dE_r) standardized on the training data + full_mono flag (1 = full is mono)
Model: encoder q(z | x, c), decoder p(x | z, c) with a learned diagonal Gaussian; ELBO with weight beta on the KL.
Tuning: 20 random configurations scored by the validation negative ELBO on an inner 20 % hold-out.
Inverse use (no forward model): generate N_GEN geometries for (target, full_mono = 1), drop those outside the
  bounds or failing the geometry constraints (the caller's `feasible`), cluster the rest in the [0, 1] design space
  and return the medoids of the largest clusters (largest first), at least MIN_DIST apart.

Saved format (cabd.models.io): <name>.pt = decoder + encoder state_dict,
<name>.json = {"kind": "cvae", "cfg", "mean", "sd", "epochs", "score", "seed"}.
"""
import numpy as np
import torch

from .. import data as D
from ..config import Y_COLS

TRIALS, MAX_EPOCHS, PATIENCE = 20, 4000, 300
SPACE = {"width": [64, 128, 256], "depth": [2, 3], "latent": [2, 3, 4], "beta": [0.5, 1.0, 2.0],
         "lr": [3e-4, 1e-3, 3e-3], "wd": [0.0, 1e-5, 1e-4]}
N_GEN, N_CAND, MIN_DIST = 200, 5, 0.1


def mlp(n_in, n_out, cfg):
    layers, n = [], n_in
    for _ in range(cfg["depth"]):
        layers += [torch.nn.Linear(n, cfg["width"]), torch.nn.SiLU()]
        n = cfg["width"]
    return torch.nn.Sequential(*layers, torch.nn.Linear(n, n_out))


class CVAE(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg, z = cfg, cfg["latent"]
        self.enc = mlp(8 + 4, 2 * z, cfg)
        self.dec = mlp(z + 4, 16, cfg)

    def decode(self, z, c):
        o = self.dec(torch.cat([z, c], 1))
        return o[:, :8], o[:, 8:].clamp(-7, 2)

    def loss(self, x, c):
        h = self.enc(torch.cat([x, c], 1))
        mu, logv = h[:, : self.cfg["latent"]], h[:, self.cfg["latent"]:].clamp(-8, 4)
        z = mu + torch.randn_like(mu) * (0.5 * logv).exp()
        xm, xlv = self.decode(z, c)
        rec = 0.5 * (xlv + (x - xm) ** 2 / xlv.exp()).sum(1)
        kl = 0.5 * (mu ** 2 + logv.exp() - 1 - logv).sum(1)
        return (rec + self.cfg["beta"] * kl).mean()


def _train(X, Cn, cfg, seed, Xv=None, Cv=None, epochs=None):
    torch.manual_seed(seed)
    m = CVAE(cfg)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    X, Cn = torch.tensor(X, dtype=torch.float32), torch.tensor(Cn, dtype=torch.float32)
    if Xv is not None:
        Xv, Cv = torch.tensor(Xv, dtype=torch.float32), torch.tensor(Cv, dtype=torch.float32)
    best, best_ep, state, wait = np.inf, 0, None, 0
    for ep in range(1, (epochs or MAX_EPOCHS) + 1):
        opt.zero_grad()
        m.loss(X, Cn).backward()
        opt.step()
        if Xv is not None and ep % 10 == 0:
            with torch.no_grad():
                torch.manual_seed(0)
                v = np.mean([m.loss(Xv, Cv).item() for _ in range(5)])
            if v < best:
                best, best_ep, state, wait = v, ep, {k: t.clone() for k, t in m.state_dict().items()}, 0
            else:
                wait += 10
                if wait >= PATIENCE:
                    break
    if state is not None:
        m.load_state_dict(state)
    return m, best_ep, best


class Generator:
    """cVAE fitted on a set of partial-bi geometries."""
    kind = "cvae"

    def fit(self, d_bi, seed):
        X = (D.scale_x(d_bi) * 2 - 1).astype(np.float32)
        Y = d_bi[Y_COLS].to_numpy(float)
        self.mean, self.sd = Y.mean(0), Y.std(0)
        mono = d_bi.full_class.map(D.state_label).eq("mono").to_numpy(float)
        Cn = np.column_stack([(Y - self.mean) / self.sd, mono]).astype(np.float32)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X))
        va, tr = perm[: len(X) // 5], perm[len(X) // 5:]
        trials = []
        for t in range(TRIALS):
            cfg = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
            _, ep, score = _train(X[tr], Cn[tr], cfg, seed * 1000 + t, X[va], Cn[va])
            trials.append((score, ep, cfg))
        self.score, ep, self.cfg = min(trials, key=lambda r: r[0])
        self.epochs = ep
        self.model, _, _ = _train(X, Cn, self.cfg, seed * 1000 + 999, epochs=ep)
        self.seed = seed
        return self

    def sample(self, target, n=N_GEN, seed=0):
        """target = (theta2, u2, logE). Returns (n, 8) designs in [0, 1] units (unclipped)."""
        c = np.r_[(np.asarray(target, float) - self.mean) / self.sd, 1.0]
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            z = torch.randn(n, self.cfg["latent"], generator=g)
            xm, _ = self.model.decode(z, torch.tensor(np.tile(c, (n, 1)), dtype=torch.float32))
        return (xm.numpy() + 1) / 2

    def candidates(self, target, feasible, seed=0):
        """Medoids of the largest clusters among feasible samples, largest first, at least MIN_DIST apart.

        feasible(u) -> bool for one design u in [0, 1] units (bounds and geometry constraints)."""
        from scipy.cluster.hierarchy import fcluster, linkage
        S = self.sample(target, seed=seed)
        S = S[[feasible(s) for s in S]]
        if len(S) == 0:
            return np.empty((0, 8))
        if len(S) == 1:
            return S
        lab = fcluster(linkage(S, "average"), MIN_DIST, "distance")
        groups = sorted((S[lab == k] for k in np.unique(lab)), key=len, reverse=True)
        out = []
        for gpts in groups:
            med = gpts[np.argmin(((gpts[:, None] - gpts[None]) ** 2).sum(-1).sum(1))]
            if all(np.linalg.norm(med - o) >= MIN_DIST for o in out):
                out.append(med)
            if len(out) == N_CAND:
                break
        return np.array(out)


def fit(d, seed):
    """cVAE on the partial-bi rows of d (single-threaded torch)."""
    torch.set_num_threads(1)
    return Generator().fit(d[d.is_reg].reset_index(drop=True), seed)
