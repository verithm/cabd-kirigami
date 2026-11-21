"""MLP ensemble forward model.

  regression: networks output mean and log-variance of (theta2, u2, log10 dE_r) standardized on the training rows,
      Gaussian NLL loss
  classification: networks output 3-class logits (bi>bi, bi>mono, mono>mono), cross-entropy
  ensemble: 5 networks that differ only in initialization; mixture mean and variance
  tuning: 20 random configurations, scored on an inner 20 % hold-out of the training rows (single network per trial);
      the chosen configuration's epoch count is reused for the final ensemble.
  intervals: normalized split-conformal; the scores come from the chosen configuration trained on the inner 80 % and
      evaluated on the inner 20 % hold-out, so an outer test fold is never used.
Inputs: design variables on [-1, 1] (float32). Training is single-threaded (torch.set_num_threads(1)).

Saved format (cabd.models.io): <name>.pt = {"reg": [state_dict] * 5, "cls": [state_dict] * 5},
<name>.json = {"kind": "mlp", "cfg_r", "cfg_c", "mean", "sd", "q"}; the networks are rebuilt by net(n_out, cfg).
"""
import numpy as np

from .. import data as D
from ..config import Y_COLS
from ..metrics import conformal_q

TRIALS, MEMBERS, MAX_EPOCHS, PATIENCE = 20, 5, 3000, 200
SPACE = {"width": [32, 64, 128, 256], "depth": [2, 3, 4], "lr": [3e-4, 1e-3, 3e-3],
         "wd": [0.0, 1e-5, 1e-4, 1e-3], "act": ["silu", "tanh"]}


def net(n_out, cfg):
    import torch
    act = {"silu": torch.nn.SiLU, "tanh": torch.nn.Tanh}[cfg["act"]]
    layers, n = [], 8
    for _ in range(cfg["depth"]):
        layers += [torch.nn.Linear(n, cfg["width"]), act()]
        n = cfg["width"]
    return torch.nn.Sequential(*layers, torch.nn.Linear(n, n_out))


def loss_fn(kind, out, y):
    import torch
    if kind == "reg":
        mu, logv = out[:, :3], out[:, 3:].clamp(-10, 6)
        return (0.5 * (logv + (y - mu) ** 2 / logv.exp())).mean()
    return torch.nn.functional.cross_entropy(out, y)


def train(kind, X, y, cfg, seed, Xv=None, yv=None, epochs=None):
    """Full-batch Adam. With a validation set: early stopping, returns (model, best epoch, best val loss)."""
    import torch
    torch.manual_seed(seed)
    m = net(6 if kind == "reg" else 3, cfg)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
    if Xv is not None:
        Xv, yv = torch.tensor(Xv, dtype=torch.float32), torch.tensor(yv)
    best, best_ep, best_state, wait = np.inf, 0, None, 0
    for ep in range(1, (epochs or MAX_EPOCHS) + 1):
        opt.zero_grad()
        loss_fn(kind, m(X), y).backward()
        opt.step()
        if Xv is not None and ep % 10 == 0:
            with torch.no_grad():
                v = loss_fn(kind, m(Xv), yv).item()
            if v < best:
                best, best_ep, best_state, wait = v, ep, {k: t.clone() for k, t in m.state_dict().items()}, 0
            else:
                wait += 10
                if wait >= PATIENCE:
                    break
    if best_state is not None:
        m.load_state_dict(best_state)
    return m, best_ep, best


def ensemble(kind, nets, X):
    """Ensemble prediction. reg -> (mixture mean, mixture sd) in standardized units; cls -> mean probabilities."""
    import torch
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        outs = [m(X).numpy() for m in nets]
    if kind == "reg":
        mu = np.stack([o[:, :3] for o in outs])
        var = np.stack([np.exp(np.clip(o[:, 3:], -10, 6)) for o in outs])
        mean = mu.mean(0)
        return mean, np.sqrt((var + mu ** 2).mean(0) - mean ** 2)
    p = np.stack([torch.softmax(torch.tensor(o), 1).numpy() for o in outs])
    return p.mean(0)


def fit_kind(kind, X, y, seed):
    """Tune on an inner hold-out, then train the ensemble on all of X. Returns nets, conformal q (reg) and config."""
    rng = np.random.default_rng(seed)
    n = len(X)
    perm = rng.permutation(n)
    va, tr = perm[: n // 5], perm[n // 5:]
    trials = []
    for t in range(TRIALS):
        cfg = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
        _, ep, score = train(kind, X[tr], y[tr], cfg, seed * 1000 + t, X[va], y[va])
        trials.append((score, ep, cfg))
    score, ep, cfg = min(trials, key=lambda r: r[0])
    q = None
    if kind == "reg":
        inner = [train(kind, X[tr], y[tr], cfg, seed * 1000 + 500 + m, epochs=ep)[0] for m in range(MEMBERS)]
        mu, sd = ensemble(kind, inner, X[va])
        q = np.array([conformal_q(np.abs(y[va, j] - mu[:, j]) / sd[:, j]) for j in range(3)])
    nets = [train(kind, X, y, cfg, seed * 1000 + 700 + m, epochs=ep)[0] for m in range(MEMBERS)]
    return nets, q, dict(cfg, epochs=ep, val_loss=score)


def fit(d, seed):
    """Fit on every row of d. q = conformal quantile per output (in sd units of the ensemble)."""
    import torch
    torch.set_num_threads(1)
    X = (D.scale_x(d) * 2 - 1).astype(np.float32)
    r = d.is_reg.to_numpy()
    Y = d.loc[r, Y_COLS].to_numpy(float)
    mean, sd = Y.mean(0), Y.std(0)
    reg, q, cfg_r = fit_kind("reg", X[r], ((Y - mean) / sd).astype(np.float32), seed)
    ok = d.label.notna().to_numpy()
    yc = D.class_index(d.label[ok])
    cls, _, cfg_c = fit_kind("cls", X[ok], yc, seed)
    return {"kind": "mlp", "reg": reg, "cls": cls, "mean": mean, "sd": sd, "cfg_r": cfg_r, "cfg_c": cfg_c, "q": q}


def predict_unit(m, U, X=None):
    """U: (n, 8) inputs on [0, 1] -> mu (n, 3), sd (n, 3), P (n, 3) in physical output units."""
    Xs = (U * 2 - 1).astype(np.float32)
    mu, sd = ensemble("reg", m["reg"], Xs)
    return mu * m["sd"] + m["mean"], sd * m["sd"], ensemble("cls", m["cls"], Xs)
