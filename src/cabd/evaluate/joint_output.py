"""Should theta2 and u2 be predicted jointly? 5-fold CV on the 1536 training geometries (bi only, 1217), test set untouched.

theta2 and u2 are strongly correlated (r = 0.93; GP test residuals r = 0.95), so four GP variants are compared,
all with the same GP as the GP model's regressor (const x ARD Matern 5/2 + white, normalize_y, 19 restarts, seed 0):
  indep     current: one GP per output
  pca       outputs standardised and rotated to principal components, one GP per component, rotated back
            (a simple linear model of coregionalization; gives a full 2x2 predictive covariance)
  kin       theta2 = ROM arm kinematics at the predicted u2 + GP on the kinematic residual (theta2 - kin(u2_true));
            ROM factors of the 1536-geometry fit (rom_theta_n1536)
  chain     theta2 GP with u2 as a 9th input (true u2 when fitting, predicted u2 when predicting)
Metrics per variant, pooled over folds: RMSE of theta2 and u2, joint hit rate max(|e|/tau) <= 1 over both outputs
(the inverse success rule restricted to these two), and the 2-D Gaussian NLL of the truth under the predictive
distribution (indep, kin and chain assume independent errors; pca uses its full covariance).
Result (results/joint_output/cv_summary.csv): pca and kin stay within about 1.5 % of indep in RMSE and joint hit rate,
chain is worse (theta2 RMSE +6 %); pca has the lowest 2-D NLL because it models the residual correlation.
Full run: 30 GP fits, ~1.6 h with 15 jobs.

usage: python -m cabd.evaluate.joint_output [--jobs 15] [--summary-only]
writes results/joint_output/cv_predictions.csv and cv_summary.csv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .. import data as D
from ..config import RESULTS, TAU as TAU3
from ..models import io
from ..models.gp import reg_model
from ..models.rom import kinematic_theta2

OUT = RESULTS / "joint_output"
TAU = np.array(TAU3[:2])
SEED = 0


def kin(d, u2):
    return kinematic_theta2(d, u2, io.load("rom_theta_n1536")["theta"])


def fit_one(name, X, y):
    warnings.filterwarnings("ignore")
    return name, reg_model(SEED, X.shape[1]).fit(X, y)


def run(jobs):
    from joblib import Parallel, delayed
    from sklearn.model_selection import KFold
    d = D.train_set(1536)
    d = d[d.is_reg & d.logE.notna()].reset_index(drop=True)
    X = D.scale_x(d)
    Y = d[["theta2", "u2"]].to_numpy(float)
    K = kin(d, Y[:, 1])
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(X))

    tasks = []
    for f, (tr, te) in enumerate(folds):
        m, s = Y[tr].mean(0), Y[tr].std(0)
        Z = (Y[tr] - m) / s
        w, V = np.linalg.eigh(np.cov(Z.T))
        Pc = Z @ V
        tasks += [((f, "theta2"), X[tr], Y[tr, 0]), ((f, "u2"), X[tr], Y[tr, 1]),
                  ((f, "pc0"), X[tr], Pc[:, 0]), ((f, "pc1"), X[tr], Pc[:, 1]),
                  ((f, "kinres"), X[tr], Y[tr, 0] - K[tr]),
                  ((f, "chain"), np.column_stack([X[tr], (Y[tr, 1] - m[1]) / s[1]]), Y[tr, 0])]
    fits = dict(Parallel(n_jobs=jobs, verbose=5)(delayed(fit_one)(*t) for t in tasks))
    return predictions(d, X, Y, folds, fits)


def predictions(d, X, Y, folds, fits):
    rows = []
    for f, (tr, te) in enumerate(folds):
        m, s = Y[tr].mean(0), Y[tr].std(0)
        w, V = np.linalg.eigh(np.cov(((Y[tr] - m) / s).T))

        def g(k):
            return fits[(f, k)].predict(X[te], return_std=True)
        (t_mu, t_sd), (u_mu, u_sd) = g("theta2"), g("u2")
        (p0, s0), (p1, s1) = g("pc0"), g("pc1")
        # pca: back-rotate mean and covariance  y = m + s * (V @ pc)
        mu_p = m + s * (np.column_stack([p0, p1]) @ V.T)
        cov_p = np.einsum("ij,nj,kj->nik", V, np.column_stack([s0, s1]) ** 2, V) * np.outer(s, s)
        # kin: theta2 = kin(u2_hat) + residual GP; sd from the residual GP and the u2 sd through the kinematic slope
        k_hat = kin(d.iloc[te], u_mu)
        k_eps = kin(d.iloc[te], u_mu + 1e-3)
        slope = (k_eps - k_hat) / 1e-3
        r_mu, r_sd = g("kinres")
        k_mu, k_sd = k_hat + r_mu, np.sqrt(r_sd ** 2 + (slope * u_sd) ** 2)
        c_mu, c_sd = fits[(f, "chain")].predict(np.column_stack([X[te], (u_mu - m[1]) / s[1]]), return_std=True)
        variants = {"indep": (t_mu, t_sd, u_mu, u_sd, None),
                    "pca": (mu_p[:, 0], np.sqrt(cov_p[:, 0, 0]), mu_p[:, 1], np.sqrt(cov_p[:, 1, 1]), cov_p[:, 0, 1]),
                    "kin": (k_mu, k_sd, u_mu, u_sd, None),
                    "chain": (c_mu, c_sd, u_mu, u_sd, None)}
        for name, (tm, ts, um, us, cxy) in variants.items():
            rows.append(pd.DataFrame(dict(variant=name, fold=f, geom_id=d.geom_id.iloc[te].to_numpy(),
                                          theta2=Y[te, 0], u2=Y[te, 1], theta2_mu=tm, theta2_sd=ts, u2_mu=um, u2_sd=us,
                                          cov=np.zeros_like(tm) if cxy is None else cxy)))
    return pd.concat(rows, ignore_index=True)


def summarize(p):
    out = []
    for name, g in p.groupby("variant", sort=False):
        e = np.column_stack([g.theta2_mu - g.theta2, g.u2_mu - g.u2])
        cxy = g["cov"].to_numpy(float)          # not g.cov (DataFrame.cov method)
        S = np.stack([np.stack([g.theta2_sd ** 2, cxy], -1), np.stack([cxy, g.u2_sd ** 2], -1)], -2)
        Sinv = np.linalg.inv(S)
        nll = 0.5 * (np.einsum("ni,nij,nj->n", e, Sinv, e) + np.log(np.linalg.det(S)) + 2 * np.log(2 * np.pi))
        hit = (np.abs(e) / TAU).max(1) <= 1
        out.append(dict(variant=name, n=len(g), rmse_theta2=np.sqrt((e[:, 0] ** 2).mean()), rmse_u2=np.sqrt((e[:, 1] ** 2).mean()),
                        joint_hit=hit.mean(), nll2d=nll.mean(), resid_corr=np.corrcoef(e.T)[0, 1]))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=15)
    ap.add_argument("--summary-only", action="store_true", help="recompute cv_summary.csv from cv_predictions.csv")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        p = pd.read_csv(args.out / "cv_predictions.csv")
    else:
        p = run(args.jobs)
        p.to_csv(args.out / "cv_predictions.csv", index=False, float_format="%.6g")
    sm = summarize(p)
    sm.to_csv(args.out / "cv_summary.csv", index=False, float_format="%.5f")
    print(sm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
