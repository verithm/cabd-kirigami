"""Seungmin single MLP/GP and Jinwoo XGBoost, with optional ROM features.

No hyperparameter search or prediction averaging is performed here. Each call to
fit_candidate fits one configuration and one seed. The MLP retains Seungmin's
3-class CE and joint Gaussian-NLL regression; GP retains two-stage classifiers;
XGBoost retains Jinwoo's state-conditioned binary classification.
"""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
for directory in [BASE, BASE / 'forward', BASE.parent / 'src']:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from common import FEATURES, TARGETS, CLASSES
from cabd.geometry import params as geometry
from cabd.models import mlp as original_mlp
from forward_six.physics import fit_physics, physics_features

NAMES = ('mlp', 'xgboost', 'gp', 'rom_mlp', 'rom_xgboost', 'rom_gp')
SCHEMA_VERSION = 1
DEFAULTS = {
    'mlp': {'width': 32, 'depth': 2, 'lr': .001, 'wd': 1e-4, 'act': 'silu',
            'max_epochs': 1200, 'patience': 160, 'check_every': 20, 'validation_fraction': .2},
    'xgboost': {'n_estimators': 300, 'max_depth': 3, 'learning_rate': .05,
                'subsample': .85, 'colsample_bytree': .9, 'min_child_weight': 1.,
                'reg_lambda': 1., 'reg_alpha': 0.},
    'gp': {'length_scale': 1., 'nu': 2.5, 'noise_level': 1e-3, 'signal_level': 1.,
           'optimize_kernel': True, 'max_optimizer_iterations': 60, 'restarts': 0},
}


def default_config(name):
    if name not in NAMES:
        raise ValueError(f'Unknown candidate: {name}')
    cfg = DEFAULTS[name.removeprefix('rom_')]
    return {'classification': deepcopy(cfg), 'regression': deepcopy(cfg)}


def resolve_config(name, supplied=None):
    cfg = default_config(name)
    for part, values in (supplied or {}).items():
        if part not in cfg:
            raise ValueError(f'Unknown config section: {part}')
        unknown = set(values) - set(cfg[part])
        # Fixed epochs supports a separately chosen training duration; no internal holdout.
        if name.removeprefix('rom_') == 'mlp':
            unknown -= {'epochs'}
        if unknown:
            raise ValueError(f'Unknown {part} settings: {sorted(unknown)}')
        cfg[part].update(values)
    return cfg


def normalized_dimensions(d):
    bounds = np.array([geometry.RANGES[k] for k in FEATURES])
    x = d[FEATURES].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite geometry dimensions')
    return (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def inputs(d, theta=None):
    x = normalized_dimensions(d)
    if theta is None:
        return x, x, x
    values, partial_force, full_force = physics_features(d, theta)
    return (np.column_stack([x, values]), np.column_stack([x, partial_force]),
            np.column_stack([x, full_force]))


class ROMInputScaler:
    """Preserve baseline geometry scaling; standardize only the added ROM columns."""
    def __init__(self, mlp=False):
        self.mlp = mlp

    def fit(self, x):
        self.extra = StandardScaler().fit(np.asarray(x)[:, 8:])
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        dimensions = 2 * x[:, :8] - 1 if self.mlp else x[:, :8]
        return np.column_stack([dimensions, self.extra.transform(x[:, 8:])])


def build_network(n_inputs, cfg, classification):
    """Same hidden blocks/output heads as Seungmin, with a variable input width."""
    import torch
    activation = {'silu': torch.nn.SiLU, 'tanh': torch.nn.Tanh}[cfg['act']]
    layers = []
    width = n_inputs
    for _ in range(cfg['depth']):
        layers += [torch.nn.Linear(width, cfg['width']), activation()]
        width = cfg['width']
    layers.append(torch.nn.Linear(width, 3 if classification else 6))
    return torch.nn.Sequential(*layers)


def fit_single_network(x, y, cfg, seed, classification):
    """Select epoch on an inner Train holdout, then refit ONE network on all rows."""
    import torch
    torch.set_num_threads(1)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64 if classification else np.float32)
    kind = 'cls' if classification else 'reg'

    def target_scalers(rows):
        if classification:
            return np.zeros(3), np.ones(3)
        return y[rows].mean(0), np.maximum(y[rows].std(0), 1e-8)

    def train(rows, mean, sd, epochs, validation=None):
        torch.manual_seed(seed)
        net = build_network(x.shape[1], cfg, classification)
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
        xx = torch.tensor(x)
        yy = torch.tensor(y if classification else (y - mean) / sd,
                          dtype=torch.long if classification else torch.float32)
        best, best_epoch, wait = float('inf'), 1, 0
        for epoch in range(1, epochs + 1):
            net.train()
            optimizer.zero_grad()
            loss = original_mlp.loss_fn(kind, net(xx[rows]), yy[rows])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite MLP loss')
            loss.backward()
            optimizer.step()
            if validation is not None and (epoch % cfg['check_every'] == 0 or epoch == epochs):
                net.eval()
                with torch.no_grad():
                    out = net(xx[validation])
                    if classification:
                        from common import metrics_cls
                        score = -metrics_cls(y[validation] == 1, out.argmax(1).numpy() == 1)['f1']
                    else:
                        score = float(torch.abs(out[:, :3] - yy[validation]).mean())
                if score < best - 1e-6:
                    best, best_epoch, wait = score, epoch, 0
                else:
                    wait += cfg['check_every']
                if wait >= cfg['patience']:
                    break
        net.eval()
        return net, best_epoch

    all_rows = np.arange(len(x))
    if cfg.get('epochs') is not None:
        epochs = int(cfg['epochs'])
    else:
        tr, va = train_test_split(all_rows, test_size=cfg['validation_fraction'], random_state=seed,
                                  stratify=y if classification else None)
        mean, sd = target_scalers(tr)
        _, epochs = train(tr, mean, sd, cfg['max_epochs'], va)
    if epochs < 1:
        raise ValueError('epochs must be positive')
    mean, sd = target_scalers(all_rows)
    net, _ = train(all_rows, mean, sd, epochs)
    # One network; no list of members or mixture prediction.
    return {'network': net, 'mean': mean, 'sd': sd, 'epochs': epochs,
            'input_features': x.shape[1]}


class KernelOptimizer:
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations
        self.records = []

    def __call__(self, objective, initial, bounds):
        result = minimize(objective, initial, method='L-BFGS-B', jac=True, bounds=bounds,
                          options={'maxiter': self.max_iterations, 'ftol': 1e-7})
        self.records.append({'success': bool(result.success), 'nit': int(result.nit),
                             'message': str(result.message), 'objective': float(result.fun)})
        return result.x, result.fun


def gp_estimator(cfg, seed, classification, n_features):
    kernel = ConstantKernel(cfg['signal_level'], (1e-3, 1e3)) * Matern(
        np.full(n_features, cfg['length_scale']), (1e-2, 1e3), nu=cfg['nu'])
    options = dict(kernel=kernel, random_state=seed, n_restarts_optimizer=cfg['restarts'],
                   optimizer=KernelOptimizer(cfg['max_optimizer_iterations']) if cfg['optimize_kernel'] else None)
    if classification:
        return GaussianProcessClassifier(**options, max_iter_predict=60)
    options['kernel'] += WhiteKernel(cfg['noise_level'], (1e-8, 1.))
    return GaussianProcessRegressor(**options, normalize_y=True)


def binary_fit(family, x, y, cfg, seed):
    labels = np.asarray(y, dtype=int)
    if np.unique(labels).size != 2:
        raise ValueError('Classifier training subset must contain both classes')
    if family == 'gp':
        return gp_estimator(cfg, seed, True, x.shape[1]).fit(x, labels)
    from xgboost import XGBClassifier
    return XGBClassifier(**cfg, random_state=seed, n_jobs=1).fit(x, labels)


def fit_candidate(name, training, config=None, seed=0, *, curves_path=None, rom_cache=None,
                  rom_max_nfev=None, task='both'):
    """Fit one config on a Train partition. task separates classification/regression HPO."""
    if name not in NAMES:
        raise ValueError(name)
    if task not in ['both', 'classification', 'regression']:
        raise ValueError(task)
    if training.empty or not training.role.eq('train').all():
        raise ValueError('fit_candidate requires a nonempty Train-only partition')
    if training.geom_id.duplicated().any():
        raise ValueError('Duplicate geometry IDs')
    cfg = resolve_config(name, config)
    theta, physics = None, None
    if name.startswith('rom_'):
        if curves_path is None:
            raise ValueError('ROM candidates require completed_force_curves.json')
        theta, physics = fit_physics(training, curves_path, rom_cache, max_nfev=rom_max_nfev)
    family = name.removeprefix('rom_')
    labelled = training[training.label.notna()]
    regression = training[training.is_reg.astype(bool)]
    model = {'schema_version': SCHEMA_VERSION, 'candidate': name, 'seed': seed, 'config': cfg,
             'task': task, 'training_ids': training.geom_id.tolist(), 'theta': theta, 'physics': physics,
             'prediction_averaging': False, 'features': FEATURES, 'targets': TARGETS,
             'classes': CLASSES, 'regression_rows': len(regression), 'classification_rows': len(labelled)}
    if task in ['both', 'regression']:
        if regression.empty:
            raise ValueError('Need partial-bi regression samples')
        reg_x, _, _ = inputs(regression, theta)
        y = regression[TARGETS].to_numpy(float)
        if not np.isfinite(y).all():
            raise ValueError('Non-finite regression targets')
        scaler = ROMInputScaler(mlp=True).fit(reg_x) if family == 'mlp' and theta is not None else (
            ROMInputScaler().fit(reg_x) if family == 'gp' and theta is not None else None)
        xr = scaler.transform(reg_x) if scaler is not None else (2 * reg_x - 1 if family == 'mlp' else reg_x)
        model['reg_scaler'] = scaler
        if family == 'mlp':
            model['regression'] = fit_single_network(xr, y, cfg['regression'], seed, False)
        else:
            regressors = []
            for j in range(3):
                if family == 'gp':
                    reg = gp_estimator(cfg['regression'], seed, False, xr.shape[1])
                else:
                    from xgboost import XGBRegressor
                    reg = XGBRegressor(**cfg['regression'], random_state=seed, n_jobs=1)
                regressors.append(reg.fit(xr, y[:, j]))
            model['regression'] = regressors
    if task in ['both', 'classification']:
        if labelled.empty or not labelled.label.isin(CLASSES).all():
            raise ValueError('Need valid labelled classification samples')
        cls_x, partial_x, full_x = inputs(labelled, theta)
        partial_bi = labelled.partial_label.eq('bistable').to_numpy()
        if family == 'mlp':
            scaler = ROMInputScaler(mlp=True).fit(cls_x) if theta is not None else None
            xc = scaler.transform(cls_x) if scaler is not None else 2 * cls_x - 1
            model['cls_scaler'] = scaler
            model['classification'] = fit_single_network(
                xc, labelled.label.map(dict(zip(CLASSES, range(3)))).to_numpy(), cfg['classification'], seed, True)
        elif family == 'gp':
            first_scaler = ROMInputScaler().fit(partial_x) if theta is not None else None
            second_scaler = ROMInputScaler().fit(full_x[partial_bi]) if theta is not None else None
            first_x = first_scaler.transform(partial_x) if first_scaler is not None else partial_x
            second_x = second_scaler.transform(full_x[partial_bi]) if second_scaler is not None else full_x[partial_bi]
            model['classification'] = {'kind': 'two_stage', 'first_scaler': first_scaler,
                'second_scaler': second_scaler,
                'first': binary_fit('gp', first_x, partial_bi, cfg['classification'], seed),
                'second': binary_fit('gp', second_x, labelled.full_label.eq('monostable').to_numpy()[partial_bi],
                                     cfg['classification'], seed)}
        else:
            x = np.vstack([np.column_stack([partial_x, np.zeros(len(labelled))]),
                           np.column_stack([full_x, np.ones(len(labelled))])])
            target = np.r_[partial_bi, labelled.full_label.eq('bistable').to_numpy()]
            model['classification'] = {'kind': 'state_conditioned',
                'model': binary_fit('xgboost', x, target, cfg['classification'], seed)}
    return model


def _positive(model, x):
    return model.predict_proba(x)[:, list(model.classes_).index(1)]


def predict_regression(model, d):
    reg_x, _, _ = inputs(d, model['theta'])
    family = model['candidate'].removeprefix('rom_')
    scaler = model['reg_scaler']
    xr = scaler.transform(reg_x) if scaler is not None else (2 * reg_x - 1 if family == 'mlp' else reg_x)
    if family == 'mlp':
        import torch
        with torch.inference_mode():
            out = model['regression']['network'](torch.tensor(xr, dtype=torch.float32)).numpy()
        y = out[:, :3] * model['regression']['sd'] + model['regression']['mean']
    else:
        y = np.column_stack([reg.predict(xr) for reg in model['regression']])
    if not np.isfinite(y).all():
        raise FloatingPointError('Non-finite regression prediction')
    return y


def predict_classification(model, d):
    reg_x, partial_x, full_x = inputs(d, model['theta'])
    family = model['candidate'].removeprefix('rom_')
    cls = model['classification']
    if family == 'mlp':
        import torch
        scaler = model['cls_scaler']
        xc = scaler.transform(reg_x) if scaler is not None else 2 * reg_x - 1
        with torch.inference_mode():
            logits = cls['network'](torch.tensor(xc, dtype=torch.float32))
            probabilities = torch.softmax(logits, dim=1).numpy()
        predicted_class = np.asarray(CLASSES)[probabilities.argmax(1)]
    elif family == 'gp':
        xp = cls['first_scaler'].transform(partial_x) if cls['first_scaler'] is not None else partial_x
        xf = cls['second_scaler'].transform(full_x) if cls['second_scaler'] is not None else full_x
        p1, p2 = _positive(cls['first'], xp), _positive(cls['second'], xf)
        probabilities = np.column_stack([p1 * (1 - p2), p1 * p2, 1 - p1])
        predicted_class = np.asarray(CLASSES)[probabilities.argmax(1)]
    else:
        xp = np.column_stack([partial_x, np.zeros(len(d))])
        xf = np.column_stack([full_x, np.ones(len(d))])
        p1, pf = _positive(cls['model'], xp), _positive(cls['model'], xf)
        probabilities = np.column_stack([p1 * pf, p1 * (1 - pf), 1 - p1])
        partial_decision, full_decision = cls['model'].predict(xp), cls['model'].predict(xf)
        predicted_class = np.where(partial_decision == 0, 'mono>mono',
                                   np.where(full_decision == 0, 'bi>mono', 'bi>bi'))
    if not np.isfinite(probabilities).all():
        raise FloatingPointError('Non-finite classification probability')
    return predicted_class, probabilities


def predict_candidate(model, d):
    """Full model prediction. Task-specific checkpoints use predict_classification/regression."""
    if model['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported candidate artifact schema')
    y = predict_regression(model, d)
    predicted_class, probabilities = predict_classification(model, d)
    result = pd.DataFrame(y, columns=TARGETS, index=d.index)
    result.insert(0, 'geom_id', d.geom_id.to_numpy() if 'geom_id' in d else np.arange(len(d)).astype(str))
    result['predicted_class'] = predicted_class
    result['bi_mono'] = predicted_class == 'bi>mono'
    result['p_bi_mono'] = probabilities[:, 1]
    result['barrier_mJ'] = 10. ** result.log10_barrier
    if not np.isfinite(result[TARGETS + ['p_bi_mono', 'barrier_mJ']].to_numpy()).all():
        raise FloatingPointError('Non-finite model prediction')
    return result
