"""Separate classification, joint angle/displacement, barrier and hinge stress.

The historical models.py and its TPE study remain immutable. This API is a new
experiment: each regression group has its own eligible rows/configuration.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from forward_six.models import (NAMES, default_config, resolve_config, fit_candidate,
    predict_classification, inputs, ROMInputScaler, gp_estimator)
from forward_six.physics import fit_physics
from common import FEATURES, load_data

GROUPS = {
    'angle_displacement': ['angle_deg', 'deformation_mm'],
    'barrier': ['log10_barrier'],
    'stress': ['hinge_transition_max_MPa'],
}
TASKS = ('classification', *GROUPS)


def eligible(d, group):
    if group == 'classification':
        return d.label.notna()
    if group not in GROUPS:
        raise ValueError(group)
    columns = GROUPS[group]
    valid = np.isfinite(d[columns].to_numpy(float)).all(axis=1)
    if group == 'stress':
        valid &= d[columns[0]].to_numpy(float) >= 0
    return d.partial_label.eq('bistable') & valid


def attach_stress(data, scalars, stress):
    """Attach a transition peak only when raw run, landmarks and geometry match.

    No values are inferred for missing/mismatched records. Legacy role labels
    are ignored: current geometry Train/Test/fold assignment is authoritative.
    """
    keys = ['geom_id', 'state']
    if scalars.duplicated(keys).any() or stress.duplicated(keys).any():
        raise ValueError('Duplicate geometry/state stress or scalar record')
    left = scalars[scalars.state.eq('partial') & scalars.stability.eq('bistable')]
    right = stress[stress.state.eq('partial')]
    joined = left.merge(right, on=keys, suffixes=('_current', '_stress'), how='left',
                        validate='one_to_one')
    matched = joined.stress_data_status.eq('transition_available')
    for col in ['archive', 'curve_file', 'mesh_rule', 'peak_set', 'valley_set']:
        matched &= joined[col + '_current'].eq(joined[col + '_stress'])
    for col in FEATURES:
        matched &= np.isclose(joined[col + '_current'], joined[col + '_stress'],
                              rtol=0, atol=1e-10, equal_nan=False)
    values = pd.to_numeric(joined.initial_to_second_max_MPa, errors='coerce')
    matched &= np.isfinite(values) & (values >= 0)
    target = pd.Series(values.where(matched).to_numpy(), index=joined.geom_id)
    result = data.copy()
    result['hinge_transition_max_MPa'] = result.geom_id.map(target)
    return result, {'partial_bi_records': len(joined), 'verified_stress_records': int(matched.sum()),
                    'missing_or_mismatched': int((~matched).sum()),
                    'definition': 'partial: max hinge stress from initial through second energy minimum, inclusive',
                    'source_column': 'initial_to_second_max_MPa', 'unit': 'MPa'}


def prepare_data(stress_path, destination):
    from runtime_paths import RUNS
    scalar_path = RUNS / 'unified-forward/data/scalar_targets.csv'
    data, audit = attach_stress(load_data(), pd.read_csv(scalar_path), pd.read_csv(stress_path))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination / 'geometry_targets.csv', index=False)
    audit['sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [scalar_path, Path(stress_path), RUNS/'unified-forward/data/folds.csv']}
    audit['population'] = {role: {task: int(eligible(data[data.role.eq(role)], task).sum())
                                  for task in TASKS} for role in ['train', 'test']}
    (destination/'data_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return data, audit


def _fit_network(x, y, cfg, seed):
    """Single Gaussian-NLL MLP with a variable number of response heads."""
    import torch
    torch.set_num_threads(1)
    x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
    n_out = y.shape[1]

    def train(rows, epochs, validation=None):
        mean = y[rows].mean(0)
        sd = np.maximum(y[rows].std(0), 1e-8)
        torch.manual_seed(seed)
        activation = {'silu': torch.nn.SiLU, 'tanh': torch.nn.Tanh}[cfg['act']]
        layers, width = [], x.shape[1]
        for _ in range(cfg['depth']):
            layers += [torch.nn.Linear(width, cfg['width']), activation()]
            width = cfg['width']
        net = torch.nn.Sequential(*layers, torch.nn.Linear(width, 2*n_out))
        opt = torch.optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
        xx = torch.tensor(x)
        yy = torch.tensor((y-mean)/sd)
        best, selected, wait = float('inf'), 1, 0
        for epoch in range(1, epochs+1):
            net.train(); opt.zero_grad()
            out = net(xx[rows])
            mu, logv = out[:, :n_out], out[:, n_out:].clamp(-10, 6)
            loss = (0.5*(logv+(yy[rows]-mu)**2/logv.exp())).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite grouped MLP loss')
            loss.backward(); opt.step()
            if validation is not None and (epoch % cfg['check_every'] == 0 or epoch == epochs):
                net.eval()
                with torch.no_grad():
                    score = float((net(xx[validation])[:, :n_out]-yy[validation]).abs().mean())
                if score < best-1e-6:
                    best, selected, wait = score, epoch, 0
                else:
                    wait += cfg['check_every']
                if wait >= cfg['patience']:
                    break
        net.eval()
        return {'network': net, 'mean': mean, 'sd': sd}, selected

    rows = np.arange(len(x))
    epochs = cfg.get('epochs')
    if epochs is None:
        tr, va = train_test_split(rows, test_size=cfg['validation_fraction'], random_state=seed)
        _, epochs = train(tr, cfg['max_epochs'], va)
    if int(epochs) < 1:
        raise ValueError('epochs must be positive')
    model, _ = train(rows, int(epochs))
    return {**model, 'epochs': int(epochs), 'n_outputs': n_out}


def fit_group(name, training, group, config=None, seed=42, *, curves_path=None,
              rom_cache=None, rom_max_nfev=None):
    """Fit one independent task. config is a flat, task-specific setting dict."""
    if name not in NAMES or group not in TASKS:
        raise ValueError((name, group))
    if training.empty or not training.role.eq('train').all():
        raise ValueError('fit_group requires a nonempty Train-only partition')
    if training.geom_id.duplicated().any():
        raise ValueError('Duplicate geometry IDs')
    if group == 'classification':
        return {'group': group, 'model': fit_candidate(name, training,
            {'classification': config or {}}, seed, task='classification', curves_path=curves_path,
            rom_cache=rom_cache, rom_max_nfev=rom_max_nfev)}
    cfg = resolve_config(name, {'regression': config or {}})['regression']
    theta, physics = None, None
    if name.startswith('rom_'):
        if curves_path is None:
            raise ValueError('ROM candidates require completed_force_curves.json')
        theta, physics = fit_physics(training, curves_path, rom_cache, max_nfev=rom_max_nfev)
    rows = training[eligible(training, group)]
    if len(rows) < 2:
        raise ValueError('Need at least two eligible group training rows')
    family = name.removeprefix('rom_')
    x, _, _ = inputs(rows, theta)
    scaler = (ROMInputScaler(mlp=family=='mlp').fit(x)
              if theta is not None and family in ['mlp', 'gp'] else None)
    x = scaler.transform(x) if scaler is not None else (2*x-1 if family=='mlp' else x)
    y = rows[GROUPS[group]].to_numpy(float)
    response_scaler = None
    if family == 'mlp':
        estimator = _fit_network(x, y, cfg, seed)
    elif family == 'gp':
        # sklearn multi-target GP: shared optimized kernel, independent output
        # processes conditional on that kernel; no cross-output covariance model.
        estimator = gp_estimator(cfg, seed, False, x.shape[1]).fit(x, y)
    else:
        from xgboost import XGBRegressor
        response_scaler = StandardScaler().fit(y)
        estimator = XGBRegressor(**cfg, random_state=seed, n_jobs=1,
                                  multi_strategy='one_output_per_tree')
        estimator.fit(x, response_scaler.transform(y))
    return {'schema_version': 'grouped-v1', 'group': group, 'candidate': name,
            'targets': list(GROUPS[group]), 'config': deepcopy(cfg), 'seed': seed,
            'training_ids': training.geom_id.tolist(), 'eligible_training_ids': rows.geom_id.tolist(),
            'theta': theta, 'physics': physics, 'input_scaler': scaler,
            'response_scaler': response_scaler, 'estimator': estimator, 'prediction_averaging': False}


def predict_group(model, data):
    if model['group'] == 'classification':
        return predict_classification(model['model'], data)
    family = model['candidate'].removeprefix('rom_')
    x, _, _ = inputs(data, model['theta'])
    scaler = model['input_scaler']
    x = scaler.transform(x) if scaler is not None else (2*x-1 if family=='mlp' else x)
    estimator = model['estimator']
    if family == 'mlp':
        import torch
        with torch.inference_mode():
            raw = estimator['network'](torch.tensor(x, dtype=torch.float32)).numpy()
        y = raw[:, :len(model['targets'])]*estimator['sd']+estimator['mean']
    else:
        y = np.asarray(estimator.predict(x)).reshape(len(data), len(model['targets']))
        if model['response_scaler'] is not None:
            y = model['response_scaler'].inverse_transform(y)
    if not np.isfinite(y).all():
        raise FloatingPointError('Non-finite grouped prediction')
    return pd.DataFrame(y, index=data.index, columns=model['targets'])
