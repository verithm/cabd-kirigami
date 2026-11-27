"""Independent angle/displacement regressors; preserve historical group source."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from forward_six.grouped import _fit_network, predict_group
from forward_six.models import NAMES, resolve_config, inputs, ROMInputScaler, gp_estimator
from forward_six.physics import fit_physics

GROUPS = {'angle': ['angle_deg'], 'displacement': ['deformation_mm']}


def eligible(data, group):
    if group not in GROUPS:
        raise ValueError(group)
    return data.partial_label.eq('bistable') & np.isfinite(data[GROUPS[group]].to_numpy(float)).all(1)


def fit_group(name, training, group, config=None, seed=42, *, curves_path=None, rom_cache=None):
    if name not in NAMES or group not in GROUPS:
        raise ValueError((name, group))
    if training.empty or not training.role.eq('train').all():
        raise ValueError('Independent fitting requires a nonempty Train-only partition')
    if training.geom_id.duplicated().any():
        raise ValueError('Duplicate geometry IDs')
    cfg = resolve_config(name, {'regression': config or {}})['regression']
    theta, physics = None, None
    if name.startswith('rom_'):
        if curves_path is None:
            raise ValueError('ROM candidate requires completed force curves')
        theta, physics = fit_physics(training, curves_path, rom_cache)
    rows = training[eligible(training, group)]
    if len(rows) < 2:
        raise ValueError('Need at least two eligible independent regression rows')
    family = name.removeprefix('rom_')
    x, _, _ = inputs(rows, theta)
    scaler = (ROMInputScaler(mlp=family=='mlp').fit(x)
              if theta is not None and family in ['mlp','gp'] else None)
    x = scaler.transform(x) if scaler is not None else (2*x-1 if family=='mlp' else x)
    y = rows[GROUPS[group]].to_numpy(float)
    response_scaler = None
    if family == 'mlp':
        estimator = _fit_network(x, y, cfg, seed)
    elif family == 'gp':
        estimator = gp_estimator(cfg, seed, False, x.shape[1]).fit(x, y)
    else:
        from xgboost import XGBRegressor
        response_scaler = StandardScaler().fit(y)
        estimator = XGBRegressor(**cfg, random_state=seed, n_jobs=1,
                                  multi_strategy='one_output_per_tree')
        estimator.fit(x, response_scaler.transform(y))
    return {'schema_version':'independent-v1', 'group':group, 'candidate':name,
        'targets':list(GROUPS[group]), 'config':cfg, 'seed':seed,
        'training_ids':training.geom_id.tolist(), 'eligible_training_ids':rows.geom_id.tolist(),
        'theta':theta, 'physics':physics, 'input_scaler':scaler,
        'response_scaler':response_scaler, 'estimator':estimator, 'prediction_averaging':False}
