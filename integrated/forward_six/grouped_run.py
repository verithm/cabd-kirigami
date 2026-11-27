"""Prepare and smoke-check the four-task Forward structure, without HPO."""
import argparse
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
for folder in [BASE, BASE/'forward', BASE.parent/'src']:
    sys.path.insert(0, str(folder))
import joblib
import numpy as np
import pandas as pd
from common import dump
from runtime_paths import RUNS
from forward_six.grouped import TASKS, GROUPS, prepare_data, fit_group, predict_group
from forward_six.models import NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'smoke'])
    parser.add_argument('--stress', type=Path)
    args = parser.parse_args()
    root = RUNS/'forward-grouped'
    if args.action == 'prepare':
        if args.stress is None:
            parser.error('--stress is required for preparation')
        _, audit = prepare_data(args.stress, root/'data')
        dump(root/'protocol.json', {
            'version': 'grouped-v1', 'candidates': list(NAMES), 'tasks': list(TASKS),
            'regression_groups': GROUPS, 'prediction_averaging': False,
            'classification': 'Historical classifier architecture retained, independent task.',
            'angle_displacement': {'mlp': 'Shared hidden layers, 2 response means + 2 log variances',
                'gp': 'Multi-target GP, shared kernel optimized jointly; no output cross covariance',
                'xgboost': 'Multi-target wrapper, one_output_per_tree; common task config/objective'},
            'barrier': 'Independent single-output log10 barrier regressor',
            'stress': audit['definition'], 'stress_unit': 'MPa',
            'tuning': 'Pending. Each task needs independent configuration selection.',
            'historical_tuning': 'forward-six/tuning is old three-response regression; do not overwrite or reuse as new tuning.',
            'test_scored': False})
        print(json.dumps(audit['population'], indent=2))
        return
    d = pd.read_csv(root/'data/geometry_targets.csv')
    # Small execution check; no validation or Test performance claim.
    tr = d[d.role.eq('train') & d.label.notna()].groupby('label', group_keys=False).head(16)
    rows = []
    for name in NAMES:
        family = name.removeprefix('rom_')
        config = ({'width': 8, 'depth': 1, 'epochs': 4} if family == 'mlp' else
                  {'optimize_kernel': False} if family == 'gp' else
                  {'n_estimators': 8, 'max_depth': 2})
        for group in TASKS:
            model = fit_group(name, tr, group, config, seed=42,
                curves_path=RUNS/'unified-forward/data/completed_force_curves.json',
                rom_cache=root/'smoke/rom', rom_max_nfev=2)
            before = predict_group(model, tr.iloc[:6])
            path = root/'smoke'/name/(group+'.joblib')
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, path)
            after = predict_group(joblib.load(path), tr.iloc[:6])
            if group == 'classification':
                np.testing.assert_array_equal(before[0], after[0])
                np.testing.assert_allclose(before[1], after[1], rtol=0, atol=0)
            else:
                np.testing.assert_allclose(before.to_numpy(), after.to_numpy(), rtol=0, atol=0)
                assert list(before.columns) == GROUPS[group]
            rows.append({'candidate': name, 'task': group, 'roundtrip': True})
            print('GROUP_SMOKE_OK', name, group, flush=True)
    dump(root/'smoke/validation.json', {'checks': rows, 'training_geometries': len(tr),
        'test_scored': False, 'not_final_weights': True, 'prediction_averaging': False})


if __name__ == '__main__':
    main()
