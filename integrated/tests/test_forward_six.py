"""Behavior checks for the new candidates; no HPO and no Test-set evaluation."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

_ISOLATED_RUNS = tempfile.TemporaryDirectory()
os.environ.setdefault('CABD_INTEGRATED_RUNS', _ISOLATED_RUNS.name)
BASE = Path(__file__).resolve().parents[1]
for folder in [BASE, BASE / 'forward', BASE.parent / 'src']:
    sys.path.insert(0, str(folder))
import numpy as np
import pandas as pd
from common import FEATURES, TARGETS, CLASSES
from cabd.geometry import params as G
from cabd.models import rom
from forward_six.models import default_config, resolve_config, build_network, fit_candidate, predict_candidate
from forward_six.physics import fit_physics


def synthetic_training(n=24):
    rows = []
    for i in range(n):
        label = CLASSES[i % 3]
        p = dict(G.REF, t_h=.18 + .005 * i)
        rows.append(dict(geom_id=f'TR{i:04d}', role='train', label=label, is_reg=label != 'mono>mono',
                         partial_label='monostable' if label == 'mono>mono' else 'bistable',
                         full_label='bistable' if label == 'bi>bi' else 'monostable',
                         angle_deg=55. + i / 2, deformation_mm=10. + i / 10,
                         log10_barrier=-1. + i / 100, **{k:p[k] for k in FEATURES}))
    return pd.DataFrame(rows)


class SixCandidateTests(unittest.TestCase):
    def test_parallel_gp_folds_match_original_evaluator(self):
        from unittest.mock import patch
        from forward_six import tune
        from forward_six.parallel_classification import evaluate_parallel
        d = synthetic_training(30)
        d['fold'] = np.arange(len(d)) % 5
        cfg = {'classification': {'optimize_kernel': False}}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(tune, 'DEST', Path(tmp)/'sequential'):
                a = tune.evaluate_trial('gp','classification',0,cfg,d)
            with patch.object(tune, 'DEST', Path(tmp)/'parallel'):
                b = evaluate_parallel('gp','classification',0,cfg,d,fold_workers=2)
            self.assertEqual(a['objective'],b['objective'])
            self.assertEqual(a['metrics'],b['metrics'])
            for fold in range(5):
                tail = Path('studies/gp/classification/trial0000')/f'fold{fold}.npz'
                with np.load(Path(tmp)/'sequential'/tail) as x, np.load(Path(tmp)/'parallel'/tail) as y:
                    for key in ['ids','actual','predicted','probabilities']:
                        np.testing.assert_array_equal(x[key],y[key])

    def test_train_only_guard_and_no_implicit_unknown_config(self):
        d = synthetic_training()
        d.loc[0, 'role'] = 'test'
        with self.assertRaisesRegex(ValueError, 'Train-only'):
            fit_candidate('mlp', d)
        with self.assertRaisesRegex(ValueError, 'Unknown regression settings'):
            resolve_config('mlp', {'regression': {'learnign_rate': .001}})

    def test_rom_preserves_baseline_geometry_scaling(self):
        from forward_six.models import ROMInputScaler
        x = np.arange(44, dtype=float).reshape(4, 11) / 44
        gp = ROMInputScaler().fit(x).transform(x)
        mlp = ROMInputScaler(mlp=True).fit(x).transform(x)
        np.testing.assert_array_equal(gp[:, :8], x[:, :8])
        np.testing.assert_array_equal(mlp[:, :8], 2 * x[:, :8] - 1)
        np.testing.assert_allclose(gp[:, 8:].mean(0), 0, atol=1e-12)

    def test_single_mlp_is_direct_head_not_average(self):
        import torch
        d = synthetic_training()
        cfg = {part: {'epochs': 2, 'width': 8, 'depth': 1} for part in ['classification', 'regression']}
        m = fit_candidate('mlp', d, cfg, seed=2)
        self.assertNotIn('nets', m['regression'])
        self.assertNotIn('nets', m['classification'])
        self.assertFalse(m['prediction_averaging'])
        from forward_six.models import normalized_dimensions
        with torch.no_grad():
            raw = m['regression']['network'](torch.tensor(2 * normalized_dimensions(d) - 1, dtype=torch.float32)).numpy()
        want = raw[:, :3] * m['regression']['sd'] + m['regression']['mean']
        np.testing.assert_allclose(predict_candidate(m, d)[TARGETS], want, rtol=0, atol=0)
        self.assertEqual(m['regression']['network'][0].in_features, 8)
        self.assertEqual(build_network(11, default_config('rom_mlp')['regression'], False)[0].in_features, 11)

    def test_rom_ignores_nontraining_curves_and_separates_partition_cache(self):
        d = synthetic_training(6)
        known = np.array([0., 0., 0., -5., 1.])
        curves = {}
        for _, row in d.iterrows():
            for state in ['partial', 'full']:
                force = rom.curve(rom.geometry(row), state, known)[1]
                curves[f'{row.geom_id}|{state}'] = {'u': rom.U_GRID.tolist(), 'F': force.tolist()}
        # If the calibrator read unrelated rows, this invalid curve would raise.
        curves['TE9999|partial'] = {'u': [0], 'F': [float('nan')]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'curves.json'
            path.write_text(json.dumps(curves))
            theta, meta = fit_physics(d, path, Path(tmp) / 'cache', max_nfev=2)
            self.assertTrue(all(key.startswith('TR') for key in meta['calibration_curve_keys']))
            self.assertEqual(meta['training_ids'], sorted(d.geom_id))
            again, cached = fit_physics(d, path, Path(tmp) / 'cache', max_nfev=2)
            np.testing.assert_array_equal(theta, again)
            _, other = fit_physics(d.iloc[:3], path, Path(tmp) / 'cache', max_nfev=2)
            self.assertNotEqual(meta['fingerprint'], other['fingerprint'])
            self.assertNotIn('TE9999', ''.join(other['calibration_curve_keys']))


if __name__ == '__main__':
    unittest.main()

class TuningExecutionTests(unittest.TestCase):
    def test_task_specific_fit_does_not_train_other_head(self):
        from forward_six.models import predict_classification, predict_regression
        d = synthetic_training()
        cfg = {part: {'epochs': 2, 'width': 8, 'depth': 1} for part in ['classification', 'regression']}
        classifier = fit_candidate('mlp', d, cfg, task='classification')
        self.assertNotIn('regression', classifier)
        self.assertEqual(predict_classification(classifier, d)[1].shape, (len(d), 3))
        regressor = fit_candidate('mlp', d, cfg, task='regression')
        self.assertNotIn('classification', regressor)
        self.assertEqual(predict_regression(regressor, d).shape, (len(d), 3))

    def test_hpo_five_fold_metrics_resume_without_retraining(self):
        from unittest.mock import patch
        from forward_six import tune
        d = synthetic_training(45)
        d['fold'] = np.arange(len(d)) % 5
        cfg = {'regression': {'epochs': 2, 'width': 8, 'depth': 1}}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(tune, 'DEST', Path(tmp)):
                result = tune.evaluate_trial('mlp','regression',0,cfg,d)
                self.assertEqual(len(result['folds']),5)
                self.assertAlmostEqual(result['objective'], np.mean([f['metrics']['scaled_mae'] for f in result['folds']]))
                self.assertFalse(result['test_scored'])
                with patch.object(tune,'fit_candidate',side_effect=AssertionError('Cached folds must not refit')):
                    resumed = tune.evaluate_trial('mlp','regression',0,cfg,d)
                self.assertEqual(result['objective'],resumed['objective'])
