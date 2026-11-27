"""Regression groups do not share weights, missing-label filters or tuning configs."""
from pathlib import Path
import sys
import unittest
import numpy as np

BASE = Path(__file__).resolve().parents[1]
for folder in [BASE, BASE/'forward', BASE.parent/'src', BASE/'tests']:
    sys.path.insert(0, str(folder))
from test_forward_six import synthetic_training
from forward_six.grouped import fit_group, predict_group, eligible, attach_stress


class GroupedTests(unittest.TestCase):
    def test_missing_other_group_target_does_not_exclude_sample(self):
        d = synthetic_training()
        d['hinge_transition_max_MPa'] = 4.
        d.loc[0, 'log10_barrier'] = np.nan
        self.assertTrue(eligible(d, 'angle_displacement')[0])
        self.assertTrue(eligible(d, 'stress')[0])
        self.assertFalse(eligible(d, 'barrier')[0])
        self.assertFalse(eligible(d, 'stress')[2])  # partial mono

    def test_mlp_heads_and_group_training_are_independent(self):
        d = synthetic_training()
        d['hinge_transition_max_MPa'] = np.linspace(2, 6, len(d))
        d.loc[0, 'log10_barrier'] = np.nan
        models = {g: fit_group('mlp', d, g, {'width': 8, 'depth': 1, 'epochs': 2})
                  for g in ['angle_displacement', 'barrier', 'stress']}
        for group, count in [('angle_displacement', 2), ('barrier', 1), ('stress', 1)]:
            m = models[group]
            self.assertEqual(m['estimator']['network'][-1].out_features, 2*count)
            self.assertEqual(predict_group(m, d).shape, (len(d), count))
        self.assertIn(d.geom_id[0], models['angle_displacement']['eligible_training_ids'])
        self.assertNotIn(d.geom_id[0], models['barrier']['eligible_training_ids'])
        self.assertIsNot(models['barrier']['estimator'], models['stress']['estimator'])

    def test_gp_pair_has_one_shared_kernel_and_two_response_columns(self):
        d = synthetic_training()
        m = fit_group('gp', d, 'angle_displacement', {'optimize_kernel': False})
        self.assertEqual(m['estimator'].y_train_.shape[1], 2)
        self.assertEqual(predict_group(m, d).shape, (len(d), 2))
        d.loc[0, 'role'] = 'test'
        with self.assertRaisesRegex(ValueError, 'Train-only'):
            fit_group('gp', d, 'angle_displacement')

    def test_stress_landmark_mismatch_is_not_silently_used(self):
        import pandas as pd
        d = synthetic_training(3)
        s = d[['geom_id', 't_h','L_h','t_a','L_a','L','t_f','c','rho']].copy()
        s['state'] = 'partial'; s['stability'] = 'bistable'
        for k, v in [('archive','a.zip'),('curve_file','partial.csv'),('mesh_rule','m'),
                     ('peak_set',2),('valley_set',5)]:
            s[k] = v
        h = s.copy(); h['initial_to_second_max_MPa'] = 3.
        h['stress_data_status'] = 'transition_available'
        h.loc[0, 'valley_set'] = 6
        merged, audit = attach_stress(d, s, h)
        self.assertTrue(pd.isna(merged.hinge_transition_max_MPa[0]))
        self.assertEqual(merged.hinge_transition_max_MPa[1], 3.)
        self.assertEqual(audit['missing_or_mismatched'], 1)

    def test_grouped_cv_resume_and_group_specific_objectives(self):
        import tempfile
        from unittest.mock import patch
        from forward_six import grouped_tune as tune
        d = synthetic_training(30)
        d['fold'] = np.arange(len(d)) % 5
        d['hinge_transition_max_MPa'] = np.linspace(2, 6, len(d))
        cfg = {'width': 8, 'depth': 1, 'epochs': 2}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(tune, 'DEST', Path(tmp)):
                for group in ['angle_displacement', 'barrier', 'stress']:
                    r = tune.evaluate_trial('mlp', group, 0, cfg, d)
                    self.assertEqual(len(r['folds']), 5)
                    values = []
                    for fold in range(5):
                        with np.load(Path(tmp)/'studies/mlp'/group/'trial0000'/f'fold{fold}.npz') as z:
                            error = abs(z['actual']-z['predicted']).mean(0)
                            values.append(float(np.mean(error/z['training_target_sd']))
                                          if group=='angle_displacement' else float(error.mean()))
                    self.assertAlmostEqual(r['objective'], np.mean(values))
                    with patch.object(tune, 'fit_group', side_effect=AssertionError('Cached fold refit')):
                        resumed = tune.evaluate_trial('mlp', group, 0, cfg, d)
                    self.assertEqual(r['objective'], resumed['objective'])
