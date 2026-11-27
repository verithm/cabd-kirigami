"""Independent targets must not affect each other's learned regression."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
BASE = Path(__file__).resolve().parents[1]
for folder in [BASE, BASE/'forward', BASE.parent/'src', BASE/'tests']:
    sys.path.insert(0,str(folder))
from test_forward_six import synthetic_training
from forward_six.single_models import fit_group, predict_group, eligible


class IndependentTests(unittest.TestCase):
    def test_other_target_changes_do_not_change_angle_network(self):
        d = synthetic_training(30)
        cfg = {'width':8,'depth':1,'epochs':2}
        a = fit_group('mlp',d,'angle',cfg)
        changed = d.copy(); changed['deformation_mm'] = np.nan
        changed['log10_barrier'] = 999.
        b = fit_group('mlp',changed,'angle',cfg)
        np.testing.assert_array_equal(predict_group(a,d),predict_group(b,d))
        self.assertEqual(a['estimator']['network'][-1].out_features,2)
        self.assertTrue(eligible(changed,'angle')[0])
        self.assertFalse(eligible(changed,'displacement')[0])
        self.assertEqual(predict_group(a,d).columns.tolist(),['angle_deg'])

    def test_single_gp_and_train_guard(self):
        d = synthetic_training()
        m = fit_group('gp',d,'displacement',{'optimize_kernel':False})
        self.assertEqual(m['estimator'].y_train_.shape[1],1)
        self.assertEqual(predict_group(m,d).shape,(len(d),1))
        d.loc[0,'role']='test'
        with self.assertRaisesRegex(ValueError,'Train-only'):
            fit_group('gp',d,'angle')

    def test_single_cv_objectives_are_physical_mae_and_resume(self):
        from forward_six import single_tune as tune
        d = synthetic_training(30); d['fold'] = np.arange(len(d))%5
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(tune,'DEST',Path(tmp)):
                for group in ['angle','displacement']:
                    r=tune.evaluate_trial('mlp',group,0,{'width':8,'depth':1,'epochs':2},d)
                    scores=[]
                    for fold in range(5):
                        with np.load(Path(tmp)/'studies/mlp'/group/'trial0000'/f'fold{fold}.npz') as z:
                            scores.append(float(abs(z['actual']-z['predicted']).mean()))
                    self.assertAlmostEqual(r['objective'],np.mean(scores))
                    with patch.object(tune,'fit_group',side_effect=AssertionError('Cached fold refit')):
                        self.assertEqual(tune.evaluate_trial('mlp',group,0,{'width':8,'depth':1,'epochs':2},d)['objective'],r['objective'])
