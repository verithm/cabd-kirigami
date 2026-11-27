"""Kernel diagnostics preserve baseline semantics and Train-only pipeline."""
from pathlib import Path
import sys
import unittest
import numpy as np
BASE=Path(__file__).resolve().parents[1]
for p in [BASE,BASE/'forward',BASE.parent/'src']:sys.path.insert(0,str(p))
from forward_six.models import gp_estimator
from forward_six.gp_improvement import estimator


class GPImprovementTests(unittest.TestCase):
    def test_new_search_initial_signal_is_inside_every_cap(self):
        from forward_six.gp_controls import new_suggest
        class SmallCapHighInitial:
            def suggest_categorical(self,name,values):return values[0]
            def suggest_float(self,name,low,high,**kwargs):return high
        cfg=new_suggest(SmallCapHighInitial())
        self.assertLessEqual(cfg['signal_initial'],cfg['signal_upper'])
        self.assertLessEqual(cfg['length_initial'],cfg['length_upper'])
        self.assertEqual(estimator(cfg,42,8).optimizer.max_iterations,150)

    def test_control_changes_only_one_factor(self):
        from unittest.mock import patch
        from forward_six import gp_controls
        baseline={'kernel':'matern25','signal_upper':1e3,'length_upper':1e3,'max_optimizer_iterations':60}
        with patch.object(gp_controls.gp,'base_config',return_value=baseline.copy()):
            changed=gp_controls.config('gp','rbf_original_bounds')
            self.assertEqual({k for k in changed if changed[k]!=baseline[k]},{'kernel'})
        with patch.object(gp_controls.gp,'base_config',return_value=baseline.copy()):
            changed=gp_controls.config('gp','signal_cap100')
            self.assertEqual({k for k in changed if changed[k]!=baseline[k]},{'signal_upper'})

    def test_old_bounds_reproduce_native_fit(self):
        x=np.random.default_rng(2).uniform(size=(30,3));y=(x[:,0]+x[:,1]>.9).astype(int)
        cfg={'length_initial':1.2,'signal_initial':.7,'signal_upper':1e3,'length_upper':1e3,
             'ard':True,'kernel':'matern25','max_optimizer_iterations':60}
        native=gp_estimator({'length_scale':1.2,'signal_level':.7,'nu':2.5,'restarts':0,
                            'optimize_kernel':True,'max_optimizer_iterations':60},42,True,3).fit(x,y)
        changed=estimator(cfg,42,3).fit(x,y)
        np.testing.assert_allclose(native.kernel_.theta,changed.kernel_.theta,rtol=0,atol=0)
        np.testing.assert_allclose(native.predict_proba(x),changed.predict_proba(x),rtol=0,atol=0)

    def test_structural_kernel_search_changes_actual_kernel(self):
        cfg={'length_initial':1.,'signal_initial':1.,'signal_upper':1e6,'length_upper':1e5,
             'ard':True,'kernel':'rbf','max_optimizer_iterations':150}
        m=estimator(cfg,42,8)
        self.assertEqual(m.kernel.n_dims,9)
        np.testing.assert_allclose(np.exp(m.kernel.bounds)[0],[1e-3,1e6])
        self.assertEqual(m.optimizer.max_iterations,150)
        cfg['ard']=False
        self.assertEqual(estimator(cfg,42,8).kernel.n_dims,2)
        cfg['kernel']='matern05'
        self.assertEqual(estimator(cfg,42,8).kernel.k2.nu,.5)


if __name__=='__main__':unittest.main()
