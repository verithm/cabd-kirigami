"""Check optimized GP fold parallelism against the unchanged evaluator."""
from pathlib import Path
import sys,tempfile,json
from unittest.mock import patch
BASE=Path(__file__).resolve().parents[1]
for folder in [BASE,BASE/'forward',BASE.parent/'src',BASE/'tests']:
    sys.path.insert(0,str(folder))
import numpy as np
from threadpoolctl import threadpool_limits
from test_forward_six import synthetic_training
from forward_six import tune
from forward_six.parallel_classification import evaluate_parallel
from runtime_paths import RUNS


def main():
    d=synthetic_training(30);d['fold']=np.arange(len(d))%5
    cfg={'classification':{'optimize_kernel':True,'max_optimizer_iterations':3}}
    with tempfile.TemporaryDirectory() as tmp,threadpool_limits(limits=1):
        with patch.object(tune,'DEST',Path(tmp)/'sequential'):
            a=tune.evaluate_trial('gp','classification',0,cfg,d)
        with patch.object(tune,'DEST',Path(tmp)/'parallel'):
            b=evaluate_parallel('gp','classification',0,cfg,d,fold_workers=2)
        assert a['objective']==b['objective'] and a['metrics']==b['metrics']
        for fold in range(5):
            tail=Path('studies/gp/classification/trial0000')/f'fold{fold}.npz'
            with np.load(Path(tmp)/'sequential'/tail) as x,np.load(Path(tmp)/'parallel'/tail) as y:
                for key in ['ids','actual','predicted','probabilities']:
                    np.testing.assert_array_equal(x[key],y[key])
    path=RUNS/'forward-refinement/optimized-parallel-validation.json'
    path.write_text(json.dumps({'optimized_GP':True,'inner_iterations_for_check':3,'folds_checked':5,
        'labels_exact':True,'probabilities_exact':True,'metrics_exact':True,
        'trial_objective_exact':True,'test_scored':False},indent=2)+'\n')
    print('OPTIMIZED_PARALLEL_EQUIVALENCE_PASSED',flush=True)


if __name__=='__main__':main()
