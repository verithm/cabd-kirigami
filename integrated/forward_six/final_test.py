"""Frozen 50-trial configs -> full Train refits -> one fixed Test report.

No tuning, Test-driven selection, production replacement, or Git operations.
Resume only identical frozen inputs, configurations and source code.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time
import warnings

BASE = Path(__file__).resolve().parents[1]
for folder in [BASE, BASE/'forward', BASE.parent/'src']:
    sys.path.insert(0, str(folder))
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from threadpoolctl import threadpool_limits
from runtime_paths import RUNS
from common import dump, metrics_cls, metrics_reg, CLASSES
from forward_six.grouped import GROUPS, TASKS, eligible, fit_group, predict_group
from forward_six.models import NAMES
from forward_six.physics import fit_physics
from forward_six.tune import now

DEST = RUNS/'forward-final50'
DATA = RUNS/'forward-grouped/data/geometry_targets.csv'
CURVES = RUNS/'unified-forward/data/completed_force_curves.json'
SELECTION = RUNS/'forward-refinement/selected_configs_50trials.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trial_folder(name, task, number):
    parent = 'forward-six' if task == 'classification' else 'forward-grouped'
    return RUNS/parent/'tuning/studies'/name/task/f'trial{number:04d}'


def final_config(name, task, selection):
    """Epoch selection is frozen from CV, before any Test model prediction."""
    config = deepcopy(selection['config'])
    if task == 'classification':
        config = config['classification']
    epochs = []
    if name.removeprefix('rom_') == 'mlp':
        record = json.loads((trial_folder(name, task, selection['trial'])/'result.json').read_text())
        epochs = [int(f['diagnostics']['epochs']) for f in record['folds']]
        assert len(epochs) == 5 and min(epochs) > 0
        config['epochs'] = int(np.median(epochs))
    return config, epochs


def freeze():
    DEST.mkdir(parents=True, exist_ok=True)
    chosen = json.loads(SELECTION.read_text())
    selected = {}
    sources = [Path(__file__), *[BASE/'forward_six'/p for p in ['grouped.py','models.py','physics.py']],
               BASE/'forward/common.py', BASE.parent/'src/cabd/models/mlp.py']
    files = [SELECTION, DATA, CURVES, RUNS/'forward-grouped/data/data_audit.json',
             *[RUNS/'unified-forward/data'/p for p in ['folds.csv','jobs.csv','scalar_targets.csv']]]
    for name in NAMES:
        selected[name] = {}
        for task in TASKS:
            choice = chosen[name][task]
            cfg, epochs = final_config(name, task, choice)
            result = trial_folder(name, task, choice['trial'])/'result.json'
            files.append(result)
            selected[name][task] = {'selected_trial': choice['trial'], 'cv_objective': choice['objective'],
                                   'fit_config': cfg, 'cv_fold_epochs': epochs}
    d = pd.read_csv(DATA)
    assert d.geom_id.is_unique and set(d.role) == {'train','test'}
    tr, te = d[d.role.eq('train')], d[d.role.eq('test')]
    assert set(tr.geom_id).isdisjoint(te.geom_id)
    protocol = {'version': 1, 'created_from_budget': 50, 'seed': 42, 'candidates': list(NAMES),
        'tasks': list(TASKS), 'selected': selected, 'epoch_rule': 'median of five CV-selected epochs; fresh single network fit on every eligible Train row, no held-out validation in final fit',
        'test_policy': 'fixed existing Test; no tuning/reselection after Test; CV family recommendations retained separately',
        'classification_positive': 'bi>mono', 'classification_threshold': 'existing prediction rule unchanged',
        'prediction_averaging': False, 'data_source_hashes': {str(p): sha(p) for p in files},
        'source_hashes': {str(p): sha(p) for p in sources},
        'population': {role: {'geometries': len(q), **{t:int(eligible(q,t).sum()) for t in TASKS}}
                       for role,q in [('train',tr),('test',te)]},
        'rom': 'calibrate once on full Train curves only; all ROM tasks reuse same Train-only coefficients',
        'training_ids': tr.geom_id.tolist(), 'test_ids': te.geom_id.tolist(),
        'gp_policy': 'unchanged inner optimizer limits; non-convergence warnings retained',
        'production_replacement': False}
    path = DEST/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Frozen experiment mismatch'
    else:
        dump(path, protocol)
        dump(DEST/'frozen_at.json', {'time': now(), 'protocol_sha256': sha(path), 'test_predictions_started': False})
    return protocol


def worker(name, task):
    protocol = json.loads((DEST/'protocol.json').read_text())
    folder = DEST/'models'/name/task
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder/'fit.json'
    if complete.exists():
        record = json.loads(complete.read_text())
        assert record['model_sha256'] == sha(folder/'model.pkl')
        assert record['protocol_sha256'] == sha(DEST/'protocol.json')
        return name,task,'cached'
    d = pd.read_csv(DATA)
    tr = d[d.role.eq('train')].reset_index(drop=True)
    assert tr.geom_id.tolist() == protocol['training_ids']
    config = protocol['selected'][name][task]['fit_config']
    started = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = fit_group(name,tr,task,config,42,curves_path=CURVES,rom_cache=DEST/'rom')
        core = model['model'] if task == 'classification' else model
        assert set(core['training_ids']) == set(tr.geom_id)
        if task != 'classification':
            assert set(core['eligible_training_ids']) == set(tr.loc[eligible(tr,task),'geom_id'])
        assert not set(core['training_ids']) & set(protocol['test_ids'])
        if name.startswith('rom_'):
            assert core['physics']['solver_success']
            assert set(core['physics']['training_ids']) == set(tr.geom_id)
        probe = tr[eligible(tr,task)].head(8)
        expected = predict_group(model,probe)
        path = folder/'model.pkl'
        temp = folder/'model.tmp.pkl'
        with temp.open('wb') as h: pickle.dump(model,h,pickle.HIGHEST_PROTOCOL)
        temp.replace(path)
        del model
        with path.open('rb') as h: restored=pickle.load(h)
        actual = predict_group(restored,probe)
        if task == 'classification':
            np.testing.assert_array_equal(actual[0],expected[0])
            np.testing.assert_allclose(actual[1],expected[1],rtol=0,atol=0)
        else:
            np.testing.assert_allclose(actual,expected,rtol=0,atol=0)
        core = restored['model'] if task=='classification' else restored
        est = core['classification'] if task=='classification' else core['estimator']
        family = name.removeprefix('rom_')
        diagnostics = {}
        if family == 'mlp': diagnostics = {'epochs': est['epochs']}
        if family == 'gp':
            estimators = [est['first'],est['second']] if task=='classification' else [est]
            diagnostics = {'kernels':[str(e.kernel_) for e in estimators],
                           'optimization':[getattr(e.optimizer,'records',[]) for e in estimators]}
    dump(complete, {'candidate':name,'task':task,'config':config,'seed':42,
        'train_geometry_count':len(tr),'eligible_train_count':int(eligible(tr,task).sum()),
        'training_ids':tr.loc[eligible(tr,task),'geom_id'].tolist(),'test_rows_used':0,
        'protocol_sha256':sha(DEST/'protocol.json'),'model_sha256':sha(path),
        'reload_prediction_verified':True,'diagnostics':diagnostics,
        'warnings':sorted(set(str(w.message) for w in caught)),
        'seconds':time.perf_counter()-started,'finished_at':now()})
    print('FIT_DONE',name,task,round(time.perf_counter()-started,2),flush=True)
    return name,task,'fit'


def evaluate():
    protocol = json.loads((DEST/'protocol.json').read_text())
    d = pd.read_csv(DATA); te = d[d.role.eq('test')].reset_index(drop=True)
    assert te.geom_id.tolist() == protocol['test_ids']
    for name in NAMES:
        for task in TASKS:
            assert (DEST/'models'/name/task/'fit.json').exists(), 'All fits must finish before Test scoring'
    dump(DEST/'test_evaluation_started.json',{'time':now(),'all_24_refits_completed':True,
                                             'frozen_protocol_sha256':sha(DEST/'protocol.json')})
    rows=[]; reports={}; long=[]
    for name in NAMES:
        row={'candidate':name}; reports[name]={}
        for task in TASKS:
            folder=DEST/'models'/name/task
            record=json.loads((folder/'fit.json').read_text())
            assert sha(folder/'model.pkl') == record['model_sha256']
            with (folder/'model.pkl').open('rb') as h: model=pickle.load(h)
            held=te[eligible(te,task)].reset_index(drop=True)
            with threadpool_limits(limits=1): pred=predict_group(model,held)
            out=held[['geom_id']].copy()
            if task=='classification':
                labels,probs=pred; actual=held.label.to_numpy(str)
                assert np.isfinite(probs).all()
                np.testing.assert_allclose(probs.sum(1),1,atol=1e-6)
                metric=metrics_cls(actual=='bi>mono',labels=='bi>mono')
                row.update({k:v for k,v in metric.items()})
                out['actual']=actual;out['predicted']=labels
                for j,c in enumerate(CLASSES):out['probability_'+c]=probs[:,j]
                reports[name][task]={'positive_bi_to_mono':metric,
                    'three_class_order':CLASSES,
                    'three_class_confusion_matrix':confusion_matrix(actual,labels,labels=CLASSES).tolist(),
                    'three_class_metrics':classification_report(actual,labels,labels=CLASSES,output_dict=True,zero_division=0)}
            else:
                actual=held[GROUPS[task]].to_numpy(float); p=pred.to_numpy(float)
                assert np.isfinite(p).all()
                metric={}
                for j,t in enumerate(GROUPS[task]):
                    metric[t]=metrics_reg(actual[:,j],p[:,j])
                    out['actual_'+t]=actual[:,j];out['predicted_'+t]=p[:,j]
                if task=='barrier':
                    av,pv=10.**actual[:,0],10.**p[:,0]
                    assert np.isfinite(pv).all()
                    metric['barrier_mJ']=metrics_reg(av,pv)
                    out['actual_barrier_mJ']=av;out['predicted_barrier_mJ']=pv
                reports[name][task]=metric
                for t,m in metric.items():
                    row.update({t+'_'+k:v for k,v in m.items()})
                    long.append({'candidate':name,'task':task,'target':t,**m})
            predpath=DEST/'predictions'/name/(task+'.csv');predpath.parent.mkdir(parents=True,exist_ok=True)
            out.to_csv(predpath,index=False)
        rows.append(row)
        print('TEST_SCORED',name,flush=True)
    pd.DataFrame(rows).to_csv(DEST/'test_comparison.csv',index=False)
    pd.DataFrame(long).to_csv(DEST/'test_regression_metrics.csv',index=False)
    dump(DEST/'test_metrics.json',reports)
    # Explicitly distinguish CV recommendations from the descriptive Test leader.
    cv=json.loads((RUNS/'forward-refinement/stage_50.json').read_text())['objectives']
    cv_winners={t:(max if t=='classification' else min)(NAMES,key=lambda n:cv[n+'|'+t]) for t in TASKS}
    df=pd.DataFrame(rows).set_index('candidate')
    test_leaders={k:str(df[k].idxmax() if k=='f1' else df[k].idxmin()) for k in
                  ['f1','angle_deg_mae','deformation_mm_mae','log10_barrier_mae','barrier_mJ_mae','hinge_transition_max_MPa_mae']}
    fits=[json.loads(p.read_text()) for p in (DEST/'models').glob('*/*/fit.json')]
    dump(DEST/'complete.json',{'time':now(),'refits':len(fits),'test_scored':True,'tuning_trials_per_study':50,
        'population':protocol['population'],'cv_recommended_candidates':cv_winners,
        'descriptive_test_leaders':test_leaders,'test_used_for_tuning_or_reselection':False,
        'all_models_reloaded_and_checked':all(r['reload_prediction_verified'] for r in fits),
        'total_fit_seconds':sum(r['seconds'] for r in fits),'production_replacement':False,
        'warnings':{r['candidate']+'|'+r['task']:r['warnings'] for r in fits if r['warnings']}})
    files=[p for p in DEST.rglob('*') if p.is_file() and p.name!='manifest.json']
    dump(DEST/'manifest.json',{str(p.relative_to(DEST)):sha(p) for p in files})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args();protocol=freeze()
    if args.freeze_only:
        print(json.dumps({'population':protocol['population'],'selected':protocol['selected']},indent=2));return
    tr=pd.read_csv(DATA);tr=tr[tr.role.eq('train')].reset_index(drop=True)
    _,meta=fit_physics(tr,CURVES,DEST/'rom')
    assert meta['solver_success'] and set(meta['training_ids'])==set(protocol['training_ids'])
    dump(DEST/'rom_full_train.json',meta)
    jobs=[(n,t) for n in ['rom_gp','gp','rom_mlp','mlp','rom_xgboost','xgboost'] for t in TASKS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,n,t) for n,t in jobs]
        for f in as_completed(futures): print('TASK_COMPLETE',f.result(),flush=True)
    evaluate()
    print('FINAL_TEST_COMPLETE',DEST,flush=True)


if __name__=='__main__':
    main()
