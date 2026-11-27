"""Train-only GP classification diagnostic and conditional fresh 50-trial TPE.

Existing studies, final Test and deployed model files are never modified.
All experiments preserve the original two-stage classifier and F1 decision rule.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import hashlib
import json
import pickle
from pathlib import Path
import sqlite3
import sys
import time
import warnings
from unittest.mock import patch
BASE=Path(__file__).resolve().parents[1]
for p in [BASE,BASE/'forward',BASE.parent/'src']:sys.path.insert(0,str(p))
import numpy as np
import pandas as pd
import optuna
from optuna.trial import TrialState
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel,Matern,RBF
from threadpoolctl import threadpool_limits
from common import dump,metrics_cls
from runtime_paths import RUNS
from forward_six import models as original
from forward_six.physics import fit_physics
from forward_six.tune import now

DEST=RUNS/'forward-gp-improvement'
DATA=RUNS/'forward-grouped/data/geometry_targets.csv'
CURVES=RUNS/'unified-forward/data/completed_force_curves.json'
SELECTION=RUNS/'forward-refinement/selected_configs_50trials.json'
ROM=RUNS/'forward-six/tuning/rom'
NAMES=['gp','rom_gp']
VARIANTS=['bounds_only','bounds_iterations','rbf','matern05']
SPACE={'kernel':['matern05','matern15','matern25','rbf'],
       'ard':[True,False], 'signal_upper':[1e3,1e4,1e6],
       'length_upper':[1e3,1e5], 'signal_initial_log':[.01,1e3],
       'length_initial_log':[.05,50.], 'max_optimizer_iterations':150,
       'restarts':0,'max_iter_predict':60}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def training():
    d=pd.read_csv(DATA)
    # Test excluded immediately; never predict or inspect Test scores here.
    return d[d.role.eq('train')].reset_index(drop=True)


def old_choice(name):return json.loads(SELECTION.read_text())[name]['classification']


def base_config(name):
    cfg=old_choice(name)['config']['classification']
    return {'kernel':'matern'+str(int(cfg['nu']*10)), 'ard':True,
            'signal_upper':1e3,'length_upper':1e3,'length_initial':cfg['length_scale'],
            'signal_initial':cfg['signal_level'],'max_optimizer_iterations':60}


def pilot_config(name,variant):
    cfg=base_config(name)
    cfg.update(signal_upper=1e6,length_upper=1e5)
    if variant!='bounds_only':cfg['max_optimizer_iterations']=150
    if variant in ['rbf','matern05']:cfg['kernel']=variant
    return cfg


def estimator(cfg,seed,n_features):
    length=np.full(n_features,cfg['length_initial']) if cfg['ard'] else cfg['length_initial']
    bounds=(1e-2,cfg['length_upper'])
    if cfg['kernel']=='rbf':kernel=RBF(length,bounds)
    else:kernel=Matern(length,bounds,nu={'matern05':.5,'matern15':1.5,'matern25':2.5}[cfg['kernel']])
    kernel=ConstantKernel(cfg['signal_initial'],(1e-3,cfg['signal_upper']))*kernel
    return GaussianProcessClassifier(kernel=kernel,random_state=seed,n_restarts_optimizer=0,
        optimizer=original.KernelOptimizer(cfg['max_optimizer_iterations']),max_iter_predict=60)


def fit(name,tr,cfg):
    """Change only GP kernel builder, restoring original module immediately."""
    def builder(unused,seed,classification,n_features):
        assert classification
        return estimator(cfg,seed,n_features)
    with patch.object(original,'gp_estimator',builder):
        model=original.fit_candidate(name,tr,old_choice(name)['config'],42,
            task='classification',curves_path=CURVES,rom_cache=ROM)
    model['gp_improvement_config']=cfg
    return model


def fold_worker(name,cfg,fold,folder):
    folder=Path(folder);meta=folder/f'fold{fold}.json';pred=folder/f'fold{fold}.npz'
    if meta.exists() and pred.exists():
        r=json.loads(meta.read_text());assert r['config']==cfg
        return r
    d=training();tr=d[d.fold.ne(fold)];held=d[d.fold.eq(fold)&d.label.notna()].reset_index(drop=True)
    begun=time.perf_counter()
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model=fit(name,tr,cfg)
        assert set(model['training_ids'])==set(tr.geom_id)
        assert not set(model['training_ids'])&set(held.geom_id)
        labels,probabilities=original.predict_classification(model,held)
        actual=held.label.eq('bi>mono').to_numpy();predicted=labels=='bi>mono'
        parts=[model['classification'][k] for k in ['first','second']]
        diagnostics={'kernels':[str(m.kernel_) for m in parts],
            'kernel_thetas':[m.kernel_.theta.tolist() for m in parts],
            'kernel_optimization':[m.optimizer.records for m in parts]}
    folder.mkdir(parents=True,exist_ok=True)
    temp=pred.with_suffix('.tmp.npz')
    np.savez_compressed(temp,ids=held.geom_id.to_numpy(str),actual=actual,predicted=predicted,
                        probabilities=probabilities,labels=labels)
    temp.replace(pred)
    record={'candidate':name,'config':cfg,'fold':fold,'metrics':metrics_cls(actual,predicted),
        'n_train_geometries':len(tr),'eligible_training_ids':tr.loc[tr.label.notna(),'geom_id'].tolist(),
        'n_validation':len(held),'diagnostics':diagnostics,'seconds':time.perf_counter()-begun,
        'warnings':sorted(set(str(w.message) for w in caught)),'finished_at':now(),'test_scored':False}
    dump(meta,record)
    print('GP_FOLD_DONE',name,folder.name,fold,round(record['seconds'],2),flush=True)
    return record


def aggregate(name,cfg,folder):
    folder=Path(folder);d=training();ys=[];ps=[];folds=[]
    for f in range(5):
        r=json.loads((folder/f'fold{f}.json').read_text());assert r['config']==cfg
        held=d[d.fold.eq(f)&d.label.notna()]
        with np.load(folder/f'fold{f}.npz') as z:
            np.testing.assert_array_equal(z['ids'],held.geom_id.to_numpy(str))
            np.testing.assert_array_equal(z['actual'],held.label.eq('bi>mono'))
            np.testing.assert_allclose(z['probabilities'].sum(axis=1),1,atol=1e-6)
            y,p=z['actual'],z['predicted']
        m=metrics_cls(y,p)
        for k,v in m.items():np.testing.assert_allclose(v,r['metrics'][k],rtol=1e-12,atol=1e-12)
        ys.append(y);ps.append(p);folds.append(r)
    metric=metrics_cls(np.concatenate(ys),np.concatenate(ps))
    result={'candidate':name,'config':cfg,'objective':metric['f1'],'metrics':metric,
        'folds':folds,'seconds':sum(r['seconds'] for r in folds),'finished_at':now(),'test_scored':False}
    dump(folder/'result.json',result)
    return result


def frozen_protocol():
    DEST.mkdir(parents=True,exist_ok=True)
    files=[DATA,CURVES,SELECTION,Path(__file__),BASE/'forward_six/models.py',BASE/'forward_six/physics.py',
           BASE/'forward/common.py',RUNS/'unified-forward/data/folds.csv']
    p={'version':1,'seed':42,'tasks':['classification'],'models':NAMES,'folds':5,
       'selection_metric':'pooled OOF positive bi>mono F1, unchanged hard decision rule',
       'pilot':{n:{v:pilot_config(n,v) for v in VARIANTS} for n in NAMES},
       'pilot_trigger':'F1 gain >= 0.002 with >=3 improving folds OR F1 gain >=0.005',
       'new_tuning_if_triggered':{'complete_trials_per_qualified_family':50,'sampler':'TPE seed42, first 10 startup trials including queued pilot winner','pruning':False,'space':SPACE},
       'test_policy':'Never score Test or choose settings on Test in this experiment',
       'comparison_disclosure':'GP gets additional pilot/search-design effort; original equal-50 comparison preserved. New-50 is not equal total historical opportunity.',
       'source_data_sha256':{str(p):sha(p) for p in files},
       'baseline':{n:old_choice(n) for n in NAMES}}
    path=DEST/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p,'Protocol changed: use new experiment directory'
    else:dump(path,p);dump(DEST/'started_at.json',{'time':now(),'test_scored':False})
    return p


def evaluate_pilot(workers):
    jobs=[(n,v,f) for n in NAMES for v in VARIANTS for f in range(5)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fold_worker,n,pilot_config(n,v),f,str(DEST/'pilot'/n/v)) for n,v,f in jobs]
        for future in as_completed(futures):future.result()
    summary=[];decisions={}
    for n in NAMES:
        choice=old_choice(n)
        oldroot=RUNS/'forward-six/tuning/studies'/n/'classification'/f'trial{choice["trial"]:04d}'
        old=json.loads((oldroot/'result.json').read_text())
        baseline=np.array([x['metrics']['f1'] for x in old['folds']])
        candidates=[]
        for v in VARIANTS:
            result=aggregate(n,pilot_config(n,v),DEST/'pilot'/n/v)
            gains=np.array([x['metrics']['f1'] for x in result['folds']])-baseline
            delta=result['objective']-old['objective']
            trigger=bool((delta>=.002 and (gains>1e-12).sum()>=3) or delta>=.005)
            row={'candidate':n,'variant':v,'baseline_f1':old['objective'],'pilot_f1':result['objective'],
                'gain_F1_percentage_points':100*delta,'folds_improved':int((gains>1e-12).sum()),
                'folds_worse':int((gains< -1e-12).sum()),'trigger_new_50':trigger,
                'precision':result['metrics']['precision'],'recall':result['metrics']['recall'],
                'paired_fold_gains':gains.tolist(),'config':result['config']}
            summary.append(row);candidates.append(row)
        best=max(candidates,key=lambda r:r['pilot_f1'])
        qualified=[r for r in candidates if r['trigger_new_50']]
        winner=max(qualified,key=lambda r:r['pilot_f1']) if qualified else best
        decisions[n]={'run_new_50':bool(qualified),'pilot_winner':winner,
            'basis':'predeclared Train-only OOF trigger; not statistical significance'}
    dump(DEST/'pilot_summary.json',summary);dump(DEST/'pilot_decisions.json',decisions)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['paired_fold_gains','config']} for r in summary]).to_csv(DEST/'pilot_summary.csv',index=False)
    print('PILOT_DECISIONS',json.dumps(decisions),flush=True)
    return decisions


def suggest(trial):
    return {'kernel':trial.suggest_categorical('kernel',SPACE['kernel']),
        'ard':trial.suggest_categorical('ard',SPACE['ard']),
        'signal_upper':trial.suggest_categorical('signal_upper',SPACE['signal_upper']),
        'length_upper':trial.suggest_categorical('length_upper',SPACE['length_upper']),
        'signal_initial':trial.suggest_float('signal_initial',.01,1e3,log=True),
        'length_initial':trial.suggest_float('length_initial',.05,50.,log=True),
        'max_optimizer_iterations':150}


def run_study(name,decision,workers):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folder=DEST/'studies'/name;folder.mkdir(parents=True,exist_ok=True)
    samplerpath=folder/'sampler.pkl'
    sampler=pickle.loads(samplerpath.read_bytes()) if samplerpath.exists() else optuna.samplers.TPESampler(seed=42,n_startup_trials=10)
    study=optuna.create_study(study_name=name+'-improved-classification',storage='sqlite:///'+str(folder/'study.sqlite3'),
        sampler=sampler,load_if_exists=True,direction='maximize')
    if not study.trials:
        initial=deepcopy(decision['pilot_winner']['config']);initial.pop('max_optimizer_iterations')
        study.enqueue_trial(initial)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while len(study.get_trials(states=[TrialState.COMPLETE]))<50:
            running=study.get_trials(states=[TrialState.RUNNING]);assert len(running)<=1
            if running:
                trial=optuna.Trial(study,running[0]._trial_id)
            else:trial=study.ask()
            cfg=suggest(trial)
            temp=samplerpath.with_suffix('.tmp');temp.write_bytes(pickle.dumps(study.sampler));temp.replace(samplerpath)
            trialfolder=folder/f'trial{trial.number:04d}'
            identity={'candidate':name,'trial':trial.number,'config':cfg}
            idpath=trialfolder/'config.json'
            if idpath.exists():assert json.loads(idpath.read_text())==identity
            else:dump(idpath,identity)
            fs=[pool.submit(fold_worker,name,cfg,f,str(trialfolder)) for f in range(5)]
            for future in as_completed(fs):future.result()
            result=aggregate(name,cfg,trialfolder)
            study.tell(trial,result['objective'])
            dump(folder/'best.json',{'trial':study.best_trial.number,'objective':study.best_value,'params':study.best_trial.params})
            print('NEW_TPE_TRIAL_DONE',name,len(study.get_trials(states=[TrialState.COMPLETE])),result['objective'],study.best_value,flush=True)
    records=[]
    for t in study.get_trials(states=[TrialState.COMPLETE]):
        records.append({'trial':t.number,'objective':t.value,'best_so_far':max(x.value for x in study.get_trials(states=[TrialState.COMPLETE]) if x.number<=t.number)})
    pd.DataFrame(records).to_csv(folder/'history.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--pilot-only',action='store_true');args=p.parse_args()
    frozen_protocol()
    d=training()
    assert d.geom_id.is_unique and set(d.fold)==set(range(5))
    # Pre-existing ROM coefficients are fold-Train-only; never use full-Train ROM here.
    for f in range(5):
        _,meta=fit_physics(d[d.fold.ne(f)],CURVES,ROM)
        assert meta['solver_success'] and set(meta['training_ids'])==set(d.loc[d.fold.ne(f),'geom_id'])
    decisions=evaluate_pilot(args.workers)
    if args.pilot_only:return
    for n in NAMES:
        if decisions[n]['run_new_50']:run_study(n,decisions[n],args.workers)
    dump(DEST/'complete.json',{'time':now(),'pilot_completed':True,'test_scored':False,
        'new_50_completed_for':[n for n in NAMES if decisions[n]['run_new_50']],
        'original_studies_modified':False,'production_models_modified':False})


if __name__=='__main__':main()
