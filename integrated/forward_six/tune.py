"""Resumable TPE studies: one Train-only 5-fold evaluation per trial, no Test scoring."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time
import warnings
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parents[1]
for folder in [BASE, BASE / 'forward', BASE.parent / 'src']:
    sys.path.insert(0, str(folder))
import numpy as np
import pandas as pd
import optuna
from optuna.trial import TrialState
from threadpoolctl import threadpool_limits
from runtime_paths import RUNS
from common import load_data, dump, FEATURES, TARGETS, metrics_cls, reg_report
from forward_six.models import (NAMES, default_config, fit_candidate, predict_regression,
                                 predict_classification, predict_candidate)
from forward_six.physics import fit_physics

DEST = RUNS / 'forward-six' / 'tuning'
TASKS = ['classification', 'regression']
SEED = 42
SPACE = {
    'mlp': {'width': [16,32,64,128,256], 'depth': [1,2,3,4],
            'lr_log': [1e-4,1e-2], 'wd': 'zero or log[1e-7,1e-2]', 'act': ['silu','tanh']},
    'xgboost': {'max_depth': [2,6], 'min_child_weight_log': [1,20],
                'learning_rate_log': [.01,.2], 'n_estimators': [100,1500],
                'subsample': [.6,1.], 'colsample_bytree': [.6,1.],
                'reg_lambda_log': [1e-3,30.], 'reg_alpha': 'zero or log[1e-6,10]'},
    'gp': {'nu': [1.5,2.5], 'length_scale_init_log': [.1,10.],
           'signal_level_init_log': [.1,10.], 'noise_level_init_log_regression_only': [1e-6,.1]},
}


def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()


def suggest(trial, name, task):
    family = name.removeprefix('rom_')
    if family == 'mlp':
        cfg = {'width': trial.suggest_categorical('width', [16,32,64,128,256]),
               'depth': trial.suggest_int('depth',1,4),
               'lr': trial.suggest_float('lr',1e-4,1e-2,log=True),
               'wd': trial.suggest_float('wd_value',1e-7,1e-2,log=True) if trial.suggest_categorical('wd_enabled',[False,True]) else 0.,
               'act': trial.suggest_categorical('act',['silu','tanh'])}
    elif family == 'xgboost':
        cfg = {'max_depth': trial.suggest_int('max_depth',2,6),
               'min_child_weight': trial.suggest_float('min_child_weight',1,20,log=True),
               'learning_rate': trial.suggest_float('learning_rate',.01,.2,log=True),
               'n_estimators': trial.suggest_int('n_estimators',100,1500),
               'subsample': trial.suggest_float('subsample',.6,1.),
               'colsample_bytree': trial.suggest_float('colsample_bytree',.6,1.),
               'reg_lambda': trial.suggest_float('reg_lambda',1e-3,30.,log=True),
               'reg_alpha': trial.suggest_float('reg_alpha_value',1e-6,10.,log=True) if trial.suggest_categorical('reg_alpha_enabled',[False,True]) else 0.}
    else:
        cfg = {'nu': trial.suggest_categorical('nu',[1.5,2.5]),
               'length_scale': trial.suggest_float('length_scale',.1,10.,log=True),
               'signal_level': trial.suggest_float('signal_level',.1,10.,log=True)}
        if task == 'regression':
            cfg['noise_level'] = trial.suggest_float('noise_level',1e-6,.1,log=True)
    return {task: cfg}


def protocol(training):
    files = {name:hashlib.sha256((RUNS/'unified-forward/data'/name).read_bytes()).hexdigest()
             for name in ['jobs.csv','scalar_targets.csv','folds.csv','completed_force_curves.json']}
    code = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), BASE/'forward_six/models.py', BASE/'forward_six/physics.py']}
    return {'version': 1, 'candidates': list(NAMES), 'tasks': TASKS, 'sampler': 'Optuna 4.6.0 TPESampler',
            'startup_trials': 10, 'seed': SEED, 'cv_folds': 5, 'pruning': False,
            'classification_objective': 'maximize pooled OOF bi>mono F1',
            'regression_objective': 'minimize mean of 5 fold-train-SD-normalized 3-output MAEs',
            'regression_settings': 'one common setting for three independently fitted GPs/XGBoosts; joint MLP',
            'search_space': SPACE, 'data_sha256': files, 'source_sha256': code,
            'training_shapes': len(training), 'classification_shapes': int(training.label.notna().sum()),
            'regression_shapes': int(training.is_reg.sum()), 'prediction_averaging': False,
            'training_seed_per_fold': SEED, 'test_scored': False, 'refit_full_train': False,
            'gp_internal_optimizer': {'type': 'L-BFGS-B marginal likelihood', 'max_iterations': 60,
                                      'random_restarts': 0, 'ARD': True,
                                      'note': 'TPE searches nu and initialization; continuous kernel parameters are learned internally.'},
            'defaults': {name:default_config(name) for name in NAMES}}


def get_train():
    # load_data reads the frozen source table; only Train rows are passed to fitting/scoring.
    d = load_data()
    return d[d.role.eq('train')].copy().reset_index(drop=True)


def prepare():
    DEST.mkdir(parents=True,exist_ok=True)
    d = get_train()
    current = protocol(d)
    p = DEST/'protocol.json'
    if p.exists():
        if json.loads(p.read_text()) != current:
            raise ValueError('Tuning protocol/code/data changed. Use a separate experiment directory.')
    else:
        dump(p,current)
    for fold in range(5):
        training = d[d.fold != fold]
        theta, meta = fit_physics(training,RUNS/'unified-forward/data/completed_force_curves.json',DEST/'rom')
        if not meta['solver_success']:
            raise RuntimeError(f'ROM calibration did not converge for fold {fold}: {meta["solver_message"]}')
        assert set(meta['training_ids']).isdisjoint(set(d[d.fold == fold].geom_id))
        dump(DEST/'rom'/f'fold{fold}-provenance.json',meta)
        print('ROM_READY',fold,len(meta['training_ids']),flush=True)
    return d


def fitted_diagnostics(model, task):
    family = model['candidate'].removeprefix('rom_')
    if family == 'mlp':
        return {'epochs': model[task]['epochs']}
    if family == 'gp':
        models = model['regression'] if task == 'regression' else [model['classification']['first'],model['classification']['second']]
        return {'kernels':[str(m.kernel_) for m in models],
                'kernel_optimization':[getattr(m.optimizer,'records',[]) for m in models]}
    return {}


def evaluate_trial(name,task,number,config,training):
    folder = DEST/'studies'/name/task/f'trial{number:04d}'
    identity = {'candidate':name,'task':task,'trial':number,'config':config,'seed':SEED}
    p = folder/'config.json'
    if p.exists():
        assert json.loads(p.read_text()) == identity
    else:
        dump(p,identity)
    parts, fold_scores = [], []
    for fold in range(5):
        held_all = training[training.fold == fold]
        held = held_all[held_all.label.notna() if task == 'classification' else held_all.is_reg].reset_index(drop=True)
        fit_rows = training[training.fold != fold]
        meta_path = folder/f'fold{fold}.json'
        pred_path = folder/f'fold{fold}.npz'
        if meta_path.exists() and pred_path.exists():
            record = json.loads(meta_path.read_text())
            with np.load(pred_path) as z:
                ids, actual, predicted = z['ids'].astype(str), z['actual'], z['predicted']
            np.testing.assert_array_equal(ids,held.geom_id.to_numpy(str))
        else:
            begun = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model = fit_candidate(name,fit_rows,config,SEED,task=task,
                                      curves_path=RUNS/'unified-forward/data/completed_force_curves.json',
                                      rom_cache=DEST/'rom')
                if task == 'classification':
                    labels, probabilities = predict_classification(model,held)
                    actual = held.label.eq('bi>mono').to_numpy()
                    predicted = labels == 'bi>mono'
                    metric = metrics_cls(actual,predicted)
                    extra = {'probabilities':probabilities}
                else:
                    actual = held[TARGETS].to_numpy(float)
                    predicted = predict_regression(model,held)
                    sd = np.maximum(fit_rows.loc[fit_rows.is_reg,TARGETS].std(ddof=0).to_numpy(),1e-8)
                    metric = reg_report(actual,predicted)
                    metric['scaled_mae'] = float(np.mean(np.abs(actual-predicted).mean(0)/sd))
                    extra = {'training_target_sd':sd}
            record = {'fold':fold,'n_train_geometries':len(fit_rows),'n_validation':len(held),
                      'metrics':metric,'diagnostics':fitted_diagnostics(model,task),
                      'warnings':sorted(set(str(w.message) for w in caught)),
                      'seconds':time.perf_counter()-begun,'finished_at':now()}
            ids = held.geom_id.to_numpy(str)
            folder.mkdir(parents=True,exist_ok=True)
            temp = pred_path.with_suffix('.tmp.npz')
            np.savez_compressed(temp,ids=ids,actual=actual,predicted=predicted,**extra)
            temp.replace(pred_path)
            dump(meta_path,record)
            del model
            print('FOLD_DONE',name,task,number,fold,round(record['seconds'],2),flush=True)
        parts.append((actual,predicted))
        fold_scores.append(record)
    y,pred = (np.concatenate([p[j] for p in parts]) for j in [0,1])
    if task == 'classification':
        metrics = metrics_cls(y,pred)
        objective = metrics['f1']
    else:
        metrics = reg_report(y,pred)
        objective = float(np.mean([r['metrics']['scaled_mae'] for r in fold_scores]))
        metrics['scaled_mae'] = objective
    result = {**identity,'objective':objective,'metrics':metrics,
              'folds':fold_scores,'seconds':sum(r['seconds'] for r in fold_scores),
              'completed_at':now(),'test_scored':False}
    dump(folder/'result.json',result)
    return result


def study_worker(name,task,total_trials):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    training = get_train()
    folder = DEST/'studies'/name/task
    folder.mkdir(parents=True,exist_ok=True)
    state_path = folder/'sampler.pkl'
    sampler = pickle.loads(state_path.read_bytes()) if state_path.exists() else optuna.samplers.TPESampler(seed=SEED,n_startup_trials=10)
    storage = optuna.storages.RDBStorage('sqlite:///'+str(folder/'study.sqlite3'),
                                       engine_kwargs={'connect_args':{'timeout':60}})
    study = optuna.create_study(study_name=f'{name}-{task}',storage=storage,sampler=sampler,
                               direction='maximize' if task=='classification' else 'minimize',load_if_exists=True)
    if not study.trials:
        baseline = default_config(name)[task]
        if name.endswith('mlp'):
            params={k:baseline[k] for k in ['width','depth','lr','act']}
            params.update(wd_enabled=True,wd_value=baseline['wd'])
        elif name.endswith('xgboost'):
            params={k:baseline[k] for k in ['max_depth','min_child_weight','learning_rate','n_estimators','subsample','colsample_bytree','reg_lambda']}
            params.update(reg_alpha_enabled=False)
        else:
            params={k:baseline[k] for k in ['nu','length_scale','signal_level']}
            if task=='regression':params['noise_level']=baseline['noise_level']
        study.enqueue_trial(params)
    with threadpool_limits(limits=1):
        while len(study.get_trials(states=[TrialState.COMPLETE])) < total_trials:
            running = study.get_trials(states=[TrialState.RUNNING])
            if len(running)>1:
                raise RuntimeError('Multiple RUNNING trials; only one worker per study is supported')
            trial = optuna.trial.Trial(study,running[0]._trial_id) if running else study.ask()
            config = suggest(trial,name,task)
            temp=state_path.with_suffix('.tmp');temp.write_bytes(pickle.dumps(study.sampler));temp.replace(state_path)
            try:
                result=evaluate_trial(name,task,trial.number,config,training)
            except Exception as exc:
                dump(folder/f'trial{trial.number:04d}'/'failure.json',{'error':repr(exc),'time':now()})
                study.tell(trial,state=TrialState.FAIL)
                # Do not silently replace failed settings and alter the equal-trial comparison.
                raise
            trial.set_user_attr('config',config)
            trial.set_user_attr('metrics',result['metrics'])
            trial.set_user_attr('seconds',result['seconds'])
            study.tell(trial,result['objective'])
            temp=state_path.with_suffix('.tmp');temp.write_bytes(pickle.dumps(study.sampler));temp.replace(state_path)
            study.trials_dataframe().to_csv(folder/'trials.csv',index=False)
            dump(folder/'best.json',{'candidate':name,'task':task,'trial':study.best_trial.number,
                                    'objective':study.best_value,'config':study.best_trial.user_attrs['config'],
                                    'metrics':study.best_trial.user_attrs['metrics'],
                                    'completed_trials':len(study.get_trials(states=[TrialState.COMPLETE])),
                                    'test_scored':False})
            print('TRIAL_DONE',name,task,trial.number,result['objective'],flush=True)
    return name,task,len(study.get_trials(states=[TrialState.COMPLETE]))


def summarize():
    rows=[]
    for name in NAMES:
        for task in TASKS:
            folder=DEST/'studies'/name/task
            path=folder/'best.json'
            if not path.exists():continue
            best=json.loads(path.read_text());metrics=best['metrics']
            r={k:best[k] for k in ['candidate','task','trial','objective','completed_trials']}
            r['test_scored']=False
            if task=='classification':r.update(metrics)
            else:
                for target in TARGETS+['barrier_mJ']:
                    for key,value in metrics[target].items():r[f'{target}_{key}']=value
            rows.append(r)
    if rows:
        pd.DataFrame(rows).to_csv(DEST/'summary.csv',index=False)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total-trials',type=int,default=20)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--models',nargs='+',choices=NAMES,default=list(NAMES))
    parser.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.report_only:
        print(json.dumps(summarize(),ensure_ascii=False,indent=2));return
    prepare()
    if args.prepare_only:return
    begun=now()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(study_worker,name,task,args.total_trials) for name in args.models for task in args.tasks]
        for future in as_completed(futures):
            print('STUDY_DONE',future.result(),flush=True);summarize()
    rows=summarize()
    dump(DEST/'batch-complete.json',{'started_at':begun,'finished_at':now(),
                                    'requested_total_per_study':args.total_trials,
                                    'studies':[{'candidate':r['candidate'],'task':r['task'],'completed_trials':r['completed_trials']} for r in rows],
                                    'test_scored':False})


if __name__=='__main__':
    main()
