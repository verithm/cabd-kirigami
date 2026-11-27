"""Resumable independent angle/displacement TPE, compared with frozen joint studies."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
import optuna
from optuna.trial import TrialState
from threadpoolctl import threadpool_limits
from runtime_paths import RUNS
from common import dump, metrics_reg, FEATURES
from forward_six.models import NAMES, default_config
from forward_six.single_models import GROUPS, eligible, fit_group, predict_group
from forward_six.physics import fit_physics
from forward_six.tune import suggest, SPACE, now

DEST = RUNS/'forward-independent/tuning'
DATA = RUNS/'forward-grouped/data/geometry_targets.csv'
CURVES = RUNS/'unified-forward/data/completed_force_curves.json'
ROM = RUNS/'forward-six/tuning/rom'
SEED = 42


def get_train():
    d = pd.read_csv(DATA)
    return d[d.role.eq('train')].reset_index(drop=True)


def experiment_protocol():
    files = [DATA, RUNS/'forward-grouped/data/data_audit.json', CURVES,
        *[RUNS/'unified-forward/data'/n for n in ['jobs.csv','scalar_targets.csv','folds.csv']],
        RUNS/'forward-six/tuning/selected_configs_20trials.json',
        RUNS/'forward-six/tuning/comparison_20trials.csv',
        RUNS/'forward-grouped/tuning/selected_configs_20trials.json',
        RUNS/'forward-grouped/tuning/results_20trials.csv']
    sources = [Path(__file__), *[BASE/'forward_six'/n for n in
        ['single_models.py','grouped.py','models.py','physics.py','tune.py']], BASE/'forward/common.py',
        BASE.parent/'src/cabd/models/mlp.py']
    return {'version': 1, 'candidates': list(NAMES), 'groups': GROUPS,
        'sampler': 'Optuna 4.6.0 TPESampler', 'seed': SEED, 'cv_folds': 5,
        'initial_trials': '1 baseline + 9 random', 'startup_trials': 10,
        'pruning': False, 'search_space': SPACE,
        'objectives': {'angle': 'mean 5-fold angle MAE in degrees',
            'displacement': 'mean 5-fold displacement MAE in mm'},
        'classification': 'Retain previous 20-trial result; not re-tuned or changed.',
        'prediction_averaging': False, 'test_scored': False, 'refit_full_train': False,
        'data_sha256': {str(p.relative_to(RUNS)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'source_sha256': {str(p.relative_to(BASE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'comparison_limitations': ['Joint: one 20-trial study for angle/displacement; independent: one 20-trial study per response.',
            'Compare output-level OOF errors on identical geometries, not differently normalized joint objective values.',
            'No extension beyond 20 until candidate selection; independent response tuning has twice the total opportunities of one joint study.'],
        'defaults': {n: default_config(n)['regression'] for n in NAMES}}


def prepare():
    DEST.mkdir(parents=True, exist_ok=True)
    proto = experiment_protocol()
    path = DEST/'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != proto:
            raise ValueError('Independent tuning code/data changed; use a new experiment directory')
    else:
        dump(path, proto)
    d = get_train()
    if d.geom_id.duplicated().any() or set(d.fold.unique()) != set(range(5)):
        raise ValueError('Invalid geometry Train folds')
    from forward_six.grouped import eligible as joint_eligible
    joint_ids = set(d.loc[joint_eligible(d, 'angle_displacement'), 'geom_id'])
    for group in GROUPS:
        if set(d.loc[eligible(d, group), 'geom_id']) != joint_ids:
            raise ValueError('Independent/joint eligible geometries differ; choose an explicit common-population protocol')
    for fold in range(5):
        tr = d[d.fold.ne(fold)]
        _, meta = fit_physics(tr, CURVES, ROM)
        assert meta['solver_success']
        assert set(meta['training_ids']) == set(tr.geom_id)
        assert not set(meta['training_ids']) & set(d[d.fold.eq(fold)].geom_id)
        dump(DEST/'rom'/f'fold{fold}-provenance.json', meta)
        print('INDEPENDENT_ROM_READY', fold, len(tr), flush=True)
    population = {g: int(eligible(d, g).sum()) for g in GROUPS}
    dump(DEST/'population.json', population)
    return d


def evaluate_trial(name, group, number, config, training):
    folder = DEST/'studies'/name/group/f'trial{number:04d}'
    identity = {'candidate': name, 'group': group, 'trial': number, 'config': config, 'seed': SEED}
    path = folder/'config.json'
    if path.exists():
        assert json.loads(path.read_text()) == identity
    else:
        dump(path, identity)
    targets = GROUPS[group]
    folds, parts = [], []
    for fold in range(5):
        tr = training[training.fold.ne(fold)]
        held = training[training.fold.eq(fold) & eligible(training, group)].reset_index(drop=True)
        meta_path, pred_path = folder/f'fold{fold}.json', folder/f'fold{fold}.npz'
        if meta_path.exists() and pred_path.exists():
            record = json.loads(meta_path.read_text())
            with np.load(pred_path) as z:
                actual, predicted = z['actual'], z['predicted']
                np.testing.assert_array_equal(z['ids'].astype(str), held.geom_id.to_numpy(str))
                np.testing.assert_allclose(actual, held[targets].to_numpy(float), rtol=0, atol=0)
        else:
            begun = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model = fit_group(name, tr, group, config, SEED, curves_path=CURVES, rom_cache=ROM)
                actual = held[targets].to_numpy(float)
                predicted = predict_group(model, held).to_numpy(float)
                sd = np.maximum(tr.loc[eligible(tr, group), targets].std(ddof=0).to_numpy(), 1e-8)
                metric = {t: metrics_reg(actual[:, j], predicted[:, j]) for j, t in enumerate(targets)}
                if group == 'barrier':
                    restored = 10.**predicted[:, 0]
                    if not np.isfinite(restored).all():
                        raise FloatingPointError('Non-finite restored barrier')
                    metric['barrier_mJ'] = metrics_reg(10.**actual[:, 0], restored)
                objective = float(np.mean(np.abs(actual-predicted).mean(0)/sd)) if group=='angle_displacement' else float(np.abs(actual-predicted).mean())
                family = name.removeprefix('rom_')
                estimator = model['estimator']
                diagnostics = ({'epochs': estimator['epochs']} if family=='mlp' else
                    {'kernels': str(estimator.kernel_), 'kernel_optimization': getattr(estimator.optimizer, 'records', [])} if family=='gp' else {})
            record = {'fold': fold, 'n_train_geometries': len(tr),
                'eligible_training_ids': model['eligible_training_ids'], 'n_validation': len(held),
                'metrics': metric, 'objective': objective, 'diagnostics': diagnostics,
                'warnings': sorted(set(str(w.message) for w in caught)),
                'seconds': time.perf_counter()-begun, 'finished_at': now()}
            temp = pred_path.with_suffix('.tmp.npz')
            np.savez_compressed(temp, ids=held.geom_id.to_numpy(str), actual=actual,
                                predicted=predicted, training_target_sd=sd)
            temp.replace(pred_path)
            dump(meta_path, record)
            del model
            print('INDEPENDENT_FOLD_DONE', name, group, number, fold, round(record['seconds'], 2), flush=True)
        folds.append(record); parts.append((actual, predicted))
    actual, predicted = [np.concatenate([p[j] for p in parts]) for j in [0,1]]
    metrics = {t: metrics_reg(actual[:, j], predicted[:, j]) for j,t in enumerate(targets)}
    if group == 'barrier':
        metrics['barrier_mJ'] = metrics_reg(10.**actual[:, 0], 10.**predicted[:, 0])
    result = {**identity, 'targets': targets, 'objective': float(np.mean([r['objective'] for r in folds])),
        'metrics': metrics, 'folds': folds, 'seconds': sum(r['seconds'] for r in folds),
        'completed_at': now(), 'test_scored': False}
    dump(folder/'result.json', result)
    return result


def study_worker(name, group, total_trials):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    training = get_train()
    folder = DEST/'studies'/name/group
    folder.mkdir(parents=True, exist_ok=True)
    sampler_path = folder/'sampler.pkl'
    sampler = pickle.loads(sampler_path.read_bytes()) if sampler_path.exists() else optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10)
    study = optuna.create_study(study_name=f'{name}-{group}', storage=optuna.storages.RDBStorage(
        'sqlite:///'+str(folder/'study.sqlite3'), engine_kwargs={'connect_args': {'timeout':60}}),
        sampler=sampler, direction='minimize', load_if_exists=True)
    if not study.trials:
        baseline = default_config(name)['regression']
        family = name.removeprefix('rom_')
        if family == 'mlp':
            params = {k: baseline[k] for k in ['width','depth','lr','act']}
            params.update(wd_enabled=True, wd_value=baseline['wd'])
        elif family == 'gp':
            params = {k: baseline[k] for k in ['nu','length_scale','signal_level','noise_level']}
        else:
            params = {k: baseline[k] for k in ['max_depth','min_child_weight','learning_rate','n_estimators','subsample','colsample_bytree','reg_lambda']}
            params['reg_alpha_enabled'] = False
        study.enqueue_trial(params)
    with threadpool_limits(limits=1):
        while len(study.get_trials(states=[TrialState.COMPLETE])) < total_trials:
            running = study.get_trials(states=[TrialState.RUNNING])
            if len(running)>1:
                raise RuntimeError('Multiple RUNNING trials in one study')
            trial = optuna.trial.Trial(study, running[0]._trial_id) if running else study.ask()
            config = suggest(trial, name, 'regression')['regression']
            tmp = sampler_path.with_suffix('.tmp'); tmp.write_bytes(pickle.dumps(study.sampler)); tmp.replace(sampler_path)
            try:
                result = evaluate_trial(name, group, trial.number, config, training)
            except Exception as exc:
                dump(folder/f'trial{trial.number:04d}/failure.json', {'error': repr(exc), 'time': now()})
                study.tell(trial, state=TrialState.FAIL)
                raise
            trial.set_user_attr('config', config)
            trial.set_user_attr('metrics', result['metrics'])
            trial.set_user_attr('seconds', result['seconds'])
            study.tell(trial, result['objective'])
            tmp = sampler_path.with_suffix('.tmp'); tmp.write_bytes(pickle.dumps(study.sampler)); tmp.replace(sampler_path)
            study.trials_dataframe().to_csv(folder/'trials.csv', index=False)
            dump(folder/'best.json', {'candidate': name, 'group': group, 'trial': study.best_trial.number,
                'objective': study.best_value, 'config': study.best_trial.user_attrs['config'],
                'metrics': study.best_trial.user_attrs['metrics'],
                'completed_trials': len(study.get_trials(states=[TrialState.COMPLETE])), 'test_scored': False})
            print('INDEPENDENT_TRIAL_DONE', name, group, trial.number, result['objective'], flush=True)
    return name, group, len(study.get_trials(states=[TrialState.COMPLETE]))


def summarize():
    rows = []
    for name in NAMES:
        for group in GROUPS:
            p = DEST/'studies'/name/group/'best.json'
            if p.exists():
                b = json.loads(p.read_text()); r = {k:b[k] for k in ['candidate','group','trial','objective','completed_trials']}
                for t, m in b['metrics'].items():
                    r.update({t+'_'+k:v for k,v in m.items()})
                rows.append(r)
    if rows:
        pd.DataFrame(rows).to_csv(DEST/'summary.csv', index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total-trials', type=int, default=20)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    prepare()
    if args.prepare_only:
        return
    begun = now()
    # Schedule the expensive GP groups first; completed workers pick up fast families.
    names = ['gp','rom_gp','mlp','rom_mlp','xgboost','rom_xgboost']
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(study_worker,n,g,args.total_trials) for n in names for g in GROUPS]
        for f in as_completed(futures):
            print('INDEPENDENT_STUDY_DONE', f.result(), flush=True); summarize()
    rows = summarize()
    dump(DEST/'batch-complete.json', {'started_at': begun, 'finished_at': now(),
        'requested_total_per_study': args.total_trials, 'studies': rows, 'test_scored': False})


if __name__ == '__main__':
    main()
