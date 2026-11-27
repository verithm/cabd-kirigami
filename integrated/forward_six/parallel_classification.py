"""Parallelize independent classification folds without changing fit/scoring code.

The immutable original evaluator aggregates exactly the same fold caches. Each
trial still waits for all five folds before Optuna proposes the next setting.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import sys,time,warnings
BASE=Path(__file__).resolve().parents[1]
for folder in [BASE,BASE/'forward',BASE.parent/'src']:
    sys.path.insert(0,str(folder))
import numpy as np
from threadpoolctl import threadpool_limits
from forward_six import tune as tuning
from forward_six.models import NAMES
_NATIVE_EVALUATE=tuning.evaluate_trial
_FOLD_POOL=None


def fit_fold(name,config,training,fold,destination,curves,rom):
    folder=Path(destination);meta=folder/f'fold{fold}.json';pred=folder/f'fold{fold}.npz'
    if meta.exists() and pred.exists():return fold
    held=training[training.fold.eq(fold)&training.label.notna()].reset_index(drop=True)
    fit_rows=training[training.fold.ne(fold)]
    begun=time.perf_counter()
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model=tuning.fit_candidate(name,fit_rows,config,tuning.SEED,task='classification',
            curves_path=Path(curves),rom_cache=Path(rom))
        labels,probabilities=tuning.predict_classification(model,held)
        actual=held.label.eq('bi>mono').to_numpy();predicted=labels=='bi>mono'
        metric=tuning.metrics_cls(actual,predicted)
    record={'fold':fold,'n_train_geometries':len(fit_rows),'n_validation':len(held),
        'metrics':metric,'diagnostics':tuning.fitted_diagnostics(model,'classification'),
        'warnings':sorted(set(str(w.message) for w in caught)),
        'seconds':time.perf_counter()-begun,'finished_at':tuning.now()}
    tmp=pred.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,ids=held.geom_id.to_numpy(str),actual=actual,predicted=predicted,probabilities=probabilities)
    tmp.replace(pred);tuning.dump(meta,record)
    print('PARALLEL_CLASS_FOLD_DONE',name,folder.name,fold,round(record['seconds'],2),flush=True)
    return fold


def evaluate_parallel(name,task,number,config,training,fold_workers=3):
    if task!='classification' or name.removeprefix('rom_')!='gp':
        return _NATIVE_EVALUATE(name,task,number,config,training)
    folder=tuning.DEST/'studies'/name/task/f'trial{number:04d}'
    identity={'candidate':name,'task':task,'trial':number,'config':config,'seed':tuning.SEED}
    p=folder/'config.json'
    if p.exists():
        import json
        assert json.loads(p.read_text())==identity
    else:tuning.dump(p,identity)
    owned=_FOLD_POOL is None
    pool=ProcessPoolExecutor(max_workers=fold_workers) if owned else _FOLD_POOL
    try:
        futures=[pool.submit(fit_fold,name,config,training,f,str(folder),
            str(tuning.RUNS/'unified-forward/data/completed_force_curves.json'),str(tuning.DEST/'rom'))
            for f in range(5) if not ((folder/f'fold{f}.json').exists() and (folder/f'fold{f}.npz').exists())]
        for future in as_completed(futures):future.result()
    finally:
        if owned:pool.shutdown()
    # This is the original, unchanged aggregation and objective calculation.
    return _NATIVE_EVALUATE(name,task,number,config,training)


def classification_worker(name,total_trials,fold_workers=3):
    global _FOLD_POOL
    if name.removeprefix('rom_')!='gp':
        return tuning.study_worker(name,'classification',total_trials)
    original=tuning.evaluate_trial
    with ProcessPoolExecutor(max_workers=fold_workers) as pool:
        _FOLD_POOL=pool;tuning.evaluate_trial=evaluate_parallel
        try:return tuning.study_worker(name,'classification',total_trials)
        finally:tuning.evaluate_trial=original;_FOLD_POOL=None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=NAMES,required=True)
    p.add_argument('--total-trials',type=int,required=True)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--fold-workers',type=int,default=3)
    args=p.parse_args();tuning.prepare()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs=[ex.submit(classification_worker,n,args.total_trials,args.fold_workers) for n in dict.fromkeys(args.models)]
        for f in as_completed(fs):
            print('PARALLEL_CLASS_STUDY_DONE',f.result(),flush=True);tuning.summarize()


if __name__=='__main__':main()
