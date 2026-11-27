"""Inference for the selected four-task, five-seed MLP ensemble."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys
import json
import hashlib
import pickle
import argparse
BASE=Path(__file__).resolve().parents[1]
for p in [BASE,BASE/'forward',BASE.parent/'src']:sys.path.insert(0,str(p))
import numpy as np
import pandas as pd
from common import FEATURES,CLASSES
from runtime_paths import RUNS
from forward_six.grouped import TASKS,GROUPS,predict_group

DEFAULT_MODEL=RUNS/'forward-mlp-ensemble'


def soft_vote(predictions):
    """Average class probabilities, then apply the existing three-class argmax."""
    p=np.asarray(predictions,float)
    if p.ndim!=3 or p.shape[0]<2 or p.shape[2]!=len(CLASSES):
        raise ValueError('Expected members x rows x three classes')
    if not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid member probabilities')
    np.testing.assert_allclose(p.sum(axis=2),1,atol=1e-6)
    mean=p.mean(axis=0)
    return np.asarray(CLASSES)[mean.argmax(axis=1)],mean,p.std(axis=0,ddof=1)


def mean_regression(task,predictions,index):
    if task not in GROUPS:raise ValueError(task)
    p=np.asarray(predictions,float)
    if p.ndim!=3 or p.shape[0]<2 or p.shape[2]!=len(GROUPS[task]) or not np.isfinite(p).all():
        raise ValueError('Invalid member regression predictions')
    mean=p.mean(axis=0);sd=p.std(axis=0,ddof=1)
    result=pd.DataFrame(mean,index=index,columns=GROUPS[task])
    for j,t in enumerate(GROUPS[task]):result[t+'_member_sd']=sd[:,j]
    if task=='barrier':
        # Fix aggregation in the learned log scale before any Test inspection.
        result['barrier_mJ']=10.**mean[:,0]
        result['barrier_mJ_member_sd']=(10.**p[:,:,0]).std(axis=0,ddof=1)
    if not np.isfinite(result.to_numpy(float)).all():raise FloatingPointError('Non-finite ensemble prediction')
    return result


def load_ensemble(folder=DEFAULT_MODEL):
    folder=Path(folder)
    selection=json.loads((folder/'forward_selection.json').read_text())
    path=folder/selection['bundle']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==selection['bundle_sha256']
    with path.open('rb') as h:bundle=pickle.load(h)
    if bundle['schema_version']!='mlp-seed-ensemble-v1':raise ValueError('Unknown ensemble schema')
    assert len(set(bundle['seeds']))==len(bundle['seeds'])>=2
    for task in TASKS:
        assert len(bundle['members'][task])==len(bundle['seeds'])
        for seed,model in zip(bundle['seeds'],bundle['members'][task]):
            core=model['model'] if task=='classification' else model
            assert core['candidate']=='mlp' and core['seed']==seed and model['group']==task
    return bundle


def predict_task(bundle,data,task):
    models=bundle['members'][task]
    if task=='classification':
        p=[predict_group(model,data)[1] for model in models]
        return soft_vote(p)
    p=[predict_group(model,data)[GROUPS[task]].to_numpy(float) for model in models]
    return mean_regression(task,p,data.index)


def predict(bundle,data):
    missing=set(FEATURES)-set(data.columns)
    if missing:raise ValueError(f'Missing geometry columns: {sorted(missing)}')
    if not np.isfinite(data[FEATURES].to_numpy(float)).all():raise ValueError('Geometry must be finite')
    out=data[['geom_id']].copy() if 'geom_id' in data else pd.DataFrame(index=data.index)
    labels,p,sd=predict_task(bundle,data,'classification')
    out['predicted_class']=labels
    for j,c in enumerate(CLASSES):out['probability_'+c]=p[:,j]
    out['p_bi_mono']=p[:,1];out['p_bi_mono_member_sd']=sd[:,1]
    for task in GROUPS:
        q=predict_task(bundle,data,task)
        for col in q:out[col]=q[col].to_numpy()
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--model-dir',type=Path,default=DEFAULT_MODEL)
    args=parser.parse_args()
    bundle=load_ensemble(args.model_dir)
    result=predict(bundle,pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False)
    print(args.output)


if __name__=='__main__':main()
