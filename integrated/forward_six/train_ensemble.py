"""Train the selected four-task MLP as a fixed five-initialization ensemble."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import hashlib,json,pickle,sys,time,warnings
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
for p in [BASE,BASE/'forward',BASE.parent/'src']:sys.path.insert(0,str(p))
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report
from threadpoolctl import threadpool_limits
from common import dump,metrics_cls,metrics_reg,CLASSES
from runtime_paths import RUNS
from forward_six.grouped import TASKS,GROUPS,eligible,fit_group,predict_group
from forward_six.ensemble import soft_vote,mean_regression,predict_task,predict,load_ensemble
from forward_six.tune import now

DEST=RUNS/'forward-mlp-ensemble'
DATA=RUNS/'forward-grouped/data/geometry_targets.csv'
FINAL=RUNS/'forward-final50'
SEEDS=[42,43,44,45,46]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def network(model,task):
    return model['model']['classification']['network'] if task=='classification' else model['estimator']['network']


def weight_hash(model,task):
    digest=hashlib.sha256()
    for key,value in network(model,task).state_dict().items():
        digest.update(key.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def freeze():
    DEST.mkdir(parents=True,exist_ok=True)
    old=json.loads((FINAL/'protocol.json').read_text())
    configs={t:deepcopy(old['selected']['mlp'][t]['fit_config']) for t in TASKS}
    files=[DATA,FINAL/'protocol.json',RUNS/'forward-refinement/selected_configs_50trials.json',
           RUNS/'unified-forward/data/folds.csv']
    sources=[Path(__file__),BASE/'forward_six/ensemble.py',BASE/'forward_six/models.py',
             BASE/'forward_six/grouped.py',BASE/'forward/common.py',BASE.parent/'src/cabd/models/mlp.py']
    d=pd.read_csv(DATA);tr=d[d.role.eq('train')];te=d[d.role.eq('test')]
    assert d.geom_id.is_unique and set(tr.geom_id).isdisjoint(te.geom_id)
    assert tr.geom_id.tolist()==old['training_ids'] and te.geom_id.tolist()==old['test_ids']
    p={'schema_version':'mlp-seed-ensemble-v1','seeds':SEEDS,'members_per_task':5,'tasks':list(TASKS),
        'configs':configs,'bootstrap':False,'training':'20 fresh networks; all eligible Train rows, differing initialization seeds',
        'epochs':'same frozen CV-median epochs as single final MLP; no per-seed or Test-based epoch selection',
        'classification_aggregation':'equal average of 3-class probabilities then argmax',
        'regression_aggregation':'equal average of member predicted means in learned response units',
        'barrier_aggregation':'mean of log10 predictions, then 10**mean_log10 for mJ point prediction',
        'member_sd':'sample SD across member means, not a calibrated confidence interval or failure probability',
        'ensemble_size_or_weights_selected_on_test':False,'test_policy':'same fixed Test for descriptive comparison only; no reweighting or seed selection',
        'train_ids':tr.geom_id.tolist(),'test_ids':te.geom_id.tolist(),
        'population':{r:{'geometries':len(q),**{t:int(eligible(q,t).sum()) for t in TASKS}}
                      for r,q in [('train',tr),('test',te)]},
        'data_sha256':{str(p):sha(p) for p in files},'source_sha256':{str(p):sha(p) for p in sources},
        'inverse_automatically_changed':False}
    path=DEST/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p,'Frozen ensemble experiment changed'
    else:dump(path,p);dump(DEST/'frozen_at.json',{'time':now(),'protocol_sha256':sha(path),'test_scored':False})
    return p


def worker(task,seed):
    folder=DEST/'members'/task/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    fitpath=folder/'fit.json';path=folder/'model.pkl'
    if fitpath.exists():
        record=json.loads(fitpath.read_text())
        assert record['model_sha256']==sha(path) and record['protocol_sha256']==sha(DEST/'protocol.json')
        return task,seed,'cached'
    protocol=json.loads((DEST/'protocol.json').read_text())
    d=pd.read_csv(DATA);tr=d[d.role.eq('train')].reset_index(drop=True)
    assert tr.geom_id.tolist()==protocol['train_ids']
    begun=time.perf_counter()
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model=fit_group('mlp',tr,task,protocol['configs'][task],seed)
        core=model['model'] if task=='classification' else model
        assert core['seed']==seed and set(core['training_ids'])==set(tr.geom_id)
        assert not set(core['training_ids'])&set(protocol['test_ids'])
        if task!='classification':assert set(core['eligible_training_ids'])==set(tr.loc[eligible(tr,task),'geom_id'])
        est=core['classification'] if task=='classification' else core['estimator']
        assert est['epochs']==protocol['configs'][task]['epochs']
        probe=tr[eligible(tr,task)].head(8)
        before=predict_group(model,probe)
        tmp=path.with_suffix('.tmp.pkl')
        with tmp.open('wb') as h:pickle.dump(model,h,pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        with path.open('rb') as h:restored=pickle.load(h)
        after=predict_group(restored,probe)
        if task=='classification':
            np.testing.assert_array_equal(before[0],after[0]);np.testing.assert_allclose(before[1],after[1],rtol=0,atol=0)
        else:np.testing.assert_allclose(before,after,rtol=0,atol=0)
        replicated_baseline=None
        if seed==42:
            with (FINAL/'models/mlp'/task/'model.pkl').open('rb') as h:baseline=pickle.load(h)
            replicated_baseline=weight_hash(model,task)==weight_hash(baseline,task)
            assert replicated_baseline,'Same seed/config must reproduce baseline weights'
    dump(fitpath,{'task':task,'seed':seed,'config':protocol['configs'][task],
        'epochs':est['epochs'],'eligible_train_ids':tr.loc[eligible(tr,task),'geom_id'].tolist(),
        'eligible_train_count':int(eligible(tr,task).sum()),'test_rows_used':0,
        'fresh_initialization':True,'model_sha256':sha(path),'network_weight_sha256':weight_hash(model,task),
        'protocol_sha256':sha(DEST/'protocol.json'),'reload_prediction_verified':True,
        'seed42_reproduces_single_baseline':replicated_baseline,'seconds':time.perf_counter()-begun,
        'warnings':sorted(set(str(w.message) for w in caught)),'finished_at':now()})
    print('MEMBER_FIT_DONE',task,seed,round(time.perf_counter()-begun,2),flush=True)
    return task,seed,'trained'


def bundle_members(protocol):
    members={};fits=[]
    for task in TASKS:
        members[task]=[];hashes=[]
        for seed in SEEDS:
            folder=DEST/'members'/task/f'seed{seed}'
            fit=json.loads((folder/'fit.json').read_text())
            assert fit['model_sha256']==sha(folder/'model.pkl')
            assert fit['seed']==seed and fit['test_rows_used']==0
            with (folder/'model.pkl').open('rb') as h:members[task].append(pickle.load(h))
            hashes.append(fit['network_weight_sha256']);fits.append(fit)
        assert len(set(hashes))==5,'Ensemble members must have distinct network weights'
    result={'schema_version':protocol['schema_version'],'seeds':SEEDS,'members':members,
            'configs':protocol['configs'],'aggregation':{k:v for k,v in protocol.items() if 'aggregation' in k}}
    path=DEST/'bundle.pkl';tmp=path.with_suffix('.tmp.pkl')
    with tmp.open('wb') as h:pickle.dump(result,h,pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    dump(DEST/'forward_selection.json',{'schema_version':protocol['schema_version'],
        'selected_model':'MLP five-seed ensemble','bundle':'bundle.pkl','bundle_sha256':sha(path),
        'seeds':SEEDS,'tasks':list(TASKS),'members_per_task':5,'total_networks':20,
        'selection_basis':'User-selected MLP family from 50-trial CV comparison; fixed equal ensemble before new Test evaluation',
        'configs':protocol['configs'],'inference_module':'forward_six.ensemble',
        'stress_unit':'MPa','barrier_unit':'mJ','regression_domain':'partial-bistable geometries; predicted values for other classes are not validated targets',
        'probability_calibration_verified':False,'inverse_automatically_changed':False})
    return result,fits


def score(task,held,pred):
    if task=='classification':
        labels,p,_=pred
        y=held.label.to_numpy(str)
        return {'positive_bi_to_mono':metrics_cls(y=='bi>mono',labels=='bi>mono'),
            'class_order':CLASSES,'confusion_matrix':confusion_matrix(y,labels,labels=CLASSES).tolist(),
            'three_class':classification_report(y,labels,labels=CLASSES,output_dict=True,zero_division=0)}
    out={t:metrics_reg(held[t],pred[t]) for t in GROUPS[task]}
    if task=='barrier':out['barrier_mJ']=metrics_reg(10.**held.log10_barrier,pred.barrier_mJ)
    return out


def evaluate(bundle,protocol,fits):
    # All sizes, weights, configs and checkpoints exist before this marker.
    dump(DEST/'test_evaluation_started.json',{'time':now(),'completed_full_train_networks':20,
         'protocol_sha256':sha(DEST/'protocol.json'),'bundle_sha256':sha(DEST/'bundle.pkl')})
    d=pd.read_csv(DATA);te=d[d.role.eq('test')].reset_index(drop=True)
    assert te.geom_id.tolist()==protocol['test_ids']
    report={};member_rows=[];row={'candidate':'mlp_ensemble5'}
    with threadpool_limits(limits=1):
        for task in TASKS:
            held=te[eligible(te,task)].reset_index(drop=True)
            pred=predict_task(bundle,held,task);metric=score(task,held,pred);report[task]=metric
            saved=held[['geom_id']].copy()
            if task=='classification':
                labels,p,sd=pred;saved['actual']=held.label;saved['predicted']=labels
                for j,c in enumerate(CLASSES):saved['probability_'+c]=p[:,j];saved['probability_'+c+'_member_sd']=sd[:,j]
                row.update(metric['positive_bi_to_mono'])
            else:
                for t in GROUPS[task]:saved['actual_'+t]=held[t]
                if task=='barrier':saved['actual_barrier_mJ']=10.**held.log10_barrier
                for col in pred:saved['predicted_'+col]=pred[col].to_numpy()
                for t,m in metric.items():row.update({t+'_'+k:v for k,v in m.items()})
            path=DEST/'predictions'/(task+'.csv');path.parent.mkdir(exist_ok=True);saved.to_csv(path,index=False)
            for seed,member in zip(SEEDS,bundle['members'][task]):
                member_pred=predict_group(member,held)
                if task=='classification':
                    m=metrics_cls(held.label.eq('bi>mono'),member_pred[0]=='bi>mono')
                    member_rows.append({'task':task,'seed':seed,'target':'bi>mono',**m})
                else:
                    for t in GROUPS[task]:member_rows.append({'task':task,'seed':seed,'target':t,**metrics_reg(held[t],member_pred[t])})
                    if task=='barrier':member_rows.append({'task':task,'seed':seed,'target':'barrier_mJ',**metrics_reg(10.**held.log10_barrier,10.**member_pred.log10_barrier)})
        # Public inference returns every output on every geometry, with domain caveat.
        full=predict(bundle,te);full.to_csv(DEST/'predictions/all_test_geometries.csv',index=False)
        loaded=load_ensemble(DEST)
        np.testing.assert_allclose(predict(loaded,te.head(8)).select_dtypes('number'),
                                   predict(bundle,te.head(8)).select_dtypes('number'),rtol=0,atol=0)
    dump(DEST/'test_metrics.json',report)
    pd.DataFrame(member_rows).to_csv(DEST/'member_test_metrics.csv',index=False)
    baseline=pd.read_csv(FINAL/'test_comparison.csv');baseline=baseline[baseline.candidate.eq('mlp')].iloc[0].to_dict()
    baseline['candidate']='mlp_single_seed42'
    pd.DataFrame([baseline,row]).to_csv(DEST/'single_vs_ensemble_test.csv',index=False)
    dump(DEST/'complete.json',{'time':now(),'full_train_networks':20,'members_per_task':5,'seeds':SEEDS,
        'test_scored':True,'ensemble_size_or_weights_selected_on_test':False,
        'baseline_seed42_weights_reproduced':all(r['seed42_reproduces_single_baseline'] for r in fits if r['seed']==42),
        'distinct_weights_per_task':5,'bundle_reload_predictions_verified':True,
        'population':protocol['population'],'total_fit_seconds':sum(r['seconds'] for r in fits),
        'inverse_automatically_changed':False,'git_pushed':False})
    print('ENSEMBLE_COMPLETE',json.dumps(row),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();protocol=freeze()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(worker,t,s) for t in TASKS for s in SEEDS]
        for future in as_completed(fs):print('MEMBER_READY',future.result(),flush=True)
    bundle,fits=bundle_members(protocol)
    evaluate(bundle,protocol,fits)


if __name__=='__main__':main()
