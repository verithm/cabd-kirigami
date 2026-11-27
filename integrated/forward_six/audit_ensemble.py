"""Recompute ensemble Test predictions from all members and verify saved metrics."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import hashlib,json,sqlite3,sys
BASE=Path(__file__).resolve().parents[1]
for p in [BASE,BASE/'forward',BASE.parent/'src']:sys.path.insert(0,str(p))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common import metrics_cls,metrics_reg,dump,CLASSES
from runtime_paths import RUNS
from forward_six.grouped import TASKS,GROUPS,eligible,predict_group
from forward_six.ensemble import load_ensemble,predict
from forward_six.train_ensemble import weight_hash
from forward_six.tune import now
ROOT=RUNS/'forward-mlp-ensemble'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for field in ['data_sha256','source_sha256']:
        for p,digest in protocol[field].items():assert sha(p)==digest,p
    bundle=load_ensemble(ROOT)
    d=pd.read_csv(RUNS/'forward-grouped/data/geometry_targets.csv')
    tr=d[d.role.eq('train')];te=d[d.role.eq('test')].reset_index(drop=True)
    assert tr.geom_id.tolist()==protocol['train_ids'] and te.geom_id.tolist()==protocol['test_ids']
    assert set(tr.geom_id).isdisjoint(te.geom_id)
    metrics=json.loads((ROOT/'test_metrics.json').read_text())
    table=pd.read_csv(ROOT/'single_vs_ensemble_test.csv').set_index('candidate')
    row=table.loc['mlp_ensemble5'];count=0
    with threadpool_limits(limits=1):
        for task in TASKS:
            held=te[eligible(te,task)].reset_index(drop=True)
            saved=pd.read_csv(ROOT/'predictions'/(task+'.csv'))
            assert saved.geom_id.tolist()==held.geom_id.tolist()
            raw=[];hashes=[]
            for seed,model in zip(bundle['seeds'],bundle['members'][task]):
                meta=json.loads((ROOT/'members'/task/f'seed{seed}'/'fit.json').read_text())
                assert meta['test_rows_used']==0 and meta['reload_prediction_verified']
                assert set(meta['eligible_train_ids'])==set(tr.loc[eligible(tr,task),'geom_id'])
                assert meta['config']==protocol['configs'][task] and meta['epochs']==protocol['configs'][task]['epochs']
                assert meta['model_sha256']==sha(ROOT/'members'/task/f'seed{seed}'/'model.pkl')
                assert meta['network_weight_sha256']==weight_hash(model,task)
                if seed==42:assert meta['seed42_reproduces_single_baseline']
                hashes.append(meta['network_weight_sha256'])
                q=predict_group(model,held);raw.append(q[1] if task=='classification' else q[GROUPS[task]].to_numpy(float));count+=1
            assert len(set(hashes))==5
            # Same declared float64 aggregation precision as the public API;
            # member Torch predictions themselves are float32.
            raw=np.asarray(raw,dtype=np.float64)
            mean=np.mean(raw,axis=0);sd=np.std(raw,axis=0,ddof=1)
            if task=='classification':
                labels=np.asarray(CLASSES)[mean.argmax(axis=1)]
                np.testing.assert_array_equal(saved.actual,held.label)
                np.testing.assert_array_equal(saved.predicted,labels)
                for j,c in enumerate(CLASSES):
                    np.testing.assert_allclose(saved['probability_'+c],mean[:,j],rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(saved['probability_'+c+'_member_sd'],sd[:,j],rtol=1e-12,atol=1e-12)
                recomputed=metrics_cls(held.label.eq('bi>mono'),labels=='bi>mono')
                for k,v in recomputed.items():
                    np.testing.assert_allclose(v,metrics[task]['positive_bi_to_mono'][k],rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(v,row[k],rtol=1e-12,atol=1e-12)
            else:
                for j,t in enumerate(GROUPS[task]):
                    np.testing.assert_allclose(saved['actual_'+t],held[t],rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(saved['predicted_'+t],mean[:,j],rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(saved['predicted_'+t+'_member_sd'],sd[:,j],rtol=1e-12,atol=1e-12)
                values={t:mean[:,j] for j,t in enumerate(GROUPS[task])}
                if task=='barrier':
                    values['barrier_mJ']=10.**mean[:,0]
                    np.testing.assert_allclose(saved.predicted_barrier_mJ,values['barrier_mJ'],rtol=1e-12,atol=1e-12)
                for t,pred in values.items():
                    actual=10.**held.log10_barrier if t=='barrier_mJ' else held[t]
                    recomputed=metrics_reg(actual,pred)
                    for k,v in recomputed.items():
                        np.testing.assert_allclose(v,metrics[task][t][k],rtol=1e-12,atol=1e-12)
                        np.testing.assert_allclose(v,row[t+'_'+k],rtol=1e-12,atol=1e-12)
        all_saved=pd.read_csv(ROOT/'predictions/all_test_geometries.csv')
        all_pred=predict(bundle,te)
        np.testing.assert_array_equal(all_saved.geom_id,all_pred.geom_id)
        np.testing.assert_array_equal(all_saved.predicted_class,all_pred.predicted_class)
        np.testing.assert_allclose(all_saved.select_dtypes('number'),all_pred.select_dtypes('number'),rtol=1e-12,atol=1e-12)
    oldroot=RUNS/'forward-final50';old=json.loads((oldroot/'artifact_manifest.json').read_text())
    for relative,digest in old.items():assert sha(oldroot/relative)==digest,relative
    for name in ['mlp','xgboost','gp','rom_mlp','rom_xgboost','rom_gp']:
        for task in TASKS:
            parent='forward-six' if task=='classification' else 'forward-grouped'
            with sqlite3.connect(RUNS/parent/'tuning/studies'/name/task/'study.sqlite3') as db:
                states=dict(db.execute('select state,count(*) from trials group by state'))
            assert states.get('COMPLETE')==50 and states.get('RUNNING',0)==0
    fields={'f1':'F1','angle_deg_mae':'degrees','deformation_mm_mae':'mm',
        'log10_barrier_mae':'log10 mJ','barrier_mJ_mae':'mJ','hinge_transition_max_MPa_mae':'MPa'}
    changes=[]
    for field,unit in fields.items():
        single=float(table.loc['mlp_single_seed42',field]);ens=float(row[field])
        changes.append({'metric':field,'unit':unit,'single':single,'ensemble5':ens,
            'improvement':ens-single if field=='f1' else single-ens,
            'F1_increase_percentage_points':100*(ens-single) if field=='f1' else None,
            'MAE_reduction_percent':100*(single-ens)/single if field!='f1' else None})
    pd.DataFrame(changes).to_csv(ROOT/'improvements.csv',index=False)
    audit={'time':now(),'networks_audited':count,'unique_weight_hashes_per_task':5,
        'all_20_full_train_members_and_Train_ids_verified':True,'train_test_disjoint':True,
        'source_data_hashes_verified':True,'ensemble_aggregation_independently_recomputed':True,
        'all_test_metrics_recomputed':True,'seed42_baseline_replication_verified':True,
        'public_inference_all_Test_rows_verified':True,'old_single_model_artifacts_unchanged':True,
        'all_original_studies_still_exactly_50':True,'test_used_for_size_weights_or_seed_selection':False}
    dump(ROOT/'audit.json',audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields_titles=[('f1','Stability F1 (higher is better)'),('angle_deg_mae','Angle MAE (degrees)'),
        ('deformation_mm_mae','Displacement MAE (mm)'),('log10_barrier_mae','Barrier MAE (log10 mJ)'),
        ('barrier_mJ_mae','Barrier MAE (mJ)'),('hinge_transition_max_MPa_mae','Hinge stress MAE (MPa)')]
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,(field,title) in zip(axes.flat,fields_titles):
        values=table.loc[['mlp_single_seed42','mlp_ensemble5'],field].to_numpy(float)
        ax.bar([0,1],values,color=['#94A3B8','#2878B5'],width=.55)
        for j,v in enumerate(values):ax.text(j,v,f'{v:.4f}',ha='center',va='bottom')
        ax.set_xticks([0,1],['Single MLP','MLP ensemble (5)']);ax.set_title(title,fontsize=11)
        if field=='f1':
            ax.set_ylim(0,1.08);ax.set_yticks(np.linspace(0,1,6))
        else:ax.set_ylim(0,max(values)*1.23)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Final Forward | Fixed five-seed MLP ensemble\nSame full Train and fixed Test | No Test-based member selection',fontsize=14)
    fig.savefig(ROOT/'single_vs_ensemble.png',dpi=180);fig.savefig(ROOT/'single_vs_ensemble.pdf');plt.close(fig)
    dump(ROOT/'manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='manifest.json'})
    print(pd.DataFrame(changes).to_string(index=False));print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
