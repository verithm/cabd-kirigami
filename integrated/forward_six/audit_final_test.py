"""Independently recompute saved Test metrics and plot the frozen comparison."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import hashlib
import json
import sqlite3
import sys
BASE=Path(__file__).resolve().parents[1]
for p in [BASE,BASE/'forward',BASE.parent/'src']:sys.path.insert(0,str(p))
import numpy as np
import pandas as pd
from common import dump,metrics_cls,metrics_reg
from runtime_paths import RUNS
from forward_six.grouped import eligible,GROUPS,TASKS
from forward_six.models import NAMES
from forward_six.tune import now


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root=RUNS/'forward-final50'
    protocol=json.loads((root/'protocol.json').read_text())
    complete=json.loads((root/'complete.json').read_text())
    assert complete['refits']==24 and complete['test_scored']
    for k in ['data_source_hashes','source_hashes']:
        for p,digest in protocol[k].items():assert sha(p)==digest,p
    manifest=json.loads((root/'manifest.json').read_text())
    # Shell stdout log is still appended after the fit process writes its manifest.
    # Freeze its final hash only in the post-exit artifact_manifest.json below.
    for p,digest in manifest.items():
        if p != 'run.log':assert sha(root/p)==digest,p
    d=pd.read_csv(RUNS/'forward-grouped/data/geometry_targets.csv')
    tr=d[d.role.eq('train')];te=d[d.role.eq('test')]
    assert set(tr.geom_id).isdisjoint(te.geom_id)
    assert tr.geom_id.tolist()==protocol['training_ids']
    assert te.geom_id.tolist()==protocol['test_ids']
    table=pd.read_csv(root/'test_comparison.csv').set_index('candidate')
    details=json.loads((root/'test_metrics.json').read_text())
    audited=0;gp=[];times=[]
    for name in NAMES:
        for task in TASKS:
            parent='forward-six' if task=='classification' else 'forward-grouped'
            with sqlite3.connect(RUNS/parent/'tuning/studies'/name/task/'study.sqlite3') as db:
                states=dict(db.execute('select state,count(*) from trials group by state'))
            assert states.get('COMPLETE',0)==50 and states.get('RUNNING',0)==0
            fit=json.loads((root/'models'/name/task/'fit.json').read_text())
            assert fit['test_rows_used']==0 and fit['reload_prediction_verified']
            assert fit['model_sha256']==sha(root/'models'/name/task/'model.pkl')
            assert fit['protocol_sha256']==sha(root/'protocol.json')
            expected_tr=tr[eligible(tr,task)]
            assert set(fit['training_ids'])==set(expected_tr.geom_id)
            assert not set(fit['training_ids']) & set(te.geom_id)
            choice=protocol['selected'][name][task]
            assert fit['config']==choice['fit_config']
            if name.removeprefix('rom_')=='mlp':
                assert fit['diagnostics']['epochs']==int(np.median(choice['cv_fold_epochs']))
            for records in fit['diagnostics'].get('optimization',[]):
                for rec in records:gp.append({'candidate':name,'task':task,**rec})
            pred=pd.read_csv(root/'predictions'/name/(task+'.csv'))
            held=te[eligible(te,task)]
            assert pred.geom_id.tolist()==held.geom_id.tolist()
            if task=='classification':
                np.testing.assert_array_equal(pred.actual.to_numpy(str),held.label.to_numpy(str))
                np.testing.assert_allclose(pred.filter(like='probability_').sum(axis=1),1,atol=1e-6)
                metrics=metrics_cls(pred.actual.eq('bi>mono'),pred.predicted.eq('bi>mono'))
                for k,v in metrics.items():
                    np.testing.assert_allclose(v,details[name][task]['positive_bi_to_mono'][k],rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(v,table.loc[name,k],rtol=1e-12,atol=1e-12)
            else:
                for t in GROUPS[task]:
                    np.testing.assert_allclose(pred['actual_'+t],held[t],rtol=1e-12,atol=1e-12)
                targets=GROUPS[task]+(['barrier_mJ'] if task=='barrier' else [])
                for t in targets:
                    metric=metrics_reg(pred['actual_'+t],pred['predicted_'+t])
                    for k,v in metric.items():
                        np.testing.assert_allclose(v,details[name][task][t][k],rtol=1e-11,atol=1e-12)
                        np.testing.assert_allclose(v,table.loc[name,t+'_'+k],rtol=1e-11,atol=1e-12)
            times.append({'candidate':name,'task':task,'seconds':fit['seconds'],'eligible_train_count':len(expected_tr),
                          'test_count':len(held),'epochs':fit['diagnostics'].get('epochs')})
            audited+=1
    rom=json.loads((root/'rom_full_train.json').read_text())
    assert rom['solver_success'] and set(rom['training_ids'])==set(tr.geom_id)
    assert all(key.split('|')[0] in set(tr.geom_id) for key in rom['calibration_curve_keys'])
    pd.DataFrame(times).to_csv(root/'fit_times.csv',index=False)
    audit={'finished_at':now(),'models_tasks_audited':audited,'saved_predictions_and_metrics_recomputed':True,
        'train_test_disjoint':True,'frozen_source_and_data_hashes_verified':True,
        'all_studies_still_exactly_50':True,'eligible_train_and_test_ids_verified':True,
        'all_checkpoint_reload_predictions_verified':True,'rom_train_only_calibration_verified':True,
        'gp_inner_optimizer_calls':len(gp),'gp_inner_calls_not_successful':sum(not x['success'] for x in gp),
        'gp_optimization_records':gp,'no_additional_tuning':True}
    dump(root/'audit.json',audit)
    # All labels/axes in English for portability on other team machines.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'mlp':'MLP','xgboost':'XGBoost','gp':'GP','rom_mlp':'ROM+MLP','rom_xgboost':'ROM+XGBoost','rom_gp':'ROM+GP'}
    panels=[('f1','Bi-to-mono classification','F1 (higher is better)'),
        ('angle_deg_mae','Stable angle','MAE (degrees)'),('deformation_mm_mae','Displacement','MAE (mm)'),
        ('log10_barrier_mae','Energy barrier: learning scale','MAE (log10 mJ)'),
        ('barrier_mJ_mae','Energy barrier: original scale','MAE (mJ)'),
        ('hinge_transition_max_MPa_mae','Hinge maximum stress','MAE (MPa)')]
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    colors=['#2878B5','#D27B39','#6B9C55','#7555A2','#A95561','#429F9A']
    for ax,(key,title,unit) in zip(axes.flat,panels):
        values=table.loc[list(NAMES),key].to_numpy(float)
        ax.bar(np.arange(6),values,color=colors)
        for x,v in enumerate(values):ax.text(x,v,f'{v:.4f}',ha='center',va='bottom',fontsize=9)
        ax.set_xticks(np.arange(6),[labels[n] for n in NAMES],rotation=35,ha='right')
        ax.set_title(title);ax.set_ylabel(unit);ax.set_ylim(0,max(values)*1.2)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Fixed Test | Full-Train refit after 50 trials per model/task\nClassification n=105; each regression target n=143 | One model per task; no prediction averaging',fontsize=14)
    fig.savefig(root/'test_comparison.png',dpi=170);plt.close(fig)
    cv=pd.read_csv(RUNS/'forward-refinement/comparison_50trials.csv').set_index('candidate')
    metric_cols=[p[0] for p in panels]
    out=[]
    for n in NAMES:
        for col in metric_cols:
            out.append({'candidate':n,'metric':col,'cv_oof':cv.loc[n,col],'test':table.loc[n,col],
                        'test_minus_cv':table.loc[n,col]-cv.loc[n,col]})
    pd.DataFrame(out).to_csv(root/'cv_vs_test.csv',index=False)
    dump(root/'artifact_manifest.json',{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})
    print(json.dumps(audit,indent=2));print(table[[p[0] for p in panels]].to_string())


if __name__=='__main__':main()
