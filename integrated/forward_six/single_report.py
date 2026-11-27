"""Audit independent studies and compare with frozen joint OOF predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
BASE=Path(__file__).resolve().parents[1]
for folder in [BASE,BASE/'forward',BASE.parent/'src']:
    sys.path.insert(0,str(folder))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import dump, metrics_reg
from runtime_paths import RUNS
from forward_six.single_models import GROUPS,eligible
from forward_six.models import NAMES

DEST=RUNS/'forward-independent/tuning'
JOINT=RUNS/'forward-grouped/tuning'
LABELS={'mlp':'MLP','xgboost':'XGBoost','gp':'GP','rom_mlp':'ROM + MLP','rom_xgboost':'ROM + XGBoost','rom_gp':'ROM + GP'}


def records(folder):
    return sorted([json.loads(p.read_text()) for p in folder.glob('trial*/result.json')],key=lambda r:r['trial'])


def target_fold_score(result,target):
    return float(np.mean([f['metrics'][target]['mae'] for f in result['folds']]))


def audit_report(budget=20):
    protocol=json.loads((DEST/'protocol.json').read_text())
    for p,sha in protocol['data_sha256'].items():
        assert hashlib.sha256((RUNS/p).read_bytes()).hexdigest()==sha,p
    for p,sha in protocol['source_sha256'].items():
        assert hashlib.sha256((BASE.parent/p).read_bytes()).hexdigest()==sha,p
    all_data=pd.read_csv(RUNS/'forward-grouped/data/geometry_targets.csv')
    d=all_data[all_data.role.eq('train')].reset_index(drop=True)
    test=set(all_data.loc[all_data.role.eq('test'),'geom_id'])
    assert not set(d.geom_id)&test
    joint_selected=json.loads((JOINT/'selected_configs_20trials.json').read_text())
    old3=pd.read_csv(RUNS/'forward-six/tuning/comparison_20trials.csv').set_index('candidate')
    comparisons,fold_comparisons,matched,traces,milestones=[],[],[],[],[]
    selected={}; gp_calls=[];selected_gp_calls=[]; caps=selected_caps=0; failures=[]
    for name in NAMES:
        selected[name]={}
        jr=records(JOINT/'studies'/name/'angle_displacement')
        assert len(jr)==20
        joint=next(r for r in jr if r['trial']==joint_selected[name]['angle_displacement']['trial'])
        for group,targets in GROUPS.items():
            target=targets[0];target_column=0 if group=='angle' else 1
            folder=DEST/'studies'/name/group
            rs=records(folder)
            assert len(rs)==budget,(name,group,len(rs))
            with sqlite3.connect(folder/'study.sqlite3') as db:
                assert {r[0] for r in db.execute("select number from trials where state='COMPLETE'")}=={r['trial'] for r in rs}
                assert db.execute("select count(*) from trials where state='RUNNING'").fetchone()[0]==0
            best=None; history=[]
            for index,r in enumerate(rs,1):
                assert r['candidate']==name and r['group']==group and not r['test_scored']
                assert r['targets']==targets
                config=json.loads((folder/f'trial{r["trial"]:04d}/config.json').read_text())
                assert config['config']==r['config']
                scores=[];yy=[];pp=[]
                for fold in range(5):
                    tr=d[d.fold.ne(fold)];held=d[d.fold.eq(fold)&eligible(d,group)]
                    fdir=folder/f'trial{r["trial"]:04d}'
                    meta=json.loads((fdir/f'fold{fold}.json').read_text())
                    assert set(meta['eligible_training_ids'])==set(tr.loc[eligible(tr,group),'geom_id'])
                    assert not set(meta['eligible_training_ids'])&(set(held.geom_id)|test)
                    with np.load(fdir/f'fold{fold}.npz') as z:
                        np.testing.assert_array_equal(z['ids'].astype(str),held.geom_id.to_numpy(str))
                        y,p=z['actual'],z['predicted']
                        np.testing.assert_allclose(y,held[targets].to_numpy(float),rtol=0,atol=0)
                        sd=np.maximum(tr.loc[eligible(tr,group),targets].std(ddof=0).to_numpy(),1e-8)
                        np.testing.assert_allclose(sd,z['training_target_sd'],rtol=0,atol=0)
                        assert y.shape==p.shape and y.shape[1]==1 and np.isfinite(p).all()
                    metric=metrics_reg(y[:,0],p[:,0])
                    for k,v in metric.items():
                        np.testing.assert_allclose(v,meta['metrics'][target][k],rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(metric['mae'],meta['objective'],rtol=1e-12,atol=1e-12)
                    scores.append(metric['mae']);yy.append(y);pp.append(p)
                    gp_calls.extend(meta['diagnostics'].get('kernel_optimization',[]))
                    caps+=meta['diagnostics'].get('epochs')==1200
                objective=float(np.mean(scores))
                np.testing.assert_allclose(objective,r['objective'],rtol=1e-12,atol=1e-12)
                metric=metrics_reg(np.concatenate(yy)[:,0],np.concatenate(pp)[:,0])
                for k,v in metric.items():
                    np.testing.assert_allclose(v,r['metrics'][target][k],rtol=1e-12,atol=1e-12)
                if best is None or objective<best['objective']:
                    best=r
                history.append(best['objective'])
                traces.append({'candidate':name,'group':group,'completed_trial':index,'optuna_trial_number':r['trial'],
                    'phase':'initial' if index<=10 else 'TPE','objective':objective,'best_so_far':best['objective'],'seconds':r['seconds']})
            saved=json.loads((folder/'best.json').read_text())
            assert saved['trial']==best['trial']
            np.testing.assert_allclose(saved['objective'],best['objective'],rtol=0,atol=0)
            selected[name][group]={'trial':best['trial'],'objective':best['objective'],'config':best['config']}
            for f in best['folds']:
                selected_gp_calls.extend(f['diagnostics'].get('kernel_optimization',[]))
                selected_caps+=f['diagnostics'].get('epochs')==1200
            failures.extend(str(p.relative_to(DEST)) for p in folder.glob('trial*/failure.json'))
            milestones.append({'candidate':name,'group':group,'trials':budget,'folds':5,
                **{f'best_at_{n}':history[n-1] for n in sorted(set([1,10,budget-5,budget]))},
                'last5_improvement':history[budget-6]-history[-1],
                'last5_relative_improvement':(history[budget-6]-history[-1])/max(abs(history[budget-6]),1e-12),
                'last_improvement_trial':max(i+1 for i in range(budget) if i==0 or history[i]<history[i-1]),
                'sum_fold_seconds':sum(r['seconds'] for r in rs)})
            # Same-hyperparameter comparisons provide a control before adaptive TPE diverges.
            for count,(a,b) in enumerate(zip(jr[:10],rs[:10]),1):
                assert a['config']==b['config'],(name,group,count)
                matched.append({'candidate':name,'group':group,'target':target,'completed_trial':count,
                    'joint_mae':a['metrics'][target]['mae'],'independent_mae':b['metrics'][target]['mae'],
                    'mae_difference_independent_minus_joint':b['metrics'][target]['mae']-a['metrics'][target]['mae']})
            # Secondary control: retarget configuration selection using the same saved joint studies.
            # Different target-selected joint configs can require TWO joint networks to deploy.
            reselected=min(jr,key=lambda r:target_fold_score(r,target))
            fold_differences=[]
            for fold in range(5):
                jp=JOINT/'studies'/name/'angle_displacement'/f'trial{joint["trial"]:04d}'/f'fold{fold}.npz'
                ip=folder/f'trial{best["trial"]:04d}'/f'fold{fold}.npz'
                with np.load(jp) as a,np.load(ip) as b:
                    np.testing.assert_array_equal(a['ids'],b['ids'])
                    np.testing.assert_allclose(a['actual'][:,target_column],b['actual'][:,0],rtol=0,atol=0)
                    jm=metrics_reg(a['actual'][:,target_column],a['predicted'][:,target_column])
                    im=metrics_reg(b['actual'][:,0],b['predicted'][:,0])
                    diff=im['mae']-jm['mae'];fold_differences.append(diff)
                    fold_comparisons.append({'candidate':name,'target':target,'fold':fold,
                        'joint_mae':jm['mae'],'independent_mae':im['mae'],'mae_difference':diff})
            jmetric=joint['metrics'][target];imetric=best['metrics'][target]
            old_col='angle_mae_deg' if group=='angle' else 'deformation_mae_mm'
            comparisons.append({'candidate':name,'label':LABELS[name],'target':target,'n':imetric['n'],
                'joint_mae':jmetric['mae'],'independent_mae':imetric['mae'],
                'mae_reduction_percent':100*(jmetric['mae']-imetric['mae'])/jmetric['mae'],
                'joint_rmse':jmetric['rmse'],'independent_rmse':imetric['rmse'],
                'joint_r2':jmetric['r2'],'independent_r2':imetric['r2'],
                'independent_better_folds':sum(v<0 for v in fold_differences),
                'joint_selected_trial':joint['trial'],'independent_selected_trial':best['trial'],
                'joint_target_reselected_trial':reselected['trial'],
                'joint_target_reselected_mae':reselected['metrics'][target]['mae'],
                'original_3output_mae':old3.loc[name,old_col]})
    # Verify paired base/ROM initial settings, plus unchanged Train-only ROM calibration.
    for family in ['mlp','xgboost','gp']:
        for group in GROUPS:
            a=records(DEST/'studies'/family/group);b=records(DEST/'studies'/('rom_'+family)/group)
            assert [r['config'] for r in a[:10]]==[r['config'] for r in b[:10]]
    for fold in range(5):
        r=json.loads((DEST/'rom'/f'fold{fold}-provenance.json').read_text())
        ids=set(d.loc[d.fold.ne(fold),'geom_id'])
        assert r['solver_success'] and set(r['training_ids'])==ids
        assert all(k.split('|')[0] in ids for k in r['calibration_curve_keys'])
    compare=pd.DataFrame(comparisons);compare.to_csv(DEST/f'joint_vs_independent_{budget}trials.csv',index=False)
    pd.DataFrame(fold_comparisons).to_csv(DEST/f'fold_comparison_{budget}trials.csv',index=False)
    paired=pd.DataFrame(matched);paired.to_csv(DEST/'matched_configs_first10.csv',index=False)
    paired.groupby(['candidate','target']).agg(mean_mae_difference=('mae_difference_independent_minus_joint','mean'),
        median_mae_difference=('mae_difference_independent_minus_joint','median'),
        independent_better_configs=('mae_difference_independent_minus_joint',lambda x:int((x<0).sum()))).to_csv(DEST/'matched_configs_first10_summary.csv')
    pd.DataFrame(milestones).to_csv(DEST/f'presentation_milestones_{budget}trials.csv',index=False)
    history=pd.DataFrame(traces);history.to_csv(DEST/f'search_history_{budget}trials.csv',index=False)
    dump(DEST/f'selected_configs_{budget}trials.json',selected)
    dump(DEST/f'audit_{budget}trials.json',{'complete_trials':len(traces),'fold_predictions_checked':len(traces)*5,
        'source_data_hashes_verified':True,'same_joint_independent_oof_ids':True,'same_initial_configs':True,
        'all_objectives_recomputed':True,'test_scored':False,'prediction_averaging':False,'operational_failures':failures,
        'gp_inner_calls':len(gp_calls),'gp_inner_not_converged':sum(not r['success'] for r in gp_calls),
        'selected_gp_inner_calls':len(selected_gp_calls),'selected_gp_inner_not_converged':sum(not r['success'] for r in selected_gp_calls),
        'mlp_fold_epoch_caps':caps,'selected_mlp_fold_epoch_caps':selected_caps})
    colors={'mlp':'#2563eb','xgboost':'#d97706','gp':'#059669'}
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    for ax,group,title,unit in zip(axes,GROUPS,['Independent angle','Independent displacement'],['MAE (degrees)','MAE (mm)']):
        for name in NAMES:
            q=history[(history.candidate==name)&(history.group==group)]
            ax.step(q.completed_trial,q.best_so_far,where='post',label=LABELS[name],
                color=colors[name.removeprefix('rom_')],linestyle='--' if name.startswith('rom_') else '-',linewidth=2)
        ax.axvline(10.5,color='#64748b',linestyle=':',linewidth=1)
        ax.set(title=title,xlabel='Completed trials (10 initial + 10 TPE)',ylabel=unit+' (lower is better)',xlim=(1,budget))
        ax.set_xticks([1,5,10,15,budget]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Independent Forward | Train-only common 5-fold | No prediction averaging')
    fig.savefig(DEST/f'search_progress_{budget}trials.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for ax,target,title,unit in zip(axes,['angle_deg','deformation_mm'],['Stable angle','Displacement'],['MAE (degrees)','MAE (mm)']):
        q=compare[compare.target.eq(target)].set_index('candidate').loc[list(NAMES)]
        y=np.arange(len(NAMES))
        ax.barh(y-.18,q.joint_mae,height=.34,label='Joint (20 trials)',color='#94a3b8')
        ax.barh(y+.18,q.independent_mae,height=.34,label='Independent (20 / output)',color='#2563eb')
        ax.set_yticks(y,[LABELS[n] for n in NAMES]);ax.invert_yaxis()
        ax.set(title=title,xlabel=unit+' (lower is better)');ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Joint vs independent | Identical OOF geometries | Separate per-output tuning budgets')
    fig.savefig(DEST/f'joint_vs_independent_{budget}trials.png',dpi=180);plt.close(fig)
    print(compare.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=int,default=20)
    audit_report(p.parse_args().trials)
