"""Audit grouped TPE studies and compare frozen three-response regression OOF."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

BASE = Path(__file__).resolve().parents[1]
for folder in [BASE, BASE/'forward', BASE.parent/'src']:
    sys.path.insert(0, str(folder))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import dump, metrics_reg
from runtime_paths import RUNS
from forward_six.grouped import GROUPS, eligible
from forward_six.models import NAMES

DEST = RUNS/'forward-grouped/tuning'
OLD = RUNS/'forward-six/tuning'
LABELS = {'mlp':'MLP','xgboost':'XGBoost','gp':'GP','rom_mlp':'ROM + MLP',
          'rom_xgboost':'ROM + XGBoost','rom_gp':'ROM + GP'}


def audit_and_report(budget):
    protocol = json.loads((DEST/'protocol.json').read_text())
    for path, digest in protocol['data_sha256'].items():
        assert hashlib.sha256((RUNS/path).read_bytes()).hexdigest()==digest, path
    for path, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((BASE.parent/path).read_bytes()).hexdigest()==digest, path
    all_data = pd.read_csv(RUNS/'forward-grouped/data/geometry_targets.csv')
    d = all_data[all_data.role.eq('train')].reset_index(drop=True)
    test_ids = set(all_data.loc[all_data.role.eq('test'), 'geom_id'])
    assert not set(d.geom_id) & test_ids
    selections, traces, milestones, failures = {}, [], [], []
    gp_calls, selected_gp_calls, mlp_caps, selected_mlp_caps = [], [], 0, 0
    observed = {}
    for name in NAMES:
        selections[name] = {}
        for group, targets in GROUPS.items():
            folder = DEST/'studies'/name/group
            records = sorted([json.loads(p.read_text()) for p in folder.glob('trial*/result.json')], key=lambda r:r['trial'])
            with sqlite3.connect(folder/'study.sqlite3') as db:
                numbers = {r[0] for r in db.execute("select number from trials where state='COMPLETE'")}
                assert db.execute("select count(*) from trials where state='RUNNING'").fetchone()[0]==0
            assert len(records)==budget, (name,group,len(records))
            assert numbers=={r['trial'] for r in records}
            observed[name+'-'+group] = len(records)
            best, best_history = None, []
            for index, r in enumerate(records, 1):
                assert r['candidate']==name and r['group']==group and not r['test_scored']
                config = json.loads((folder/f'trial{r["trial"]:04d}/config.json').read_text())
                assert config['config']==r['config']
                scores, ys, ps = [], [], []
                for fold in range(5):
                    tr = d[d.fold.ne(fold)]
                    held = d[d.fold.eq(fold) & eligible(d,group)]
                    fdir = folder/f'trial{r["trial"]:04d}'
                    f = json.loads((fdir/f'fold{fold}.json').read_text())
                    assert set(f['eligible_training_ids'])==set(tr.loc[eligible(tr,group),'geom_id'])
                    assert not set(f['eligible_training_ids']) & (set(held.geom_id)|test_ids)
                    with np.load(fdir/f'fold{fold}.npz') as z:
                        np.testing.assert_array_equal(z['ids'].astype(str), held.geom_id.to_numpy(str))
                        y,p = z['actual'],z['predicted']
                        np.testing.assert_allclose(y, held[targets].to_numpy(float), rtol=0,atol=0)
                        sd = np.maximum(tr.loc[eligible(tr,group), targets].std(ddof=0).to_numpy(),1e-8)
                        np.testing.assert_allclose(sd,z['training_target_sd'],rtol=0,atol=0)
                        assert np.isfinite(p).all() and y.shape==p.shape
                        err = np.abs(y-p).mean(0)
                        value = float(np.mean(err/sd)) if group=='angle_displacement' else float(err.mean())
                        np.testing.assert_allclose(value,f['objective'],rtol=1e-12,atol=1e-12)
                        for j,t in enumerate(targets):
                            recomputed = metrics_reg(y[:,j],p[:,j])
                            for key,v in recomputed.items():
                                np.testing.assert_allclose(v,f['metrics'][t][key],rtol=1e-12,atol=1e-12)
                        scores.append(value);ys.append(y);ps.append(p)
                    diag=f['diagnostics'];gp_calls.extend(diag.get('kernel_optimization',[]))
                    mlp_caps += diag.get('epochs')==1200
                score=float(np.mean(scores))
                np.testing.assert_allclose(score,r['objective'],rtol=1e-12,atol=1e-12)
                y,p=np.concatenate(ys),np.concatenate(ps)
                report = {t: metrics_reg(y[:,j],p[:,j]) for j,t in enumerate(targets)}
                if group=='barrier':
                    report['barrier_mJ']=metrics_reg(10.**y[:,0],10.**p[:,0])
                for t,m in report.items():
                    for key,v in m.items():
                        np.testing.assert_allclose(v,r['metrics'][t][key],rtol=1e-12,atol=1e-12)
                if best is None or score<best['objective']:
                    best=r
                best_history.append(best['objective'])
                traces.append({'candidate':name,'group':group,'completed_trial':index,
                    'optuna_trial_number':r['trial'],'phase':'initial' if index<=10 else 'TPE',
                    'objective':score,'best_so_far':best['objective'],'seconds':r['seconds']})
            saved=json.loads((folder/'best.json').read_text())
            np.testing.assert_allclose(saved['objective'],best['objective'],rtol=0,atol=0)
            assert saved['trial']==best['trial']
            selections[name][group]=best
            for f in best['folds']:
                selected_gp_calls.extend(f['diagnostics'].get('kernel_optimization',[]))
                selected_mlp_caps += f['diagnostics'].get('epochs')==1200
            milestones.append({'candidate':name,'group':group,'trials':budget,'folds':5,
                **{f'best_at_{n}':best_history[n-1] for n in sorted(set([1,10,budget-5,budget]))},
                'last5_improvement':best_history[budget-6]-best_history[-1],
                'last5_relative_improvement':(best_history[budget-6]-best_history[-1])/max(abs(best_history[budget-6]),1e-12),
                'sum_fold_seconds':sum(r['seconds'] for r in records),
                'last_improvement_trial':max(i+1 for i in range(budget) if i==0 or best_history[i]<best_history[i-1])})
            failures += [str(p.relative_to(DEST)) for p in folder.glob('trial*/failure.json')]
    startup = {}
    for family in ['mlp','xgboost','gp']:
        for group in GROUPS:
            a=[json.loads(p.read_text()) for p in (DEST/'studies'/family/group).glob('trial*/result.json')]
            b=[json.loads(p.read_text()) for p in (DEST/'studies'/('rom_'+family)/group).glob('trial*/result.json')]
            assert [r['config'] for r in sorted(a,key=lambda r:r['trial'])[:10]] == [r['config'] for r in sorted(b,key=lambda r:r['trial'])[:10]]
            startup[family+'-'+group]=True
    rom=[]
    for fold in range(5):
        r=json.loads((DEST/'rom'/f'fold{fold}-provenance.json').read_text())
        ids=set(d.loc[d.fold.ne(fold),'geom_id'])
        assert r['solver_success'] and set(r['training_ids'])==ids
        assert all(k.split('|')[0] in ids for k in r['calibration_curve_keys'])
        rom.append({'fold':fold,'train_geometries':len(ids),'heldout_overlap':0})
    old=pd.read_csv(OLD/'comparison_20trials.csv').set_index('candidate')
    old_selected=json.loads((OLD/'selected_configs_20trials.json').read_text())
    results,comparisons=[],[]
    for name, groups in selections.items():
        row={'candidate':name,'label':LABELS[name],'classification_f1_reused':old.loc[name,'oof_f1']}
        old_reg=old_selected[name]['regression']['trial']
        prior=json.loads((OLD/'studies'/name/'regression'/f'trial{old_reg:04d}/result.json').read_text())
        # Require exactly the same OOF population, not just the same number of rows.
        for group in GROUPS:
            selected=groups[group]['trial']
            for fold in range(5):
                with np.load(OLD/'studies'/name/'regression'/f'trial{old_reg:04d}'/f'fold{fold}.npz') as z:
                    with np.load(DEST/'studies'/name/group/f'trial{selected:04d}'/f'fold{fold}.npz') as q:
                        np.testing.assert_array_equal(z['ids'],q['ids'])
            for t,m in groups[group]['metrics'].items():
                row.update({t+'_'+k:v for k,v in m.items()})
                if t in prior['metrics']:
                    before=prior['metrics'][t]
                    comparisons.append({'candidate':name,'label':LABELS[name],'target':t,'n':m['n'],
                        'old_mae':before['mae'],'new_mae':m['mae'],
                        'mae_reduction_percent':100*(before['mae']-m['mae'])/before['mae'],
                        'old_rmse':before['rmse'],'new_rmse':m['rmse'],
                        'old_r2':before['r2'],'new_r2':m['r2']})
            row[group+'_selected_trial']=groups[group]['trial']
        results.append(row)
    pd.DataFrame(results).to_csv(DEST/f'results_{budget}trials.csv',index=False)
    pd.DataFrame(comparisons).to_csv(DEST/f'old_vs_grouped_{budget}trials.csv',index=False)
    history=pd.DataFrame(traces);history.to_csv(DEST/f'search_history_{budget}trials.csv',index=False)
    pd.DataFrame(milestones).to_csv(DEST/f'presentation_milestones_{budget}trials.csv',index=False)
    dump(DEST/f'selected_configs_{budget}trials.json',{n:{g:{'trial':r['trial'],'objective':r['objective'],'config':r['config']} for g,r in groups.items()} for n,groups in selections.items()})
    dump(DEST/f'audit_{budget}trials.json',{'studies':len(NAMES)*len(GROUPS),'complete_trials':len(traces),
        'fold_predictions_checked':len(traces)*5,'test_scored':False,'rom':rom,
        'source_data_hashes_verified':True,'all_objectives_recomputed':True,
        'same_old_new_oof_geometry_ids':True,'paired_initial_configs_equal':startup,
        'actual_completed_trials':observed,'operational_failures':failures,
        'gp_inner_calls':len(gp_calls),'gp_inner_not_converged':sum(not r['success'] for r in gp_calls),
        'selected_gp_inner_calls':len(selected_gp_calls),'selected_gp_inner_not_converged':sum(not r['success'] for r in selected_gp_calls),
        'mlp_fold_epoch_caps':mlp_caps,'selected_mlp_fold_epoch_caps':selected_mlp_caps})
    colors={'mlp':'#2563eb','xgboost':'#d97706','gp':'#059669'}
    fig,axes=plt.subplots(1,3,figsize=(17,4.7),layout='constrained')
    titles=['Angle + displacement','Log10 energy barrier','Hinge transition stress']
    units=['Normalized MAE','Log10 MAE','MAE (MPa)']
    for ax,group,title,unit in zip(axes,GROUPS,titles,units):
        for name in NAMES:
            q=history[(history.candidate==name)&(history.group==group)]
            ax.step(q.completed_trial,q.best_so_far,where='post',label=LABELS[name],
                color=colors[name.removeprefix('rom_')],linestyle='--' if name.startswith('rom_') else '-',linewidth=2)
        ax.axvline(10.5,color='#64748b',linestyle=':',linewidth=1)
        ax.set(title=title,xlabel='Completed trials (10 initial + 10 TPE)',ylabel=unit+' (lower is better)',xlim=(1,budget))
        ax.set_xticks(sorted(set([1,5,10,15,budget])))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Grouped Forward | Train-only common 5-fold | No prediction averaging')
    fig.savefig(DEST/f'search_progress_{budget}trials.png',dpi=180);plt.close(fig)
    # Ratio chart makes the different response units directly comparable.
    compared=pd.DataFrame(comparisons)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,target,title in zip(axes,['angle_deg','deformation_mm','log10_barrier'],['Stable angle','Displacement','Log10 barrier']):
        q=compared[compared.target.eq(target)].set_index('candidate').loc[list(NAMES)]
        ax.barh(np.arange(len(NAMES)),q.mae_reduction_percent,color=['#2563eb' if v>=0 else '#dc2626' for v in q.mae_reduction_percent])
        ax.set_yticks(np.arange(len(NAMES)),[LABELS[n] for n in NAMES]);ax.invert_yaxis()
        ax.axvline(0,color='#475569',linewidth=1);ax.grid(axis='x',alpha=.2)
        ax.set(title=title,xlabel='MAE reduction vs old protocol (%)')
    fig.suptitle('Same OOF geometries | Positive = lower error | Protocol comparison, not causal isolation')
    fig.savefig(DEST/f'old_vs_grouped_{budget}trials.png',dpi=180);plt.close(fig)
    print(pd.DataFrame(results).to_string(index=False))
    print('\nOLD VS GROUPED\n'+pd.DataFrame(comparisons).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=int,default=20)
    audit_and_report(p.parse_args().trials)
