"""Audit completed TPE trials and plot equal-budget Train-only OOF results."""
from pathlib import Path
import json
import sys
import hashlib
import sqlite3

BASE=Path(__file__).resolve().parents[1]
for folder in [BASE,BASE/'forward',BASE.parent/'src']:sys.path.insert(0,str(folder))
import numpy as np
import pandas as pd
from runtime_paths import RUNS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from runtime_paths import RUNS
from common import load_data, dump, metrics_cls, reg_report, TARGETS
from forward_six.models import NAMES

DEST=RUNS/'forward-six/tuning'
LABELS={'mlp':'MLP','xgboost':'XGBoost','gp':'GP','rom_mlp':'ROM + MLP','rom_xgboost':'ROM + XGBoost','rom_gp':'ROM + GP'}


def audit_and_report(expected=15):
    protocol=json.loads((DEST/'protocol.json').read_text())
    for name,value in protocol['data_sha256'].items():
        assert hashlib.sha256((RUNS/'unified-forward/data'/name).read_bytes()).hexdigest()==value
    for name,value in protocol['source_sha256'].items():
        assert hashlib.sha256((BASE/'forward_six'/name).read_bytes()).hexdigest()==value
    d=load_data();d=d[d.role.eq('train')].reset_index(drop=True)
    traces=[];selections={};failures=[];actual_counts={};warnings_count=0;gp_runs=0;gp_nonconverged=0;epoch_caps=0
    for name in NAMES:
        selections[name]={}
        for task in ['classification','regression']:
            folder=DEST/'studies'/name/task
            records=[]
            for p in folder.glob('trial*/result.json'):
                r=json.loads(p.read_text());assert r['candidate']==name and r['task']==task
                records.append((p.parent,r))
            # Trial number may include an operationally failed trial; completed budget is separate.
            records.sort(key=lambda t:t[1]['trial'])
            with sqlite3.connect(folder/'study.sqlite3') as db:
                db_numbers={row[0] for row in db.execute("select number from trials where state='COMPLETE'")}
                running=db.execute("select count(*) from trials where state='RUNNING'").fetchone()[0]
            assert db_numbers=={r['trial'] for _,r in records}, (name,task,'database/result mismatch')
            assert running==0, (name,task,'study still running')
            actual_counts[name+'-'+task]=len(records)
            if len(records)<expected:raise ValueError(f'{name}/{task}: need at least {expected}, got {len(records)}')
            records=records[:expected]
            yy,pp=[],[];best=None
            for completed_index,(trial_folder,r) in enumerate(records,1):
                ys,ps=[],[];scaled=[]
                assert not r['test_scored']
                for fold in range(5):
                    held=d[d.fold==fold]
                    held=held[held.label.notna() if task=='classification' else held.is_reg].reset_index(drop=True)
                    with np.load(trial_folder/f'fold{fold}.npz') as z:
                        ids,actual,predicted=z['ids'].astype(str),z['actual'],z['predicted']
                        np.testing.assert_array_equal(ids,held.geom_id.to_numpy(str))
                        truth=held.label.eq('bi>mono').to_numpy() if task=='classification' else held[TARGETS].to_numpy(float)
                        np.testing.assert_allclose(actual,truth,rtol=0,atol=0)
                        if task=='classification':assert np.allclose(z['probabilities'].sum(1),1,atol=1e-6)
                        else:
                            sd=np.maximum(d.loc[(d.fold!=fold)&d.is_reg,TARGETS].std(ddof=0).to_numpy(),1e-8)
                            np.testing.assert_allclose(z['training_target_sd'],sd)
                            scaled.append(float(np.mean(np.abs(actual-predicted).mean(0)/sd)))
                    record=json.loads((trial_folder/f'fold{fold}.json').read_text())
                    warnings_count+=len(record['warnings'])
                    for opts in record.get('diagnostics',{}).get('kernel_optimization',[]):
                        for run in opts:
                            gp_runs+=1;gp_nonconverged+=not run['success']
                    if record.get('diagnostics',{}).get('epochs')==1200:epoch_caps+=1
                    ys.append(actual);ps.append(predicted)
                y,pred=np.concatenate(ys),np.concatenate(ps)
                value=metrics_cls(y,pred)['f1'] if task=='classification' else float(np.mean(scaled))
                np.testing.assert_allclose(value,r['objective'],rtol=1e-12,atol=1e-12)
                if best is None or (value>best['objective'] if task=='classification' else value<best['objective']):best=r
                traces.append({'candidate':name,'task':task,'completed_trial':completed_index,
                               'optuna_trial_number':r['trial'],'phase':'initial' if completed_index<=10 else 'TPE',
                               'objective':value,'best_so_far':best['objective'],'seconds':r['seconds']})
            saved=json.loads((folder/'best.json').read_text())
            if task=='classification':assert saved['objective'] >= best['objective']-1e-12
            else:assert saved['objective'] <= best['objective']+1e-12
            selections[name][task]=best
            for f in folder.glob('trial*/failure.json'):
                failures.append({'path':str(f.relative_to(DEST)),**json.loads(f.read_text())})
    rom_audit=[]
    for fold in range(5):
        r=json.loads((DEST/'rom'/f'fold{fold}-provenance.json').read_text())
        ids=set(d[d.fold!=fold].geom_id)
        assert set(r['training_ids'])==ids and r['solver_success']
        assert all(key.split('|')[0] in ids for key in r['calibration_curve_keys'])
        rom_audit.append({'fold':fold,'calibration_geometries':len(ids),'heldout_overlap':0})
    paired_startup = {}
    for base in ['mlp','xgboost','gp']:
        for task in ['classification','regression']:
            a = sorted((DEST/'studies'/base/task).glob('trial*/result.json'), key=lambda p:json.loads(p.read_text())['trial'])
            b = sorted((DEST/'studies'/('rom_'+base)/task).glob('trial*/result.json'), key=lambda p:json.loads(p.read_text())['trial'])
            same = [json.loads(p.read_text())['config'] for p in a[:10]] == [json.loads(p.read_text())['config'] for p in b[:10]]
            assert same, (base,task,'paired initial search settings differ')
            paired_startup[base+'-'+task] = same
    rows=[]
    for name,parts in selections.items():
        c=parts['classification'];r=parts['regression'];cm=c['metrics'];rm=r['metrics']
        rows.append({'candidate':name,'label':LABELS[name],'classification_trials':expected,'regression_trials':expected,
                     'actual_classification_trials_preserved':actual_counts[name+'-classification'],
                     'actual_regression_trials_preserved':actual_counts[name+'-regression'],
                     'oof_f1':cm['f1'],'oof_precision':cm['precision'],'oof_recall':cm['recall'],
                     'oof_balanced_accuracy':cm['balanced_accuracy'],'classification_n':cm['n'],
                     'angle_mae_deg':rm['angle_deg']['mae'],'deformation_mae_mm':rm['deformation_mm']['mae'],
                     'log10_barrier_mae':rm['log10_barrier']['mae'],'barrier_mae_mJ':rm['barrier_mJ']['mae'],
                     'joint_scaled_mae':r['objective'],'regression_n':rm['angle_deg']['n'],
                     'classification_best_trial':c['trial'],'regression_best_trial':r['trial'],
                     'classification_initial_objective':next(t['objective'] for t in traces if t['candidate']==name and t['task']=='classification'),
                     'regression_initial_objective':next(t['objective'] for t in traces if t['candidate']==name and t['task']=='regression')})
    frame=pd.DataFrame(rows);frame.to_csv(DEST/f'comparison_{expected}trials.csv',index=False)
    pd.DataFrame(traces).to_csv(DEST/f'search_history_{expected}trials.csv',index=False)
    dump(DEST/f'selected_configs_{expected}trials.json',
         {n:{task:{'trial':r['trial'],'objective':r['objective'],'config':r['config']} for task,r in parts.items()}
          for n,parts in selections.items()})
    selected_gp_calls=[];selected_epoch_caps=0
    for name,parts in selections.items():
        for task,r in parts.items():
            for f in r['folds']:
                selected_epoch_caps += f.get('diagnostics',{}).get('epochs')==1200
                for calls in f.get('diagnostics',{}).get('kernel_optimization',[]):
                    selected_gp_calls.extend(calls)
    dump(DEST/f'audit_{expected}trials.json',{'studies':12,'complete_trials':len(traces),'fold_predictions_checked':len(traces)*5,
                                   'test_scored':False,'rom_fold_audit':rom_audit,'prediction_averaging':False,
                                   'operational_failures':failures,'warning_messages':warnings_count,
                                   'gp_inner_optimization_calls':gp_runs,'gp_inner_not_converged':gp_nonconverged,
                                   'selected_gp_inner_calls':len(selected_gp_calls),
                                   'selected_gp_inner_not_converged':sum(not c['success'] for c in selected_gp_calls),
                                   'selected_mlp_folds_at_epoch_cap':selected_epoch_caps,
                                   'mlp_folds_selected_at_epoch_cap':epoch_caps,
                                   'all_saved_objectives_recomputed_from_oof':True, 'common_prefix_budget':expected, 'actual_trials_preserved':actual_counts, 'paired_startup_configs_identical':paired_startup})
    milestones=[]
    for name in NAMES:
        for task in ['classification','regression']:
            q=pd.DataFrame([t for t in traces if t['candidate']==name and t['task']==task])
            a=float(q.loc[q.completed_trial==(expected-5),'best_so_far'].iloc[0])
            b=float(q.loc[q.completed_trial==expected,'best_so_far'].iloc[0])
            row={'candidate':name,'task':task,'folds_per_trial':5,'complete_trials':expected,
                 'first10':'1 baseline + 9 random','adaptive_trials':expected-10,
                 'sum_fold_seconds':float(q.seconds.sum())}
            for count in sorted(set([1,10,expected-5,expected])):
                row[f'best_at_{count}']=float(q.loc[q.completed_trial==count,'best_so_far'].iloc[0])
            row['last5_improvement']=b-a if task=='classification' else a-b
            row['last5_relative_improvement']=(a-b)/max(abs(a),1e-12) if task=='regression' else None
            row['interpretation']='within-budget improvement, not proof of convergence'
            milestones.append(row)
    pd.DataFrame(milestones).to_csv(DEST/f'presentation_milestones_{expected}trials.csv',index=False)
    history=pd.DataFrame(traces)
    colors={'mlp':'#2563eb','xgboost':'#d97706','gp':'#059669'}
    fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for ax,task,title,ylabel in zip(axs,['classification','regression'],
                                  ['Stability classification','Joint regression'],
                                  ['Best pooled OOF F1 (higher is better)','Best normalized MAE (lower is better)']):
        for name in NAMES:
            q=history[(history.candidate==name)&(history.task==task)]
            ax.step(q.completed_trial,q.best_so_far,where='post',color=colors[name.removeprefix('rom_')],
                    linestyle='--' if name.startswith('rom_') else '-',label=LABELS[name],linewidth=2)
        ax.axvline(10.5,color='#64748b',linestyle=':',linewidth=1)
        ax.set(title=title,xlabel=f'Completed trials (initial 10, then {expected-10} TPE)',ylabel=ylabel,xlim=(1,expected))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Six Forward candidates | Train-only 5-fold | No prediction averaging')
    fig.savefig(DEST/f'search_progress_{expected}trials.png',dpi=180);plt.close(fig)
    print(frame.to_string(index=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=int,default=20)
    audit_and_report(parser.parse_args().trials)
