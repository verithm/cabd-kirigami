"""Equal-budget continuation of six candidates and four separate Forward tasks.

Wait for the already launched 40-trial batch, audit it, then advance all studies
by the same 20-trial budget until two plateau windows or the declared cap.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,sqlite3,time,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
for folder in [BASE,BASE/'forward',BASE.parent/'src']:
    sys.path.insert(0,str(folder))
import numpy as np
import pandas as pd
from common import dump,metrics_reg
from runtime_paths import RUNS
from forward_six.models import NAMES
from forward_six import tune as classification
from forward_six import grouped_tune as regression
from forward_six.parallel_classification import classification_worker

DEST=RUNS/'forward-refinement'
TASKS=('classification','angle_displacement','barrier','stress')
LABELS={'mlp':'MLP','xgboost':'XGBoost','gp':'GP','rom_mlp':'ROM + MLP','rom_xgboost':'ROM + XGBoost','rom_gp':'ROM + GP'}


def folder(name,task):
    return (classification.DEST if task=='classification' else regression.DEST)/'studies'/name/task


def read_records(name,task):
    return sorted([json.loads(p.read_text()) for p in folder(name,task).glob('trial*/result.json')],key=lambda r:r['trial'])


def counts():
    result={}
    for name in NAMES:
        for task in TASKS:
            p=folder(name,task)/'study.sqlite3'
            if not p.exists():
                result[name+'|'+task]=0;continue
            with sqlite3.connect(p) as db:
                result[name+'|'+task]=db.execute("select count(*) from trials where state='COMPLETE'").fetchone()[0]
    return result


def verify_protocol(args):
    DEST.mkdir(parents=True,exist_ok=True)
    classification.prepare();regression.prepare()
    files=[classification.DEST/'protocol.json',regression.DEST/'protocol.json',
        classification.DEST/'comparison_20trials.csv',regression.DEST/'results_20trials.csv',Path(__file__),
        BASE/'forward_six/parallel_classification.py']
    proto={'candidates':list(NAMES),'tasks':list(TASKS),'initial_budget':20,'schedule':[20,40,50],
        'maximum_budget':args.max_trials,'plateau_windows':2,'regression_relative_improvement_threshold':.005,
        'classification_absolute_F1_improvement_threshold':.001,'folds':5,'seed':42,
        'search_space_and_training_limits':'unchanged from immutable source protocols',
        'continuation':'saved Optuna TPE state; no restart of startup trials',
        'execution':'GP classification folds parallel (3 workers), same immutable fit/scoring functions',
        'execution_history':'protocol_sequential-v1.json preserves original sequential execution protocol',
        'stop':'Explicit user instruction: all 24 studies reach exactly 50 COMPLETE trials, then stop; no further tuning',
        'interpretation':'search plateau within this fixed space/budget; not proof of global or training convergence',
        'test_scored':False,'refit_full_train':False,'prediction_averaging':False,
        'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    p=DEST/'protocol.json'
    if p.exists():
        assert json.loads(p.read_text())==proto,'Refinement protocol changed'
    else:
        dump(p,proto)
    return proto


def snapshot(budget,previous=None):
    ct=classification.get_train();rt=regression.get_train()
    held={};sd={}
    for task in TASKS:
        d=ct if task=='classification' else rt
        for f in range(5):
            mask=d.label.notna() if task=='classification' else regression.eligible(d,task)
            held[task,f]=d[d.fold.eq(f)&mask]
            if task!='classification':
                sd[task,f]=np.maximum(d.loc[d.fold.ne(f)&mask,regression.GROUPS[task]].std(ddof=0).to_numpy(),1e-8)
    rows=[];trace=[];selected={};objectives={};audited=0;selected_gp=[];epoch_caps=0
    for name in NAMES:
        row={'candidate':name,'label':LABELS[name],'trials_per_task':budget};selected[name]={}
        for task in TASKS:
            all_records=read_records(name,task)
            assert len(all_records)>=budget,(name,task,len(all_records),budget)
            with sqlite3.connect(folder(name,task)/'study.sqlite3') as db:
                complete={r[0] for r in db.execute("select number from trials where state='COMPLETE'")}
            rs=all_records[:budget];assert all(r['trial'] in complete for r in rs)
            best=None
            for index,r in enumerate(rs,1):
                assert r['candidate']==name and not r['test_scored']
                values=[];yy=[];pp=[]
                for f in range(5):
                    p=folder(name,task)/f'trial{r["trial"]:04d}'/f'fold{f}.npz'
                    with np.load(p) as z:
                        h=held[task,f]
                        np.testing.assert_array_equal(z['ids'].astype(str),h.geom_id.to_numpy(str))
                        y,pred=z['actual'],z['predicted'];assert np.isfinite(pred).all()
                        truth=h.label.eq('bi>mono').to_numpy() if task=='classification' else h[regression.GROUPS[task]].to_numpy(float)
                        np.testing.assert_allclose(y,truth,rtol=0,atol=0)
                        if task=='classification':
                            np.testing.assert_allclose(z['probabilities'].sum(1),1,atol=1e-6)
                        else:
                            np.testing.assert_allclose(z['training_target_sd'],sd[task,f],rtol=0,atol=0)
                            err=np.abs(y-pred).mean(0)
                            values.append(float(np.mean(err/sd[task,f])) if task=='angle_displacement' else float(err.mean()))
                    yy.append(y);pp.append(pred);audited+=1
                y,pred=np.concatenate(yy),np.concatenate(pp)
                if task=='classification':
                    metric=classification.metrics_cls(y,pred);objective=metric['f1']
                else:
                    metric={t:metrics_reg(y[:,j],pred[:,j]) for j,t in enumerate(regression.GROUPS[task])}
                    if task=='barrier':metric['barrier_mJ']=metrics_reg(10.**y[:,0],10.**pred[:,0])
                    objective=float(np.mean(values))
                np.testing.assert_allclose(objective,r['objective'],rtol=1e-11,atol=1e-12)
                if task=='classification':
                    for k,v in metric.items():np.testing.assert_allclose(v,r['metrics'][k],rtol=1e-11,atol=1e-12)
                else:
                    for t,m in metric.items():
                        for k,v in m.items():np.testing.assert_allclose(v,r['metrics'][t][k],rtol=1e-11,atol=1e-12)
                if best is None or (objective>best['objective'] if task=='classification' else objective<best['objective']):best=r
                trace.append({'candidate':name,'task':task,'completed_trial':index,'optuna_trial_number':r['trial'],
                    'objective':objective,'best_so_far':best['objective'],'seconds':r['seconds']})
            selected[name][task]={'trial':best['trial'],'config':best['config'],'objective':best['objective']}
            objectives[name+'|'+task]=best['objective']
            for f in best['folds']:
                diag=f.get('diagnostics',{});epoch_caps+=diag.get('epochs')==1200
                for calls in diag.get('kernel_optimization',[]):
                    selected_gp.extend(calls if isinstance(calls,list) else [calls])
            if task=='classification':
                row.update({k:best['metrics'][k] for k in ['f1','precision','recall','balanced_accuracy','n']})
            else:
                for t,m in best['metrics'].items():row.update({t+'_'+k:v for k,v in m.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(DEST/f'comparison_{budget}trials.csv',index=False)
    history=pd.DataFrame(trace);history.to_csv(DEST/f'search_history_{budget}trials.csv',index=False)
    dump(DEST/f'selected_configs_{budget}trials.json',selected)
    audit={'trials_per_study':budget,'studies':24,'fold_predictions_checked':audited,
        'all_objectives_and_metrics_recomputed':True,'data_and_fold_truth_checked':True,'test_scored':False,
        'selected_gp_inner_calls':len(selected_gp),'selected_gp_inner_not_converged':sum(not r['success'] for r in selected_gp),
        'selected_mlp_fold_epoch_caps':epoch_caps}
    dump(DEST/f'audit_{budget}trials.json',audit)
    windows=[];streaks={}
    for key,value in objectives.items():
        if previous is None:streaks[key]=0;continue
        name,task=key.split('|');old=previous['objectives'][key]
        improvement=value-old if task=='classification' else (old-value)/max(abs(old),1e-12)
        threshold=.001 if task=='classification' else .005
        meaningful=improvement>=threshold
        full_window=budget-previous['budget']==20
        streaks[key]=(0 if meaningful else previous['streaks'][key]+1) if full_window else previous['streaks'][key]
        windows.append({'candidate':name,'task':task,'from_trials':previous['budget'],'to_trials':budget,
            'previous_best':old,'current_best':value,'improvement':improvement,
            'improvement_unit':'absolute F1' if task=='classification' else 'relative error reduction',
            'threshold':threshold,'meaningful':meaningful,'plateau_windows':streaks[key],
            'window_trials':budget-previous['budget'],'full_20_trial_plateau_window':full_window})
    if windows:pd.DataFrame(windows).to_csv(DEST/f'improvements_to_{budget}trials.csv',index=False)
    state={'budget':budget,'objectives':objectives,'streaks':streaks,'plateau_studies':sum(v>=2 for v in streaks.values()),
        'all_plateau':all(v>=2 for v in streaks.values()),'test_scored':False,'finished_at':classification.now()}
    dump(DEST/f'stage_{budget}.json',state)
    plot(history,budget)
    return state


def plot(history,budget):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'mlp':'#2563eb','xgboost':'#d97706','gp':'#059669'}
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    titles=['Stability classification','Angle + displacement','Log10 energy barrier','Hinge transition stress']
    units=['Best pooled OOF F1 (higher is better)','Best normalized MAE','Best log10 MAE','Best MAE (MPa)']
    for ax,task,title,unit in zip(axes.flat,TASKS,titles,units):
        for name in NAMES:
            q=history[(history.candidate==name)&(history.task==task)]
            ax.step(q.completed_trial,q.best_so_far,where='post',label=LABELS[name],
                color=colors[name.removeprefix('rom_')],linestyle='--' if name.startswith('rom_') else '-',linewidth=1.8)
        for x in range(20,budget+1,20):ax.axvline(x,color='#64748b',linestyle=':',linewidth=.8)
        ax.set(title=title,xlabel='Completed trials (saved TPE continuation after 20)',ylabel=unit,xlim=(1,budget))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Six Forward candidates | Equal budgets | Same Train-only 5-fold')
    fig.savefig(DEST/f'search_progress_{budget}trials.png',dpi=160);plt.close(fig)


def worker(name,task,target):
    return classification_worker(name,target,3) if task=='classification' else regression.study_worker(name,task,target)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--max-trials',type=int,default=50,choices=[50])
    p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    verify_protocol(args)
    previous=json.loads((DEST/'stage_20.json').read_text()) if (DEST/'stage_20.json').exists() else snapshot(20)
    # The initial 40-trial jobs were launched separately and must not be duplicated.
    while min(counts().values())<40:
        status={'stage':'waiting_existing_40','counts':counts(),'updated_at':classification.now()}
        dump(DEST/'live_progress.json',status);print('WAIT_40',min(status['counts'].values()),max(status['counts'].values()),flush=True)
        time.sleep(20)
    for target in [40,50]:
        state_path=DEST/f'stage_{target}.json'
        if state_path.exists():
            current=json.loads(state_path.read_text())
        else:
            if target>40:
                pairs=[('gp','classification'),('rom_gp','classification')]
                pairs += [(n,t) for n in ['gp','rom_gp'] for t in TASKS[1:]]
                pairs += [(n,t) for n in NAMES if n not in ['gp','rom_gp'] for t in TASKS]
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    futures=[ex.submit(worker,n,t,target) for n,t in pairs]
                    for f in as_completed(futures):
                        print('REFINEMENT_STUDY_DONE',f.result(),flush=True)
                        dump(DEST/'live_progress.json',{'stage':target,'counts':counts(),'updated_at':classification.now()})
            current=snapshot(target,previous)
        print('REFINEMENT_STAGE_DONE',target,'plateau',current['plateau_studies'],'/24',flush=True)
        dump(DEST/'latest.json',current)
        if target==50:
            reason='explicit_user_requested_50_cap'
            dump(DEST/'complete.json',{'budget_per_study':target,'reason':reason,'plateau_studies':current['plateau_studies'],
                'finished_at':classification.now(),'test_scored':False,'refit_full_train':False,
                'user_requested_stop':True,'additional_trials_after_50':False})
            return
        previous=current


if __name__=='__main__':main()
