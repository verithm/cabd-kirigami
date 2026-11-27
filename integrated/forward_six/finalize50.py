"""Read-only validation and result packaging after the requested 50-trial stop.

Never trains a model, evaluates Test, changes production selection or pushes Git.
"""
import argparse,csv,hashlib,json,sqlite3
from pathlib import Path

NAMES=('mlp','xgboost','gp','rom_mlp','rom_xgboost','rom_gp')
TASKS=('classification','angle_displacement','barrier','stress')
REPO=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True)
    args=p.parse_args();runs=args.runs.resolve();root=runs/'forward-refinement'
    complete=json.loads((root/'complete.json').read_text())
    assert complete['budget_per_study']==50 and complete['user_requested_stop']
    assert complete['additional_trials_after_50'] is False and not complete['test_scored']
    population={}
    for name in NAMES:
        for task in TASKS:
            parent='forward-six' if task=='classification' else 'forward-grouped'
            dbpath=runs/parent/'tuning/studies'/name/task/'study.sqlite3'
            with sqlite3.connect(dbpath) as db:
                states=dict(db.execute('select state,count(*) from trials group by state'))
            assert states.get('COMPLETE',0)==50 and states.get('RUNNING',0)==0,(name,task,states)
            population[name+'|'+task]=states
    # The original fixed data/model protocols remain in force.
    verified=0
    for parent in ['forward-six','forward-grouped']:
        proto=json.loads((runs/parent/'tuning/protocol.json').read_text())
        for key,digest in proto['data_sha256'].items():
            path=Path(key) if Path(key).is_absolute() else (runs/key if '/' in key else runs/'unified-forward/data'/key)
            assert sha(path)==digest,key;verified+=1
        for key,digest in proto['source_sha256'].items():
            path=Path(key) if Path(key).is_absolute() else (REPO/key if '/' in key else REPO/'integrated/forward_six'/key)
            assert sha(path)==digest,key;verified+=1
    current=json.loads((root/'protocol.json').read_text())
    assert current['maximum_budget']==50
    for key,digest in current['sha256'].items():
        assert sha(Path(key))==digest,key;verified+=1
    audit=json.loads((root/'audit_50trials.json').read_text())
    assert audit['fold_predictions_checked']==6000 and audit['all_objectives_and_metrics_recomputed']
    assert audit['data_and_fold_truth_checked'] and not audit['test_scored']
    stages={n:{r['candidate']:r for r in csv.DictReader((root/f'comparison_{n}trials.csv').open())} for n in [20,40,50]}
    configs=json.loads((root/'selected_configs_50trials.json').read_text())
    objectives=json.loads((root/'stage_50.json').read_text())['objectives']
    recommendations={}
    for task in TASKS:
        best=(max if task=='classification' else min)(NAMES,key=lambda n:objectives[n+'|'+task])
        recommendations[task]={'candidate':best,'objective':objectives[best+'|'+task],
            **configs[best][task],'status':'provisional CV best; final Train refit/Test not performed'}
    fields={'f1':'F1','angle_deg_mae':'degrees','deformation_mm_mae':'mm',
        'log10_barrier_mae':'log10 mJ','barrier_mJ_mae':'mJ','hinge_transition_max_MPa_mae':'MPa'}
    changes=[]
    for name in NAMES:
        for field,unit in fields.items():
            values={n:float(stages[n][name][field]) for n in [20,40,50]}
            changes.append({'candidate':name,'metric':field,'unit':unit,
                **{f'at_{n}':v for n,v in values.items()},
                'improvement_20_to_50':values[50]-values[20] if field=='f1' else values[20]-values[50],
                'mae_reduction_percent':None if field=='f1' else 100*(values[20]-values[50])/values[20],
                'F1_percentage_point_increase':100*(values[50]-values[20]) if field=='f1' else None})
    with (root/'progress_20_40_50.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(changes[0]));w.writeheader();w.writerows(changes)
    summary={'status':'Completed requested work and stopped at 50 per study','models':6,'tasks_per_model':4,
        'COMPLETE_trials_per_study':50,'total_relevant_COMPLETE_trials':1200,'fold_predictions_audited':6000,
        'all_studies_exactly_50':True,'RUNNING_trials':0,'verified_source_data_hashes':verified,
        'provisional_CV_best_by_task':recommendations,'test_scored':False,'refit_full_train':False,
        'production_forward_inverse_changed':False,'Git_push_performed':False,
        'convergence_claim':'Stopped at explicit user cap; no claim of full convergence or global optimum.',
        'completed_at':complete['finished_at'],'database_states':population}
    (root/'run_summary_50.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    manifest_files=[x for x in root.iterdir() if x.is_file() and
        (any(tag in x.name for tag in ['20trials','40trials','50trials']) or x.name in
         ['progress_20_40_50.csv','run_summary_50.json','complete.json','protocol.json',
          'optimized-parallel-validation.json','parallel-execution-switch.json','user_requested_50_cap.json'])]
    manifest={'artifacts_relative_to_run_folder':{str(x.relative_to(root)):sha(x) for x in manifest_files},
              'source_files':{str(Path(__file__).resolve()):sha(Path(__file__))}}
    (root/'final_manifest_50.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='database_states'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
