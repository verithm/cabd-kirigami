"""Isolate kernel and regularization-cap effects after expanded-bound pilots.

Preserves the original pilot protocol/results. Conditional new-50 studies are
stored separately in revised50/. Uses Train CV only, never Test.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parents[1]
for p in [BASE,BASE/'forward',BASE.parent/'src']:sys.path.insert(0,str(p))
import argparse,json,hashlib
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from common import dump
from forward_six import gp_improvement as gp

ROOT=gp.DEST
CONTROLS=['rbf_original_bounds','matern05_original_bounds','signal_cap100','signal_cap10000']


def config(name,variant):
    cfg=gp.base_config(name)
    if variant=='rbf_original_bounds':cfg['kernel']='rbf'
    if variant=='matern05_original_bounds':cfg['kernel']='matern05'
    if variant=='signal_cap100':cfg['signal_upper']=100.
    if variant=='signal_cap10000':cfg['signal_upper']=1e4
    # Keep the original 60 iterations to isolate each individual change.
    return cfg


def new_suggest(trial):
    return {'kernel':trial.suggest_categorical('kernel',['matern05','matern15','matern25','rbf']),
        'ard':trial.suggest_categorical('ard',[True,False]),
        'signal_upper':trial.suggest_categorical('signal_upper',[10.,100.,1e3,1e4,1e6]),
        'length_upper':trial.suggest_categorical('length_upper',[1e3,1e5]),
        # All initial signal values must lie inside every selected signal bound.
        'signal_initial':trial.suggest_float('signal_initial',.01,10.,log=True),
        'length_initial':trial.suggest_float('length_initial',.05,50.,log=True),
        'max_optimizer_iterations':150}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--controls-only',action='store_true');args=parser.parse_args()
    assert (ROOT/'pilot_decisions.json').exists(),'Finish initial pilot first'
    proto={'version':1,'reason':'Separate kernel-only and moderate/lower signal-cap effects from expanded-bound pilots',
        'controls':{n:{v:config(n,v) for v in CONTROLS} for n in gp.NAMES},
        'predeclared_trigger':'same as initial pilot: gain>=.002 and improved folds>=3 OR gain>=.005',
        'triggering_rule_reused_without_test':True,'test_scored':False,
        'new50_space':{'kernel':['matern05','matern15','matern25','rbf'],'ard':[True,False],
            'signal_upper':[10.,100.,1e3,1e4,1e6],'length_upper':[1e3,1e5],
            'signal_initial_log':[.01,10.],'length_initial_log':[.05,50.],
            'max_optimizer_iterations':150,'restarts':0,'max_iter_predict':60},
        'source_sha256':{str(Path(__file__)):gp.sha(Path(__file__)),str(Path(gp.__file__)):gp.sha(Path(gp.__file__))},
        'original_protocol_sha256':gp.sha(ROOT/'protocol.json'),
        'prior_pilot_summary_sha256':gp.sha(ROOT/'pilot_summary.json'),
        'disclosure':'Eight pre-search pilots plus original 50 trial history per family; new 50 is not equal total historical tuning opportunity.'}
    p=ROOT/'controls_protocol.json'
    if p.exists():assert json.loads(p.read_text())==proto
    else:dump(p,proto)
    jobs=[(n,v,f) for n in gp.NAMES for v in CONTROLS for f in range(5)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(gp.fold_worker,n,config(n,v),f,str(ROOT/'controls'/n/v)) for n,v,f in jobs]
        for future in as_completed(fs):future.result()
    summary=json.loads((ROOT/'pilot_summary.json').read_text());decisions={}
    for n in gp.NAMES:
        baseline=gp.old_choice(n)
        path=gp.RUNS/'forward-six/tuning/studies'/n/'classification'/f'trial{baseline["trial"]:04d}'/'result.json'
        old=json.loads(path.read_text());foldscores=np.array([f['metrics']['f1'] for f in old['folds']])
        for v in CONTROLS:
            r=gp.aggregate(n,config(n,v),ROOT/'controls'/n/v)
            gains=np.array([f['metrics']['f1'] for f in r['folds']])-foldscores
            delta=r['objective']-old['objective']
            trigger=bool((delta>=.002 and (gains>1e-12).sum()>=3) or delta>=.005)
            summary.append({'candidate':n,'variant':v,'baseline_f1':old['objective'],'pilot_f1':r['objective'],
                'gain_F1_percentage_points':100*delta,'folds_improved':int((gains>1e-12).sum()),
                'folds_worse':int((gains< -1e-12).sum()),'trigger_new_50':trigger,
                'precision':r['metrics']['precision'],'recall':r['metrics']['recall'],
                'paired_fold_gains':gains.tolist(),'config':r['config']})
        candidates=[r for r in summary if r['candidate']==n];qualified=[r for r in candidates if r['trigger_new_50']]
        best=max(qualified or candidates,key=lambda r:r['pilot_f1'])
        decisions[n]={'run_new_50':bool(qualified),'pilot_winner':best,'criterion':'same predeclared initial-pilot threshold'}
    dump(ROOT/'all_pilot_summary.json',summary);dump(ROOT/'controls_decisions.json',decisions)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['config','paired_fold_gains']} for r in summary]).to_csv(ROOT/'all_pilot_summary.csv',index=False)
    print('ALL_PILOT_DECISIONS',json.dumps(decisions),flush=True)
    if args.controls_only:return
    newroot=ROOT/'revised50';newroot.mkdir(exist_ok=True)
    dump(newroot/'protocol.json',{'controls_protocol_sha256':gp.sha(p),'decisions':decisions,
         'space':proto['new50_space'],'startup_trials':10,'seed':42,'total_COMPLETE_trials':50,'test_scored':False})
    gp.DEST=newroot;gp.suggest=new_suggest
    for n in gp.NAMES:
        if decisions[n]['run_new_50']:gp.run_study(n,decisions[n],args.workers)
    dump(ROOT/'complete.json',{'time':gp.now(),'initial_and_control_pilots':16,'pilot_fold_fits':80,
        'new_50_completed_for':[n for n in gp.NAMES if decisions[n]['run_new_50']],
        'test_scored':False,'original_studies_modified':False,'production_models_modified':False})


if __name__=='__main__':main()
