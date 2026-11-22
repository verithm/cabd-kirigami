"""Recover selected-model Train OOF physical barrier errors and review fixed candidate pools."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from shared import *
from seungmin_models import fit,predict
DEST=RUNS/'barrier-tolerance-review'
ABS_TOL=[.02,.03,.05]
ANGLE=2.;DISP=.25

def fold_predictions(fold):
 path=DEST/f'cv_fold{fold}.csv'
 if path.exists():return str(path)
 d=load_data();tr=d[d.role=='train'];a=tr[tr.fold!=fold];b=tr[(tr.fold==fold)&tr.is_reg]
 model=fit('sm_mlp',a,'joint',0,42);p=predict(model,b)
 q=b[['geom_id','label','fold',*TARGETS]].copy()
 for j,k in enumerate(TARGETS):q[f'predicted_{k}']=p[:,j]
 q['actual_barrier_mJ']=10**q.log10_barrier;q['predicted_barrier_mJ']=10**q.predicted_log10_barrier
 q['absolute_error_mJ']=(q.actual_barrier_mJ-q.predicted_barrier_mJ).abs()
 q.to_csv(path,index=False);print('OOF FOLD COMPLETE',fold,len(q),flush=True);return str(path)

def candidate_sensitivity(method,seed):
 folder=DEST/'candidate_runs'/method/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True);geometry=Geometry()
 for i,path in enumerate(sorted((OUT/'runs'/method/f'seed{seed}').glob('TE*.json'))):
  target=folder/path.name
  if target.exists():continue
  r=json.loads(path.read_text());goal=np.array([r['goal'][k] for k in TARGETS]);energy=10**goal[2]
  with np.load(path.with_suffix('.npz')) as z:u,y,p,geo=(z[k].copy() for k in ['u','predicted','p_bi_mono','geometry_checked'])
  base=bound_mask(u)&(p>=np.float32(P_MIN))&(np.abs(y[:,0]-goal[0])<=ANGLE)&(np.abs(y[:,1]-goal[1])<=DISP)
  ae=np.abs(10**y[:,2]-energy)
  masks={f'abs_{t:.2f}mJ':base&(ae<=t) for t in ABS_TOL};masks['log10_0.17']=base&(np.abs(y[:,2]-goal[2])<=.17)
  needed=np.logical_or.reduce(list(masks.values()));extra=np.flatnonzero(needed&(geo<0))
  if len(extra):geo[extra]=geometry.many(u[extra]).astype(np.int8)
  scores={}
  for key,mask in masks.items():
   mask=mask&(geo==1);j=None
   if mask.any():
    denom=.17 if key=='log10_0.17' else float(key[4:8])
    barriererr=np.abs(y[:,2]-goal[2])/denom if key=='log10_0.17' else ae/denom
    error=np.maximum.reduce([np.abs(y[:,0]-goal[0])/ANGLE,np.abs(y[:,1]-goal[1])/DISP,barriererr]);j=np.flatnonzero(mask)[np.argmin(error[mask])]
   selected=None if j is None else dict(dimensions=dict(zip(FEATURES,to_phys(u[j]).tolist())),predicted=dict(zip(TARGETS,y[j].tolist())),predicted_barrier_mJ=float(10**y[j,2]),p_bi_mono=float(p[j]),normalized_max_error=float(error[j]))
   scores[key]=dict(success=bool(mask.any()),pass_count=int(mask.sum()),selected=selected)
  dump(target,dict(method=method,seed=seed,goal_id=r['goal_id'],goal=r['goal'],goal_barrier_mJ=float(energy),candidate_points=len(u),extra_geometry_checks=len(extra),scores=scores,source_trace=reference(path.with_suffix('.npz')),source_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()))
  if i==48:print('CANDIDATE REVIEW COMPLETE',method,seed,flush=True)
 return method,seed

def summarize():
 cv=pd.concat([pd.read_csv(DEST/f'cv_fold{i}.csv') for i in range(5)],ignore_index=True);assert len(cv)==836 and cv.geom_id.nunique()==836
 saved=json.loads((FORWARD/'cv_joint_0.json').read_text());e=cv.absolute_error_mJ.to_numpy()
 np.testing.assert_allclose([e.mean(),np.sqrt((e**2).mean())],[saved['barrier_mJ']['mae'],saved['barrier_mJ']['rmse']],rtol=1e-5,atol=1e-7)
 cv.to_csv(DEST/'train_oof_predictions.csv',index=False)
 stats=[]
 for name,q in [('Train OOF all partial bi',cv),('Train OOF bi->mono',cv[cv.label=='bi>mono'])]:
  e=q.absolute_error_mJ.to_numpy();stats.append(dict(population=name,n=len(q),mae_mJ=float(e.mean()),rmse_mJ=float(np.sqrt((e**2).mean())),median_mJ=float(np.median(e)),p80_mJ=float(np.quantile(e,.8)),p90_mJ=float(np.quantile(e,.9)),p95_mJ=float(np.quantile(e,.95)),**{f'within_{t:.2f}mJ':float((e<=t).mean()) for t in ABS_TOL}))
 pd.DataFrame(stats).to_csv(DEST/'forward_oof_error_summary.csv',index=False)
 bins=[0,.01,.05,.1,.3,.7,1.,np.inf];cv['energy_bin']=pd.cut(cv.actual_barrier_mJ,bins,right=False)
 grouped=[]
 for label,q in cv.groupby('energy_bin',observed=True):
  grouped.append(dict(energy_bin=str(label),n=len(q),mae_mJ=q.absolute_error_mJ.mean(),rmse_mJ=np.sqrt(np.mean(q.absolute_error_mJ**2)),**{f'within_{t:.2f}mJ':float((q.absolute_error_mJ<=t).mean()) for t in ABS_TOL}))
 pd.DataFrame(grouped).to_csv(DEST/'forward_error_by_energy_bin.csv',index=False)
 goals=pd.read_csv(OUT/'data/test_goals.csv');examples=[]
 for E in [.005,.01,.05,.1,.3,.5,1.,2.]:
  examples.append(dict(goal_mJ=E,log10_017_low=E*10**(-.17),log10_017_high=E*10**.17,log10_017_error_down=E*(1-10**(-.17)),log10_017_error_up=E*(10**.17-1),abs_003_low=max(0,E-.03),abs_003_high=E+.03,abs_003_percent=100*.03/E))
 pd.DataFrame(examples).to_csv(DEST/'physical_scale_examples.csv',index=False)
 rows=[];representatives=[]
 for method in METHODS:
  for seed in SEEDS:
   files=sorted((DEST/'candidate_runs'/method/f'seed{seed}').glob('TE*.json'));assert len(files)==49
   for path in files:
    r=json.loads(path.read_text())
    for key,s in r['scores'].items():
     rows.append(dict(method=method,seed=seed,goal_id=r['goal_id'],goal_barrier_mJ=r['goal_barrier_mJ'],criterion=key,candidate_points=r['candidate_points'],success=s['success'],pass_count=s['pass_count']))
     if s['selected']:representatives.append(dict(method=method,seed=seed,goal_id=r['goal_id'],criterion=key,goal_barrier_mJ=r['goal_barrier_mJ'],**s['selected']['dimensions'],**{f'predicted_{k}':v for k,v in s['selected']['predicted'].items()},predicted_barrier_mJ=s['selected']['predicted_barrier_mJ'],p_bi_mono=s['selected']['p_bi_mono']))
 df=pd.DataFrame(rows);assert len(df)==735*4;df.to_csv(DEST/'candidate_results.csv',index=False);pd.DataFrame(representatives).to_csv(DEST/'representatives.csv',index=False)
 summary=[]
 for (method,key),q in df.groupby(['method','criterion'],sort=False):
  summary.append(dict(method=method,label=LABELS[method],criterion=key,found=int(q.success.sum()),evaluations=len(q),discovery_rate=float(q.success.mean()),unique_goals_found=int(q.groupby('goal_id').success.any().sum()),generated_candidate_pass_rate=float(q.pass_count.sum()/q.candidate_points.sum()) if method in METHODS[:3] else None))
 pd.DataFrame(summary).to_csv(DEST/'inverse_sensitivity_summary.csv',index=False)
 # Descriptive Test errors are kept separate; no tolerance is chosen by inverse Test success.
 test=pd.read_csv(FORWARD/'test_predictions.csv');test=test[(test.seed==0)&(test.target=='barrier_mJ')].copy();labels=load_data().set_index('geom_id').label;test['label']=test.geom_id.map(labels);test['absolute_error_mJ']=abs(test.actual-test.predicted);test.to_csv(DEST/'descriptive_test_errors.csv',index=False)
 dump(DEST/'audit.json',dict(train_oof_shapes=len(cv),oof_matches_existing_cv_metrics=True,angle_tolerance_deg=ANGLE,displacement_tolerance_mm=DISP,barrier_abs_candidates_mJ=ABS_TOL,forward_weights_unchanged=True,probability_threshold=P_MIN,saved_search_records=735,rescored_records=len(df),new_inverse_search=False,new_forward_for_inverse=False,ansys_executed=False,threshold_basis='Training OOF physical barrier errors; Test sensitivity is descriptive only'))
 print('OOF ERROR SUMMARY',pd.DataFrame(stats).to_string(index=False),flush=True);print('INVERSE SENSITIVITY',pd.DataFrame(summary).to_string(index=False),flush=True)

def main():
 DEST.mkdir(parents=True,exist_ok=True)
 for name,h in PROTOCOL['forward_sha256'].items():assert hashlib.sha256((FORWARD/name).read_bytes()).hexdigest()==h
 dump(DEST/'protocol.json',dict(angle_tolerance_deg=ANGLE,displacement_tolerance_mm=DISP,absolute_barrier_tolerances_mJ=ABS_TOL,reference_log10_tolerance=.17,minimum_predicted_bi_mono_probability=P_MIN,official_geometry_required=True,forward_model='Fixed selected sm_mlp seed0, unchanged',train_oof_recovered_from='Fixed selected joint config0, original seed42 and five original folds',goals='Same actual Test response triples49 x inverse seeds3',ansys_executed=False,new_inverse_search=False))
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(fold_predictions,i) for i in range(5)]):f.result()
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(candidate_sensitivity,m,s) for m in METHODS for s in SEEDS]):f.result()
 summarize()
if __name__=='__main__':main()
