"""Paired reevaluation without barrier acceptance. Original barrier-conditioned pools stay fixed."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
from shared import *
SOURCE=RUNS/'inverse-target-strategies'
DEST=RUNS/'inverse-target-strategies-no-barrier'
STRATEGIES=['1_observed','2_interpolation','4_angle_conditional','5_sparse_boundary','6_independent']
SLABELS={'1_observed':'1. 실제 Test 조합','2_interpolation':'2. 가까운 조합 사이 보간','4_angle_conditional':'4. 각도별 조건부','5_sparse_boundary':'5. 희소 구간·경계','6_independent':'6. 독립 넓은 조합'}
LIMITS=np.array([2.,.25])

def reevaluate(strategy,method,start,stop):
 goals=pd.read_csv(SOURCE/'goals'/f'{strategy}.csv');dest=DEST/'runs'/strategy/method;dest.mkdir(parents=True,exist_ok=True)
 geometry=Geometry()
 for i in range(start,stop):
  row=goals.iloc[i];path=dest/f'{row.goal_id}.json'
  if path.exists():continue
  begun=time.perf_counter();src=SOURCE/'runs'/strategy/method/f'{row.goal_id}.json';old=json.loads(src.read_text());goal=np.array([old['goal'][t] for t in TARGETS])
  with np.load(src.with_suffix('.npz')) as z:u,y,p,geo=(z[k].copy() for k in ['u','predicted','p_bi_mono','geometry_checked'])
  # Baseline pools are bit-identical to the earlier fixed-seed generators.
  reused=0
  if strategy=='1_observed' and method in METHODS[:3]:
   previous=OUT/'runs'/method/'seed0'/f'{row.source_geom_id}.npz'
   if previous.exists():
    with np.load(previous) as z:
     np.testing.assert_array_equal(u,z['u']);np.testing.assert_allclose(y,z['predicted'],rtol=0,atol=0);past=z['geometry_checked']
     both=(geo>=0)&(past>=0);np.testing.assert_array_equal(geo[both],past[both]);take=(geo<0)&(past>=0);geo[take]=past[take];reused=int(take.sum())
  errors=np.abs((y[:,:2]-goal[:2])/LIMITS);distance=errors.max(1);eligible=bound_mask(u)&(p>=np.float32(P_MIN))&(distance<=1)
  indices=np.flatnonzero(eligible);extra=0;selected_index=None
  if method in METHODS[:3]:
   extra_ids=indices[geo[indices]<0]
   if len(extra_ids):geo[extra_ids]=geometry.many(u[extra_ids]).astype(np.int8)
   extra=len(extra_ids);valid=indices[geo[indices]==1]
   if len(valid):selected_index=int(valid[np.argmin(distance[valid])])
  else:
   for j in indices[np.argsort(distance[indices],kind='stable')]:
    if geo[j]<0:geo[j]=int(geometry.one(u[j]));extra+=1
    if geo[j]==1:selected_index=int(j);break
  known=np.zeros(len(u),dtype=np.int8);known[eligible]=-1;mask=eligible&(geo>=0);known[mask]=(geo[mask]==1).astype(np.int8)
  selected=None
  if selected_index is not None:
   j=selected_index;selected=dict(index=j,dimensions=dict(zip(FEATURES,to_phys(u[j]).tolist())),predicted=dict(zip(TARGETS,y[j].tolist())),predicted_barrier_mJ=float(10**y[j,2]),p_bi_mono=float(p[j]),normalized_angle_displacement_error=float(distance[j]))
  assert not old['success'] or selected is not None
  count=int((known==1).sum()) if method in METHODS[:3] else None
  if count is not None:assert count>=old['generated_pass_count']
  tmp=path.with_suffix(f'.{os.getpid()}.tmp.npz');np.savez_compressed(tmp,geometry_checked=geo,known_pass=known);tmp.replace(path.with_suffix('.npz'))
  dump(path,dict(strategy=strategy,method=method,goal_id=old['goal_id'],goal=old['goal'],source_trace=reference(src.with_suffix('.npz')),source_trace_sha256=hashlib.sha256(src.with_suffix('.npz').read_bytes()).hexdigest(),seed=0,candidate_points=len(u),old_success=old['success'],success=selected is not None,old_generated_pass_count=old['generated_pass_count'],generated_pass_count=count,extra_geometry_checks=extra,baseline_geometry_results_reused=reused,selected=selected,seconds=time.perf_counter()-begun))
  if (i+1)%5==0 or i+1==len(goals):print('NO BARRIER',strategy,method,i+1,'/',len(goals),'extra',extra,flush=True)
 return strategy,method,start,stop

def main():
 DEST.mkdir(parents=True,exist_ok=True);src=json.loads((SOURCE/'protocol.json').read_text())
 config=dict(source=reference(SOURCE),goal_counts=src['goal_counts'],seed=0,models_retrained=False,models_resampled=False,optimizers_rerun=False,barrier_still_in_generator_input=True,barrier_in_acceptance=False,active_tolerances=dict(angle_deg=2.,deformation_mm=.25),p_min=P_MIN,official_geometry_required=True,stress_filter=False,ansys_executed=False,forward_sha256=src['forward_sha256'],source_goal_hashes=src['goals_sha256'],ranking='minimum maximum angle/displacement tolerance-normalized error among official-valid passing candidates')
 for name,h in config['forward_sha256'].items():assert hashlib.sha256((FORWARD/name).read_bytes()).hexdigest()==h
 if (DEST/'protocol.json').exists():assert json.loads((DEST/'protocol.json').read_text())==config
 dump(DEST/'protocol.json',config);dump(DEST/'status.json',dict(phase='rescoring',updated=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),ansys=False,planned_evaluations=2245))
 tasks=[(strategy,method,start,min(start+5,n)) for start in range(0,100,5) for strategy,n in src['goal_counts'].items() if start<n for method in METHODS]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(reevaluate,*args) for args in tasks]):f.result()
 import importlib.util
 spec=importlib.util.spec_from_file_location('no_barrier_report',Path(__file__).with_name('no_barrier_report.py'));r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);r.main()
 dump(DEST/'status.json',dict(phase='complete',updated=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),ansys=False))
 print('NO-BARRIER PAIRED EVALUATION COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:dump(DEST/'status.json',dict(phase='failed',error=repr(e),ansys=False));raise
