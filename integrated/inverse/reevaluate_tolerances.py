"""Paired tolerance sensitivity on saved candidates; no retraining/search/ANSYS."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
from shared import *
DEST=RUNS/'unified-inverse-relaxed'
NEW_TOL=np.array([1.8,.25,.17])

def reevaluate(method,seed):
 geometry=Geometry();folder=DEST/'runs'/method/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
 for i,path in enumerate(sorted((OUT/'runs'/method/f'seed{seed}').glob('TE*.json'))):
  target=folder/path.name
  if target.exists():continue
  old=json.loads(path.read_text());goal=np.array([old['goal'][k] for k in TARGETS])
  with np.load(path.with_suffix('.npz')) as z:
   u,y,p,passed,geo=(z[k].copy() for k in ['u','predicted','p_bi_mono','passed','geometry_checked'])
  strict=bound_mask(u)&(p>=np.float32(P_MIN))&(np.abs((y-goal)/TOL).max(1)<=1)&(geo==1)
  np.testing.assert_array_equal(strict,passed)
  err=np.abs((y-goal)/NEW_TOL);distance=err.max(1)
  promising=bound_mask(u)&(p>=np.float32(P_MIN))&(distance<=1)
  extra=np.flatnonzero(promising&(geo<0))
  if len(extra):geo[extra]=geometry.many(u[extra]).astype(np.int8)
  relaxed=promising&(geo==1)
  assert (relaxed[passed]).all() and (geo[promising]>=0).all()
  selected=None
  if relaxed.any():
   j=np.flatnonzero(relaxed)[np.argmin(distance[relaxed])]
   selected=dict(index=int(j),dimensions=dict(zip(FEATURES,to_phys(u[j]).tolist())),predicted=dict(zip(TARGETS,y[j].tolist())),predicted_barrier_mJ=float(10**y[j,2]),p_bi_mono=float(p[j]),normalized_max_error=float(distance[j]))
  result=dict(method=method,seed=seed,goal_id=old['goal_id'],goal=old['goal'],goal_barrier_mJ=old['goal_barrier_mJ'],candidate_points=len(u),strict_success=old['success'],relaxed_success=bool(relaxed.any()),strict_pass_count=int(passed.sum()),relaxed_pass_count=int(relaxed.sum()),extra_geometry_checks=len(extra),selected=selected,source_trace=reference(path.with_suffix('.npz')),source_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest())
  trace=target.with_suffix('.npz');tmp=trace.with_suffix(f'.{os.getpid()}.tmp.npz')
  np.savez_compressed(tmp,passing_indices=np.flatnonzero(relaxed),extra_geometry_indices=extra,extra_geometry_results=geo[extra]);tmp.replace(trace)
  dump(target,result)
  if (i+1)%10==0 or i==48:print('RESCORE',method,seed,i+1,'/49',flush=True)
 return method,seed

def report():
 rows=[];reps=[];fwd=Forward();rep_u=[];rep_y=[];rep_p=[]
 for method in METHODS:
  for seed in SEEDS:
   files=sorted((DEST/'runs'/method/f'seed{seed}').glob('TE*.json'));assert len(files)==49
   for path in files:
    r=json.loads(path.read_text());rows.append({k:v for k,v in r.items() if k not in ['selected','goal','source_sha256']})
    if r['selected']:
     s=r['selected'];dims=s['dimensions'];pp=dict(dims,t_w=G.T_W)
     assert all(G.check_constraints(pp).values())
     assert np.float32(s['p_bi_mono'])>=np.float32(P_MIN) and s['normalized_max_error']<=1
     rep_u.append(normalized([dims[k] for k in FEATURES]));rep_y.append([s['predicted'][k] for k in TARGETS]);rep_p.append(s['p_bi_mono'])
     reps.append(dict(candidate_id=f"RELAXED_{method}_s{seed}_{r['goal_id']}",method=method,seed=seed,goal_id=r['goal_id'],**{f'goal_{k}':v for k,v in r['goal'].items()},goal_barrier_mJ=r['goal_barrier_mJ'],**dims,**{f'predicted_{k}':v for k,v in s['predicted'].items()},predicted_barrier_mJ=s['predicted_barrier_mJ'],p_bi_mono=s['p_bi_mono'],normalized_max_error=s['normalized_max_error']))
 y,p=fwd.predict(np.array(rep_u));np.testing.assert_allclose(y,rep_y,rtol=1e-6,atol=5e-5);np.testing.assert_allclose(p,rep_p,rtol=1e-6,atol=1e-6)
 df=pd.DataFrame(rows);assert len(df)==735
 old_summary=pd.read_csv(OUT/'summary.csv');summary=[];byseed=[]
 for m in METHODS:
  q=df[df.method==m];gen=m in METHODS[:3];old=old_summary[old_summary.method==m].iloc[0]
  assert q.strict_success.sum()==old.found_goal_seed
  assert q.strict_pass_count.sum()==old.total_passing_points
  summary.append(dict(method=m,label=LABELS[m],goal_seed_total=len(q),strict_found=int(q.strict_success.sum()),relaxed_found=int(q.relaxed_success.sum()),strict_discovery_rate=float(q.strict_success.mean()),relaxed_discovery_rate=float(q.relaxed_success.mean()),newly_found=int((q.relaxed_success&~q.strict_success).sum()),unique_goals_strict=int(q.groupby('goal_id').strict_success.any().sum()),unique_goals_relaxed=int(q.groupby('goal_id').relaxed_success.any().sum()),strict_generated_candidate_pass_rate=float(q.strict_pass_count.sum()/q.candidate_points.sum()) if gen else None,relaxed_generated_candidate_pass_rate=float(q.relaxed_pass_count.sum()/q.candidate_points.sum()) if gen else None,total_candidates=int(q.candidate_points.sum()),strict_passing_points=int(q.strict_pass_count.sum()),relaxed_passing_points=int(q.relaxed_pass_count.sum()),extra_geometry_checks=int(q.extra_geometry_checks.sum())))
  for seed,s in q.groupby('seed'):byseed.append(dict(method=m,seed=int(seed),goals=len(s),strict_found=int(s.strict_success.sum()),relaxed_found=int(s.relaxed_success.sum())))
 summary=pd.DataFrame(summary).sort_values('relaxed_discovery_rate',ascending=False)
 summary.to_csv(DEST/'summary.csv',index=False);df.to_csv(DEST/'goal_results.csv',index=False);pd.DataFrame(reps).to_csv(DEST/'representatives.csv',index=False);pd.DataFrame(byseed).to_csv(DEST/'metrics_by_seed.csv',index=False)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 english={'jw_cvae':'Jinwoo CVAE','sm_cvae':'Seungmin CVAE','conditional_gmm':'Conditional GMM','differential_evolution':'Forward + DE','slsqp':'Forward + SLSQP'}
 fig,axes=plt.subplots(1,2,figsize=(15.5,5.5));fig.subplots_adjust(left=.14,right=.96,bottom=.13,top=.78,wspace=.57)
 for ax,data,a,b,title in [(axes[0],summary,'strict_discovery_rate','relaxed_discovery_rate','Goal discovery (49 goals x 3 seeds)'),(axes[1],summary[summary.method.isin(METHODS[:3])],'strict_generated_candidate_pass_rate','relaxed_generated_candidate_pass_rate','Raw generated candidate pass rate')]:
  x=np.arange(len(data));ax.barh(x-.18,100*data[a],.32,label='Original',color='#8195a9');ax.barh(x+.18,100*data[b],.32,label='Slightly relaxed',color='#246fba')
  for container in ax.containers:ax.bar_label(container,fmt='%.1f%%',fontsize=9,padding=4)
  ax.set_yticks(x,[english[m] for m in data.method]);ax.invert_yaxis();ax.set_xlim(0,112);ax.set_xticks([0,20,40,60,80,100]);ax.set_title(title,pad=15);ax.set_xlabel('Percent');ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);ax.spines[['right','top']].set_visible(False)
 fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
 fig.suptitle('Same saved candidates; only response tolerances changed | No new ANSYS')
 fig.savefig(DEST/'comparison.png',dpi=170,bbox_inches='tight');plt.close(fig)
 body='<h1>역설계 허용오차 소폭 완화 비교</h1><p><strong>같은 저장 후보를 다시 판정한 민감도 분석입니다. 재학습·재생성·최적화 재탐색·ANSYS 해석은 하지 않았습니다.</strong></p><p>각도 ±1.7° → ±1.8°, 변위 ±0.23 → ±0.25 mm, log₁₀ 장벽 ±0.16 → ±0.17. 전환 예측 확률 0.9 이상과 공식 형상 제약은 유지했습니다. 기존 Test 목표 49개와 시드 0·1·2를 그대로 사용했습니다.</p>'
 body+='<table><tr><th>방법</th><th>기존 발견률</th><th>완화 발견률</th><th>추가 성공</th><th>기존 개별 생성 통과율</th><th>완화 개별 생성 통과율</th></tr>'
 for _,r in summary.iterrows():
  pct=lambda v:'—' if pd.isna(v) else f'{100*v:.1f}%'
  body+=f'<tr><td>{r.label}</td><td>{r.strict_found}/147 ({pct(r.strict_discovery_rate)})</td><td>{r.relaxed_found}/147 ({pct(r.relaxed_discovery_rate)})</td><td>{r.newly_found}</td><td>{pct(r.strict_generated_candidate_pass_rate)}</td><td>{pct(r.relaxed_generated_candidate_pass_rate)}</td></tr>'
 body+='</table><img src="comparison.png"><p>최적화 방법은 원래의 엄격한 허용오차로 수행한 탐색 기록을 재판정했습니다. 완화된 오차를 목적함수에 넣어 최적화를 다시 실행한 결과는 아닙니다. 기존 검사에서 확인되지 않았던 새 통과 가능 후보에만 공식 정밀 형상 검사를 추가했습니다. 기존 통과 후보가 모두 새 기준에서도 통과함을 검증했고, 완화 기준의 대표 후보 형상과 Forward 예측을 다시 확인했습니다. 성공률은 대리모델 기준이며 실제 해석 성공률이 아닙니다.</p>'
 (DEST/'평가 결과.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>허용오차 비교</title><style>body{font-family:Arial,"Apple SD Gothic Neo",sans-serif;max-width:1150px;margin:40px auto;line-height:1.7;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ccd3db;text-align:right}th{background:#e9eff8}img{width:100%;margin-top:30px}</style>'+body,encoding='utf-8')
 audit=dict(records=len(df),representatives_rechecked=len(reps),strict_results_reproduced=True,strict_passing_candidates_subset_of_relaxed=True,official_geometry_rechecked=True,representative_predictions_recomputed=True,forward_sha256_unchanged=True,ansys_executed=False,new_search=False,retrained=False)
 dump(DEST/'integrity_audit.json',audit);print(summary.to_string(index=False),flush=True)

def main():
 DEST.mkdir(parents=True,exist_ok=True)
 for name,h in PROTOCOL['forward_sha256'].items():assert hashlib.sha256((FORWARD/name).read_bytes()).hexdigest()==h
 config=dict(source=reference(OUT),old_tolerances=dict(zip(TARGETS,TOL.tolist())),new_tolerances=dict(zip(TARGETS,NEW_TOL.tolist())),p_min=P_MIN,geometry_unchanged=True,goals=49,seeds=SEEDS,methods=METHODS,mode='paired rescoring of the exact saved candidates; no new search or training',ansys_executed=False)
 if (DEST/'protocol.json').exists():assert json.loads((DEST/'protocol.json').read_text())==config
 dump(DEST/'protocol.json',config);dump(DEST/'status.json',dict(phase='rescoring',updated=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()))
 with ProcessPoolExecutor(max_workers=3) as pool:
  for task in as_completed([pool.submit(reevaluate,m,s) for m in METHODS for s in SEEDS]):print('DONE',task.result(),flush=True)
 report();dump(DEST/'status.json',dict(phase='complete',updated=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()))
if __name__=='__main__':main()
