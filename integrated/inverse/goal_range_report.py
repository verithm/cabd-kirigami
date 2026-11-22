"""Report and integrity checks for the five-strategy physical-tolerance benchmark."""
from goal_range_benchmark import *
import html

def wilson(k,n):
 z=1.959963984540054;den=1+z*z/n;center=(k/n+z*z/(2*n))/den;half=z*np.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den
 return center-half,center+half

def collect_and_audit():
 protocol=json.loads((DEST/'protocol.json').read_text());rows=[];reps=[];allrep=[];seen_shapes=set();fwd=Forward();baseline_checks=0;goalstats=[]
 original=load_data();original_responses=original[original.is_reg][TARGETS].to_numpy(float)
 for strategy in STRATEGIES:
  goals=pd.read_csv(DEST/'goals'/f'{strategy}.csv');assert len(goals)==protocol['goal_counts'][strategy]
  if strategy!='1_observed':assert (np.abs(goals[TARGETS].to_numpy(float)[:,None,:]-original_responses[None,:,:]).max(2).min(1)>1e-8).all()
  goalstats.append(dict(strategy=strategy,label=STRATEGY_LABELS[strategy],goals=len(goals),angle_min=goals.angle_deg.min(),angle_max=goals.angle_deg.max(),displacement_min=goals.deformation_mm.min(),displacement_max=goals.deformation_mm.max(),barrier_min_mJ=goals.goal_barrier_mJ.min(),barrier_max_mJ=goals.goal_barrier_mJ.max(),nearest_train_tolerance_distance_median=goals.nearest_train_tolerance_distance.median(),within_training_response_tolerances_fraction=goals.within_train_response_tolerances.mean()))
  for method in METHODS:
   files=sorted((DEST/'runs'/strategy/method).glob('*.json'));assert len(files)==len(goals),(strategy,method,len(files))
   assert {p.stem for p in files}==set(goals.goal_id)
   lookup=goals.set_index('goal_id')
   for path in files:
    r=json.loads(path.read_text());assert r['strategy']==strategy and r['method']==method and r['seed']==0
    g=np.array([r['goal'][t] for t in TARGETS]);np.testing.assert_allclose(g,lookup.loc[r['goal_id'],TARGETS].to_numpy(float),atol=1e-12)
    with np.load(path.with_suffix('.npz')) as z:
     u,y,p,geo,known,errors=(z[k] for k in ['u','predicted','p_bi_mono','geometry_checked','known_pass','normalized_error'])
     assert len(u)==r['forward_evaluations']<=BUDGET and np.isfinite(u).all() and np.isfinite(y).all() and np.isfinite(p).all()
     e=error_matrix(y,g);np.testing.assert_allclose(errors,e,rtol=1e-10,atol=1e-10)
     promising=bound_mask(u)&(p>=np.float32(P_MIN))&(e.max(1)<=1)
     assert int(promising.sum())==r['surrogate_joint_count']
     if method in METHODS[:3]:
      assert (geo[promising]>=0).all();np.testing.assert_array_equal(known,promising&(geo==1));assert int((known==1).sum())==r['generated_pass_count']
     if not r['success']:assert (geo[promising]==0).all()
     else:
      s=r['selected'];j=s['index'];assert promising[j] and geo[j]==1 and known[j]==1
      assert not ((e.max(1)<s['normalized_max_error'])&promising&(geo<0)).any(),'A better unknown-geometry point cannot be skipped'
      dims=to_phys(u[j]);np.testing.assert_allclose(dims,[s['dimensions'][f] for f in FEATURES],atol=1e-12)
      key=tuple(dims)
      if key not in seen_shapes:assert all(G.check_constraints(dict(zip(FEATURES,dims),t_w=G.T_W)).values());seen_shapes.add(key)
      np.testing.assert_allclose(y[j],[s['predicted'][t] for t in TARGETS],atol=1e-12)
      allrep.append((u[j],y[j],p[j]))
      reps.append(dict(candidate_id=f"TARGET_{strategy}_{method}_{r['goal_id']}",strategy=strategy,method=method,goal_id=r['goal_id'],**{f'goal_{t}':v for t,v in r['goal'].items()},goal_barrier_mJ=r['goal_barrier_mJ'],**s['dimensions'],**{f'predicted_{t}':v for t,v in s['predicted'].items()},predicted_barrier_mJ=s['predicted_barrier_mJ'],p_bi_mono=s['p_bi_mono'],normalized_max_error=s['normalized_max_error']))
     row={k:r[k] for k in ['strategy','method','goal_id','seed','forward_evaluations','success','generated_pass_count','generated_pass_fraction','surrogate_joint_count','official_geometry_checks','search_seconds','filter_seconds','total_seconds']}
     for j,t in enumerate(['angle','displacement','barrier_mJ']):row[f'marginal_{t}_pass_count']=int((e[:,j]<=1).sum())
     row['marginal_stability_pass_count']=int((p>=np.float32(P_MIN)).sum());row['bounds_pass_count']=int(bound_mask(u).sum());row['nearest_train_tolerance_distance']=float(lookup.loc[r['goal_id'],'nearest_train_tolerance_distance']);row['subtype']=str(lookup.loc[r['goal_id'],'subtype']);rows.append(row)
     if strategy=='1_observed' and method in METHODS[:3]:
      source=str(lookup.loc[r['goal_id'],'source_geom_id']);oldpath=RUNS/'barrier-tolerance-review/candidate_runs'/method/'seed0'/f'{source}.json'
      if oldpath.exists():
       old=json.loads(oldpath.read_text())['scores']['abs_0.03mJ'];assert r['generated_pass_count']==old['pass_count'] and r['success']==old['success'];baseline_checks+=1
 df=pd.DataFrame(rows);assert len(df)==2245
 if allrep:
  y,p=fwd.predict(np.array([r[0] for r in allrep]));np.testing.assert_allclose(y,np.array([r[1] for r in allrep]),rtol=1e-6,atol=5e-5);np.testing.assert_allclose(p,np.array([r[2] for r in allrep]),rtol=1e-6,atol=1e-6)
 for name,h in protocol['forward_sha256'].items():assert hashlib.sha256((FORWARD/name).read_bytes()).hexdigest()==h
 for strategy,h in protocol['goals_sha256'].items():assert hashlib.sha256((DEST/'goals'/f'{strategy}.csv').read_bytes()).hexdigest()==h
 df.to_csv(DEST/'goal_results.csv',index=False);pd.DataFrame(reps).to_csv(DEST/'representatives.csv',index=False);pd.DataFrame(goalstats).to_csv(DEST/'goal_distribution_summary.csv',index=False)
 audit=dict(goal_method_records=len(df),total_targets=449,strategies=5,methods=5,seed=0,baseline_generated_records_match_prior_physical_rescoring=baseline_checks,representatives_verified=len(allrep),unique_representative_geometries_original_checker_verified=len(seen_shapes),forward_predictions_reloaded_and_verified=True,forward_weights_unchanged=True,goal_hashes_unchanged=True,all_generative_promising_geometry_checks_complete=True,optimizer_best_passing_representatives_exact=True,new_targets_do_not_exactly_match_any_original_regression_response=True,ansys_executed=False)
 dump(DEST/'integrity_audit.json',audit)
 return df,pd.DataFrame(goalstats)

def summaries(df):
 rows=[];subtypes=[]
 for (strategy,method),q in df.groupby(['strategy','method'],sort=False):
  low,high=wilson(int(q.success.sum()),len(q));gen=method in METHODS[:3]
  rows.append(dict(strategy=strategy,strategy_label=STRATEGY_LABELS[strategy],method=method,method_label=LABELS[method],targets=len(q),found=int(q.success.sum()),goal_discovery_rate=float(q.success.mean()),descriptive_wilson_low=low,descriptive_wilson_high=high,generated_candidate_pass_rate=float(q.generated_pass_count.sum()/q.forward_evaluations.sum()) if gen else None,mean_forward_evaluations=q.forward_evaluations.mean(),mean_search_seconds=q.search_seconds.mean(),mean_filter_seconds=q.filter_seconds.mean(),median_total_seconds=q.total_seconds.median(),total_passing_generated_candidates=int(q.generated_pass_count.sum()) if gen else None,total_evaluated_points=int(q.forward_evaluations.sum())))
 for (strategy,method,subtype),q in df.groupby(['strategy','method','subtype'],sort=False):subtypes.append(dict(strategy=strategy,method=method,subtype=subtype,targets=len(q),found=int(q.success.sum()),discovery_rate=q.success.mean(),generated_candidate_pass_rate=q.generated_pass_count.sum()/q.forward_evaluations.sum() if method in METHODS[:3] else None))
 result=pd.DataFrame(rows);result.to_csv(DEST/'summary.csv',index=False);pd.DataFrame(subtypes).to_csv(DEST/'subtype_summary.csv',index=False)
 reach=[];detail=[]
 for strategy,q in df.groupby('strategy',sort=False):
  pivot=q.pivot(index='goal_id',columns='method',values='success');union=pivot.any(axis=1);gen=pivot[METHODS[:3]].any(axis=1);opt=pivot[METHODS[3:]].any(axis=1)
  reach.append(dict(strategy=strategy,targets=len(pivot),any_method_found=int(union.sum()),all_methods_failed=int((~union).sum()),optimization_only_found=int((opt&~gen).sum()),generation_only_found=int((gen&~opt).sum()),all_methods_found=int(pivot.all(axis=1).sum())))
  for goal_id,row in pivot.iterrows():detail.append(dict(strategy=strategy,goal_id=goal_id,any_method_found=bool(row.any()),**row.to_dict()))
 pd.DataFrame(reach).to_csv(DEST/'surrogate_reachability_summary.csv',index=False);pd.DataFrame(detail).to_csv(DEST/'surrogate_reachability_by_goal.csv',index=False)
 return result

def plots(summary):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 labels=['1 Actual Test','2 Neighbor interpolation','4 Angle-conditional','5 Sparse / boundary','6 Independent wide']
 english={'jw_cvae':'Jinwoo CVAE','sm_cvae':'Seungmin CVAE','conditional_gmm':'Conditional GMM','differential_evolution':'Forward + DE','slsqp':'Forward + SLSQP'}
 fig,axes=plt.subplots(1,2,figsize=(15.5,6));fig.subplots_adjust(left=.17,right=.95,bottom=.28,top=.81,wspace=.2)
 for ax,methods,column,title in [(axes[0],METHODS,'goal_discovery_rate','Goals with at least one passing candidate'),(axes[1],METHODS[:3],'generated_candidate_pass_rate','Raw generated candidate pass rate')]:
  a=np.array([[summary[(summary.strategy==s)&(summary.method==m)].iloc[0][column]*100 for m in methods] for s in STRATEGIES]);im=ax.imshow(a,vmin=0,vmax=100,cmap='RdYlGn',aspect='auto')
  ax.set_xticks(range(len(methods)),[english[m] for m in methods],rotation=30,ha='right');ax.set_yticks(range(5),labels if ax is axes[0] else ['']*5);ax.set_title(title,pad=15,fontsize=12)
  for i in range(5):
   for j in range(len(methods)):
    r=summary[(summary.strategy==STRATEGIES[i])&(summary.method==methods[j])].iloc[0];t=f'{a[i,j]:.1f}%' if column!='goal_discovery_rate' else f'{a[i,j]:.1f}%\n{int(r.found)}/{int(r.targets)}'
    ax.text(j,i,t,ha='center',va='center',fontsize=10,color='white' if a[i,j]<17 or a[i,j]>83 else '#10202a',fontweight='bold')
 fig.colorbar(im,ax=axes,fraction=.025,pad=.03,label='Percent (same 0–100% scale)');fig.suptitle('Target strategy comparison | angle +/-2 deg, displacement +/-0.25 mm, barrier +/-0.03 mJ\nFixed seed 0; 4,096 Forward evaluations per goal maximum; No ANSYS',fontsize=13)
 fig.savefig(DEST/'strategy_comparison.png',dpi=180,bbox_inches='tight');fig.savefig(DEST/'strategy_comparison.svg',bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,5));colors=['#667787','#3479b5','#329966','#ce8935','#b75774']
 for s,label,color in zip(STRATEGIES,labels,colors):
  q=pd.read_csv(DEST/'goals'/f'{s}.csv');axes[0].scatter(q.angle_deg,q.deformation_mm,s=13,alpha=.65,color=color,label=label);axes[1].scatter(q.angle_deg,q.goal_barrier_mJ,s=13,alpha=.65,color=color)
 axes[0].set_ylabel('Target second-well displacement (mm)');axes[1].set_ylabel('Target reverse barrier (mJ)')
 for ax in axes:ax.set_xlabel('Target stable angle (degrees)');ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);fig.suptitle('Evaluated target combinations only (no training-point overlay)');fig.tight_layout();fig.savefig(DEST/'target_combinations.png',dpi=170);plt.close(fig)

def main():
 df,gstats=collect_and_audit();s=summaries(df);plots(s)
 descriptions={'1_observed':'실제 Test 전환 형상 49개의 응답 조합을 그대로 사용합니다. 목표 치수는 생성 모델 학습과 최적화 시작점에 사용하지 않습니다.','2_interpolation':'Train 전환 응답에서 표준화한 각도·변위·log장벽 공간의 가까운 이웃 두 개를 고릅니다. 원래 단위의 각도·변위·장벽을 25–75% 비율로 보간한 새 목표 100개입니다.','4_angle_conditional':'Train 각도 범위를 100개 구간으로 나누어 각도를 하나씩 뽑습니다. 상관관계를 반영한 공동 Gaussian KDE를 각도에 조건화해 변위와 log장벽을 함께 샘플링합니다. Train의 각 변수 최소·최대 범위를 벗어나면 다시 뽑습니다.','5_sparse_boundary':'Train 공동 KDE에서 뽑은 후보 중, 이웃과 거리가 큰 희소 목표 50개와 주변 범위의 경계에 가까운 목표 50개를 골랐습니다. 유사 목표의 중복을 줄이기 위해 표준화 공간에서 최소 간격을 적용했습니다.','6_independent':'각도·변위·원래 단위 장벽 각각의 Train 최소·최대 범위에서 독립 Latin hypercube로 100개를 뽑았습니다. 장벽은 mJ 기준 균등 분할이며 log-균등 분포가 아닙니다. 변수 간 상관관계를 강제하지 않습니다.'}
 body=['<h1>목표 조건 생성 방법에 따른 역설계 비교</h1>','<p><strong>새 ANSYS 해석 없이 고정 Forward와 공식 형상 제약으로 평가했습니다. 안정각 ±2°, 변위 ±0.25 mm, 원래 장벽 ±0.03 mJ, 전환 예측 확률 ≥0.9입니다. 응력은 제외했습니다.</strong></p>','<p>실제 Test 목표 49개와 새 목표 4종 각 100개, 총 449개를 5개 역설계 방법으로 평가했습니다(2,245회). 시드는 0 하나로 고정했습니다. 과거의 49목표×3시드 결과와 분모가 다릅니다. 목표당 생성 후보 4,096개 또는 최적화의 Forward 평가 최대 4,096개를 적용했습니다.</p>','<h2>목표 발견률</h2>','<p>공통 조건과 형상 제약을 모두 만족하는 후보를 하나 이상 찾은 목표 비율입니다. 같은 Forward로 선별하고 채점한 대리모델 기준이며 독립적인 실제 성능 검증이 아닙니다.</p>','<table><tr><th>목표 유형</th>'+''.join(f'<th>{LABELS[m]}</th>' for m in METHODS)+'</tr>']
 for st in STRATEGIES:
  q=s[s.strategy==st].set_index('method');body.append(f'<tr><td>{STRATEGY_LABELS[st]}</td>'+''.join(f'<td>{100*q.loc[m,"goal_discovery_rate"]:.1f}% ({int(q.loc[m,"found"])}/{int(q.loc[m,"targets"])})</td>' for m in METHODS)+'</tr>')
 body+=['</table>','<h2>개별 생성 후보 통과율</h2>','<p>필터 전 전체 4,096개 생성 후보가 분모입니다. 범위 밖 후보도 실패입니다. 적응적으로 수집한 최적화 점의 비율과는 구분하므로 최적화 방법의 개별 후보 비율은 보고하지 않습니다.</p>','<table><tr><th>목표 유형</th>'+''.join(f'<th>{LABELS[m]}</th>' for m in METHODS[:3])+'</tr>']
 for st in STRATEGIES:
  q=s[s.strategy==st].set_index('method');body.append(f'<tr><td>{STRATEGY_LABELS[st]}</td>'+''.join(f'<td>{100*q.loc[m,"generated_candidate_pass_rate"]:.2f}%</td>' for m in METHODS[:3])+'</tr>')
 body+=['</table>','<img src="strategy_comparison.png">','<h2>목표를 만든 방법</h2>']
 for st in STRATEGIES:body.append(f'<p><strong>{STRATEGY_LABELS[st]}</strong>: {descriptions[st]}</p>')
 body+=['<p>새 목표 분포는 Train bi→mono 응답 394개로만 구성했습니다. 새 네 유형의 목표가 기존 회귀 데이터 전체(Train/Test)의 응답과 정확히 일치하지 않음을 확인했습니다. 다만 수치가 새롭다는 것과 물리적으로 실현 가능하다는 것은 다릅니다. 기존 응답의 허용오차 범위 안에 들어오는 가까운 목표도 있으므로 아래 거리 통계를 함께 제공합니다.</p>','<h2>범위와 기존 응답까지의 거리</h2>','<p>거리는 각도·변위·장벽 차이를 각각 2°·0.25 mm·0.03 mJ로 나눈 뒤, 세 값 중 최댓값을 사용합니다. Train 전환 응답 전체에 대해 계산한 최솟값이 1 이하면 이미 알려진 Train 응답 중 하나가 목표 허용오차 안에 있습니다. 이는 형상 동일성이나 실제 물리적 가능성을 판정하는 지표는 아닙니다.</p>','<table><tr><th>유형</th><th>각도 범위</th><th>변위 범위</th><th>장벽 범위 mJ</th><th>최근접 거리 중앙값</th><th>Train 응답 허용범위 안의 목표</th></tr>']
 for _,r in gstats.iterrows():body.append(f'<tr><td>{r.label}</td><td>{r.angle_min:.2f}–{r.angle_max:.2f}°</td><td>{r.displacement_min:.2f}–{r.displacement_max:.2f} mm</td><td>{r.barrier_min_mJ:.5f}–{r.barrier_max_mJ:.5f}</td><td>{r.nearest_train_tolerance_distance_median:.2f}</td><td>{100*r.within_training_response_tolerances_fraction:.1f}%</td></tr>')
 body+=['</table>','<img src="target_combinations.png">','<h2>공통 Forward상 탐색 가능성 진단</h2>','<p>다섯 방법 중 하나라도 후보를 찾았으면 대리모델 기준에서는 도달 가능한 목표를 발견한 것입니다. 모든 방법이 실패한 목표도 실제로 불가능하다고 증명된 것은 아닙니다.</p>','<table><tr><th>유형</th><th>하나 이상의 방법에서 발견</th><th>모든 방법 실패</th><th>최적화에서만 발견</th></tr>']
 for _,r in pd.read_csv(DEST/'surrogate_reachability_summary.csv').iterrows():body.append(f'<tr><td>{STRATEGY_LABELS[r.strategy]}</td><td>{r.any_method_found}/{r.targets}</td><td>{r.all_methods_failed}</td><td>{r.optimization_only_found}</td></tr>')
 body+=['</table>','<h2>학습·탐색·형상 검사</h2>','<p>기존에 검증으로 선정해 학습한 진우 CVAE·승민 CVAE·조건부 GMM의 시드 0 가중치를 고정했습니다. 이번 목표 세트로 재학습하거나 설정을 다시 고르지 않았습니다. Forward도 선정된 승민 MLP 시드 0으로 고정했으며 분류와 회귀 각각 신경망 5개 평균입니다. 기존 Forward 가중치 해시가 유지됐습니다.</p>','<p>두 생성 모델과 GMM의 조건 장벽은 학습한 방식대로 log₁₀로 넣고, 평가할 때 mJ로 복원합니다. DE와 SLSQP는 기존 로그 기준 탐색 기록을 재판정한 것이 아니라, 새 mJ 오차를 목적함수에 넣고 다시 탐색했습니다. DE는 초기 집단 128개×32회=4,096점, SLSQP는 동일한 Sobol 시작점 8개에서 시작점당 최대 512점으로 유한차분 탐색합니다. 먼저 종료하면 평가 수가 적을 수 있고, 유한차분 점도 예산에 포함합니다.</p>','<p>생성 모델의 공동 통과 가능 후보는 모두 공식 정밀 형상 검사를 합니다. 최적화는 개별 점 통과율을 보고하지 않으므로 응답 오차 순으로 형상을 검사해 가장 좋은 통과 대표 후보를 찾으면 중단합니다. 더 좋은 후보를 건너뛰지 않았음을 감사했습니다. 이 방식은 최적화의 목표 발견 여부나 최선 대표 후보를 바꾸지 않으면서 불필요한 전수 형상 검사를 줄입니다.</p>','<p>시간은 동일 노트북에서 세 프로세스를 병렬 실행한 기록입니다. 생성 모델의 시간에는 개별 후보 통과율을 정확히 세기 위한 전수 검사도 포함되므로 단순히 후보 한 개를 찾을 때의 시간과 같지 않습니다. summary.csv에 탐색/Forward 시간과 형상 검사 시간을 분리했습니다.</p>','<h2>해석할 때의 한계</h2>','<p>독립 넓은 조합은 특히 실제로 불가능한 목표를 포함할 수 있습니다. 실패했다고 물리적으로 불가능함이 증명되거나 모델만의 실패라고 단정할 수 없습니다. 보간이나 조건부 샘플링도 실제 가능성을 보장하지 않습니다. 한 시드의 결과이므로 시드 변동성을 평가한 실험이 아닙니다. summary.csv의 Wilson 구간은 목표별 성공 비율의 설명용 구간이며, 의존적인 합성 목표나 방법 간 차이에 대한 엄밀한 유의성 검정은 아닙니다.</p>','<h2>저장 파일</h2>','<p>summary.csv: 25개 방법/유형 집계 · goal_results.csv: 2,245개 평가 기록 · goals/: 목표값과 생성 출처 · goal_distribution_summary.csv: 범위와 거리 · representatives.csv: 통과 목표별 최선 치수 · subtype_summary.csv: 희소/경계 등 하위 유형 · runs/: 모든 평가점과 예측·검사 여부 · integrity_audit.json: 무결성 점검.</p>']
 (DEST/'평가 결과.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>목표 조건 생성 방법 비교</title><style>body{font-family:Arial,"Apple SD Gothic Neo",sans-serif;max-width:1250px;margin:35px auto;padding:0 22px;line-height:1.7;color:#18232d}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px;border:1px solid #cbd5df;text-align:right}th{background:#eaf1f8}td:first-child,th:first-child{text-align:left}img{width:100%;margin:25px 0}h2{margin-top:35px}</style>'+''.join(body),encoding='utf-8')
 print(s[['strategy','method','found','targets','goal_discovery_rate','generated_candidate_pass_rate']].to_string(index=False),flush=True)
if __name__=='__main__':main()
