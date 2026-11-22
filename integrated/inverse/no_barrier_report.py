"""Audit and report same-pool evaluation with barrier acceptance removed."""
from evaluate_no_barrier import *

def collect():
 records=[];representatives=[];verify=[];seen=set();reused=0;extra=0
 protocol=json.loads((DEST/'protocol.json').read_text())
 for strategy in STRATEGIES:
  goals=pd.read_csv(SOURCE/'goals'/f'{strategy}.csv').set_index('goal_id')
  for method in METHODS:
   paths=sorted((DEST/'runs'/strategy/method).glob('*.json'));assert len(paths)==len(goals)
   assert {p.stem for p in paths}==set(goals.index)
   for path in paths:
    r=json.loads(path.read_text());source=resolve_reference(r['source_trace']);old=json.loads(source.with_suffix('.json').read_text())
    assert r['strategy']==strategy and r['method']==method and r['seed']==0
    assert r['old_success']==old['success'] and (not r['old_success'] or r['success'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==r['source_trace_sha256']
    with np.load(source) as z:u,y,p,old_known=(z[k] for k in ['u','predicted','p_bi_mono','known_pass'])
    with np.load(path.with_suffix('.npz')) as z:geo,known=(z[k] for k in ['geometry_checked','known_pass'])
    assert len(u)==r['candidate_points']<=BUDGET
    goal=np.array([r['goal'][t] for t in TARGETS]);e=np.abs((y[:,:2]-goal[:2])/LIMITS);distance=e.max(1)
    eligible=bound_mask(u)&(p>=np.float32(P_MIN))&(distance<=1)
    if method in METHODS[:3]:
     assert (geo[eligible]>=0).all();np.testing.assert_array_equal(known,eligible&(geo==1));assert (known[old_known==1]==1).all()
     assert int((known==1).sum())==r['generated_pass_count']>=r['old_generated_pass_count']
    if r['success']:
     s=r['selected'];j=s['index'];assert eligible[j] and geo[j]==1 and known[j]==1
     assert not (eligible&(distance<s['normalized_angle_displacement_error'])&(geo<0)).any()
     dims=to_phys(u[j]);np.testing.assert_allclose(dims,[s['dimensions'][k] for k in FEATURES],atol=1e-12)
     key=tuple(dims)
     if key not in seen:assert all(G.check_constraints(dict(zip(FEATURES,dims),t_w=G.T_W)).values());seen.add(key)
     verify.append((u[j],y[j],p[j]));representatives.append(dict(candidate_id=f"NO_BARRIER_{strategy}_{method}_{r['goal_id']}",strategy=strategy,method=method,goal_id=r['goal_id'],goal_angle_deg=float(goal[0]),goal_deformation_mm=float(goal[1]),previous_input_goal_barrier_mJ=float(10**goal[2]),**s['dimensions'],**{f'predicted_{t}':v for t,v in s['predicted'].items()},predicted_barrier_mJ=s['predicted_barrier_mJ'],p_bi_mono=s['p_bi_mono'],normalized_angle_displacement_error=s['normalized_angle_displacement_error']))
    else:assert (geo[eligible]==0).all()
    records.append({k:r[k] for k in ['strategy','method','goal_id','seed','candidate_points','old_success','success','old_generated_pass_count','generated_pass_count','extra_geometry_checks','baseline_geometry_results_reused','seconds']})
    reused+=r['baseline_geometry_results_reused'];extra+=r['extra_geometry_checks']
 df=pd.DataFrame(records);assert len(df)==2245
 if verify:
  y,p=Forward().predict(np.array([a[0] for a in verify]));np.testing.assert_allclose(y,np.array([a[1] for a in verify]),rtol=1e-6,atol=5e-5);np.testing.assert_allclose(p,np.array([a[2] for a in verify]),rtol=1e-6,atol=1e-6)
 for name,h in protocol['forward_sha256'].items():assert hashlib.sha256((FORWARD/name).read_bytes()).hexdigest()==h
 for strategy,h in protocol['source_goal_hashes'].items():assert hashlib.sha256((SOURCE/'goals'/f'{strategy}.csv').read_bytes()).hexdigest()==h
 df.to_csv(DEST/'goal_results.csv',index=False);pd.DataFrame(representatives).to_csv(DEST/'representatives.csv',index=False)
 dump(DEST/'integrity_audit.json',dict(paired_records=len(df),same_goals_and_source_candidates_verified=True,source_trace_hashes_verified=True,all_previous_goal_successes_preserved=True,all_previous_generated_passing_candidates_preserved=True,all_generative_potential_passes_geometry_checked=True,representatives_checked=len(verify),unique_representative_geometries_original_checker_checked=len(seen),representative_forward_predictions_reloaded_and_verified=True,extra_geometry_checks=extra,reused_baseline_geometry_checks=reused,forward_weights_unchanged=True,barrier_removed_from_acceptance_only=True,new_inverse_training_or_search=False,ansys_executed=False))
 return df

def summarize(df):
 rows=[]
 for (strategy,method),q in df.groupby(['strategy','method'],sort=False):
  gen=method in METHODS[:3]
  rows.append(dict(strategy=strategy,strategy_label=SLABELS[strategy],method=method,method_label=LABELS[method],targets=len(q),old_found=int(q.old_success.sum()),found=int(q.success.sum()),old_discovery_rate=q.old_success.mean(),discovery_rate=q.success.mean(),newly_found=int((q.success&~q.old_success).sum()),old_generated_candidate_pass_rate=q.old_generated_pass_count.sum()/q.candidate_points.sum() if gen else None,generated_candidate_pass_rate=q.generated_pass_count.sum()/q.candidate_points.sum() if gen else None,extra_geometry_checks=int(q.extra_geometry_checks.sum())))
 summary=pd.DataFrame(rows);summary.to_csv(DEST/'summary.csv',index=False)
 return summary

def plots(s):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 english={'jw_cvae':'Jinwoo CVAE','sm_cvae':'Seungmin CVAE','conditional_gmm':'Conditional GMM','differential_evolution':'Forward + DE','slsqp':'Forward + SLSQP'}
 labels=['1 Actual Test','2 Neighbor interpolation','4 Angle-conditional','5 Sparse / boundary','6 Independent wide']
 fig,axes=plt.subplots(1,2,figsize=(17,6));fig.subplots_adjust(left=.17,right=.95,bottom=.28,top=.8,wspace=.18)
 for ax,methods,a,b,title in [(axes[0],METHODS,'old_discovery_rate','discovery_rate','Goal discovery: with barrier -> barrier ignored'),(axes[1],METHODS[:3],'old_generated_candidate_pass_rate','generated_candidate_pass_rate','Raw candidate pass rate: with -> without barrier')]:
  values=np.array([[100*s[(s.strategy==st)&(s.method==m)].iloc[0][b] for m in methods] for st in STRATEGIES]);im=ax.imshow(values,vmin=0,vmax=100,cmap='RdYlGn',aspect='auto');ax.set_title(title,fontsize=12,pad=15);ax.set_xticks(range(len(methods)),[english[m] for m in methods],rotation=30,ha='right');ax.set_yticks(range(5),labels if ax is axes[0] else ['']*5)
  for i,st in enumerate(STRATEGIES):
   for j,m in enumerate(methods):
    row=s[(s.strategy==st)&(s.method==m)].iloc[0];ax.text(j,i,f'{100*row[a]:.1f}%\n→ {100*row[b]:.1f}%',ha='center',va='center',fontsize=9,color='white' if values[i,j]<17 or values[i,j]>83 else '#17222b',fontweight='bold')
 fig.colorbar(im,ax=axes,fraction=.025,pad=.03,label='New pass rate (%)');fig.suptitle('Same candidate pools; barrier ACCEPTANCE removed only | angle +/-2 deg, displacement +/-0.25 mm\nOriginal barrier conditioning and optimizer searches unchanged; fixed seed 0; no ANSYS',fontsize=13)
 fig.savefig(DEST/'comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)

def html_report(s):
 body=['<h1>장벽 통과 조건을 제외한 동일 후보 재평가</h1>','<p><strong>새 판정은 각도 ±2°, 변위 ±0.25 mm, 전환 예측 확률 ≥0.9, 공식 형상 제약입니다. 장벽 오차는 판정과 대표 후보 순위에서 제외했습니다. 응력은 이전과 같이 제외합니다.</strong></p>','<p>이전 다섯 목표 유형(실제 Test 49개, 나머지 유형 각 100개)과 시드 0의 후보를 그대로 재판정했습니다. 449개 목표×5방법=2,245회입니다. 생성 모델의 입력에는 원래 장벽 목표가 여전히 들어 있습니다. 장벽 입력을 제거한 CVAE 재학습 실험이 아니며, DE/SLSQP 역시 기존에 장벽을 포함해 탐색했던 점들을 재판정한 것이지 새 두 조건 목적함수로 재탐색한 결과는 아닙니다.</p>','<h2>목표 발견률: 이전 → 장벽 판정 제외</h2>','<p>통과 후보를 하나 이상 찾은 목표 비율입니다. 최적화는 적응적으로 수집한 점이므로 개별 점 통과율을 생성 후보 비율과 비교하지 않습니다.</p>','<table><tr><th>목표 유형</th>'+''.join(f'<th>{LABELS[m]}</th>' for m in METHODS)+'</tr>']
 for st in STRATEGIES:
  q=s[s.strategy==st].set_index('method');body.append(f'<tr><td>{SLABELS[st]}</td>'+''.join(f'<td>{100*q.loc[m,"old_discovery_rate"]:.1f}% → {100*q.loc[m,"discovery_rate"]:.1f}%<br>({int(q.loc[m,"found"])}/{int(q.loc[m,"targets"])})</td>' for m in METHODS)+'</tr>')
 body+=['</table>','<h2>개별 생성 후보 통과율: 이전 → 장벽 판정 제외</h2>','<p>필터 전 원시 생성 후보 4,096개 전체를 분모로 합니다. 새 기준에서 통과할 가능성이 있는 모든 생성 후보에 공식 정밀 형상 판정을 확보했습니다. 기존 확인 결과와 비트 단위로 같은 baseline 후보의 과거 형상 결과는 재사용했고, 나머지는 추가 검사했습니다.</p>','<table><tr><th>목표 유형</th>'+''.join(f'<th>{LABELS[m]}</th>' for m in METHODS[:3])+'</tr>']
 for st in STRATEGIES:
  q=s[s.strategy==st].set_index('method');body.append(f'<tr><td>{SLABELS[st]}</td>'+''.join(f'<td>{100*q.loc[m,"old_generated_candidate_pass_rate"]:.2f}% → {100*q.loc[m,"generated_candidate_pass_rate"]:.2f}%</td>' for m in METHODS[:3])+'</tr>')
 body+=['</table>','<img src="comparison.png">','<h2>같은 목표의 거리 변화</h2>','<p>Train 전환 응답 394개와 비교해, 가장 가까운 응답이 목표 허용오차 안에 있는 비율입니다. 이전에는 각도·변위·장벽을 모두 사용했고, 아래 새 거리는 각도·변위만 사용합니다. 후보 통과율과는 다른 분포 진단입니다.</p>','<table><tr><th>목표 유형</th><th>기존 3차원 거리 ≤1</th><th>각도·변위 거리 ≤1</th></tr>']
 for _,r in pd.read_csv(DEST/'goal_distance_summary.csv').iterrows():body.append(f'<tr><td>{SLABELS[r.strategy]}</td><td>{100*r.within_train_3d_fraction:.1f}%</td><td>{100*r.within_train_2d_fraction:.1f}%</td></tr>')
 body+=['</table>','<h2>해석 범위와 검증</h2>','<p>이 비교는 같은 목표·시드·후보에서 장벽 통과 조건만 제거한 효과입니다. 입력 조건에서 장벽을 삭제해 새 역설계 모델을 학습했을 때의 성능과는 다릅니다. 목표 유형은 원래 세 변수로 생성했던 그대로 유지했기 때문에, 원래 장벽 차이가 있던 목표들은 지금 두 변수 판정에서는 더 비슷한 목표가 될 수 있습니다.</p>','<p>기존에 성공했던 목표가 모두 계속 성공하고, 기존에 통과했던 개별 생성 후보도 모두 새 기준에서 통과함을 확인했습니다. 대표 후보는 각도와 변위의 허용오차 정규화 오차 중 최댓값이 가장 작은 공식 형상 유효 후보입니다. 대표 치수는 원래 형상 검사 함수로 다시 확인하고, 고정 Forward를 재로딩해 예측도 검증했습니다.</p>','<p>성공률은 동일한 Forward로 선별·채점한 대리모델 기준이며 실제 ANSYS 검증 성공률이 아닙니다. 장벽 오차를 무시하므로 통과 후보가 원래 목표 장벽에 가까울 필요는 없습니다. representatives.csv의 장벽 예측은 참고 출력이며 통과 기준이 아닙니다.</p>','<h2>파일</h2>','<p>summary.csv: 유형/방법별 전후 집계 · goal_results.csv: 목표별 전후 판정 · representatives.csv: 두 조건 기준 대표 치수 · runs/: 갱신된 형상 검사와 판정 상태 · protocol.json: 범위와 원본 참조 · integrity_audit.json: 재검사 결과. 원래 후보 좌표와 Forward 예측은 기존 실험 폴더를 참조하고 복제하지 않았습니다.</p>']
 (DEST/'평가 결과.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>장벽 판정 제외 평가</title><style>body{font-family:Arial,"Apple SD Gothic Neo",sans-serif;max-width:1300px;margin:35px auto;padding:0 22px;line-height:1.7;color:#18232d}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px;border:1px solid #ccd5df;text-align:right}th{background:#eaf1f8}td:first-child,th:first-child{text-align:left}img{width:100%;margin:25px 0}h2{margin-top:35px}</style>'+''.join(body),encoding='utf-8')

def goal_distance_review():
 training=pd.read_csv(SOURCE/'target_distribution_training.csv');a=training[TARGETS[:2]].to_numpy();rows=[];allrows=[]
 for strategy in STRATEGIES:
  q=pd.read_csv(SOURCE/'goals'/f'{strategy}.csv');dist=np.abs((q[TARGETS[:2]].to_numpy()[:,None,:]-a[None,:,:])/LIMITS).max(2).min(1)
  rows.append(dict(strategy=strategy,targets=len(q),within_train_3d_fraction=q.within_train_response_tolerances.mean(),within_train_2d_fraction=(dist<=1).mean(),median_nearest_train_2d_tolerance_distance=np.median(dist)))
  for gid,d in zip(q.goal_id,dist):allrows.append(dict(strategy=strategy,goal_id=gid,nearest_train_angle_displacement_distance=float(d)))
 pd.DataFrame(rows).to_csv(DEST/'goal_distance_summary.csv',index=False);pd.DataFrame(allrows).to_csv(DEST/'goal_distance_without_barrier.csv',index=False)

def main():
 df=collect();s=summarize(df);goal_distance_review();plots(s);html_report(s);print(s[['strategy','method','old_found','found','targets','old_generated_candidate_pass_rate','generated_candidate_pass_rate']].to_string(index=False),flush=True)
if __name__=='__main__':main()
