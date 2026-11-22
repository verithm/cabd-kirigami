"""Step-7 surrogate comparison. No result is described as a new ANSYS validation."""
import html
from shared import *

def make_report():
 rows=[];representatives=[];nearest=[]
 diagnostic=json.loads((OUT/'data/forward_witness_diagnostic.json').read_text())
 for method in METHODS:
  for seed in SEEDS:
   files=sorted((OUT/'runs'/method/f'seed{seed}').glob('TE*.json'))
   assert len(files)==49,(method,seed,len(files))
   for p in files:
    r=json.loads(p.read_text());base=dict(method=method,seed=seed,goal_id=r['goal_id'],**{f'goal_{k}':v for k,v in r['goal'].items()},goal_barrier_mJ=r['goal_barrier_mJ'])
    row={**base,**{k:r[k] for k in ['success','pass_count','pass_fraction','forward_calls','candidate_points','bounds_pass_count','stability_passing_count','surrogate_joint_count','unique_designs_1e4','official_geometry_checks','search_seconds','filter_seconds','total_seconds']}}
    for b,v in r['budget_prefix'].items():row[f'found_by_{b}']=v
    for t,v in zip(TARGETS,r['property_pass_fractions']):row[f'property_only_{t}']=v
    row['nearest_valid_normalized_error']=r['nearest']['normalized_max_error'] if r['nearest'] else np.nan
    rows.append(row)
    if r['selected']:
     s=r['selected'];representatives.append(dict(candidate_id=f"INV_{method}_s{seed}_{r['goal_id']}",**base,**s['dimensions'],**{f'predicted_{k}':v for k,v in s['predicted'].items()},predicted_barrier_mJ=s['predicted_barrier_mJ'],p_bi_mono=s['p_bi_mono'],normalized_max_error=s['normalized_max_error']))
     assert np.float32(s['p_bi_mono'])>=np.float32(P_MIN) and s['normalized_max_error']<=1
    if r['nearest']:
     n=r['nearest'];nearest.append(dict(**base,**n['dimensions'],**{f'predicted_{k}':v for k,v in n['predicted'].items()},p_bi_mono=n['p_bi_mono'],normalized_max_error=n['normalized_max_error'],passes_all=n['passes_all']))
 df=pd.DataFrame(rows);assert len(df)==735
 df.to_csv(OUT/'goal_results.csv',index=False)
 pd.DataFrame(representatives).to_csv(OUT/'representatives.csv',index=False)
 pd.DataFrame(nearest).to_csv(OUT/'nearest_candidates.csv',index=False)
 summaries=[];seedrows=[]
 for method in METHODS:
  q=df[df.method==method]
  per=q.groupby('seed').success.mean()
  unique=int(q.groupby('goal_id').success.any().sum())
  isgen=method in METHODS[:3]
  summary=dict(method=method,label=LABELS[method],goals_per_seed=49,seeds=len(SEEDS),found_goal_seed=int(q.success.sum()),goal_seed_total=len(q),goal_discovery_rate=float(per.mean()),goal_discovery_std=float(per.std(ddof=0)),unique_goals_found=unique,generated_candidate_pass_rate=float(q.pass_count.sum()/q.candidate_points.sum()) if isgen else None,adaptive_visited_point_pass_rate=None if isgen else float(q.pass_count.sum()/q.candidate_points.sum()),total_passing_points=int(q.pass_count.sum()),total_evaluated_points=int(q.candidate_points.sum()),mean_forward_evaluations=float(q.forward_calls.mean()),median_seconds_per_goal=float(q.total_seconds.median()),mean_seconds_per_goal=float(q.total_seconds.mean()),mean_search_seconds=float(q.search_seconds.mean()),mean_filter_audit_seconds=float(q.filter_seconds.mean()),mean_unique_designs_1e4=float(q.unique_designs_1e4.mean()),median_nearest_error=float(q.nearest_valid_normalized_error.median()))
  for b in [512,1024,2048,4096]:summary[f'found_by_{b}']=float(q[f'found_by_{b}'].mean())
  summaries.append(summary)
  for seed,s in q.groupby('seed'):seedrows.append(dict(method=method,seed=int(seed),goals=49,found=int(s.success.sum()),rate=float(s.success.mean()),forward_evaluations=int(s.forward_calls.sum()),seconds=float(s.total_seconds.sum())))
 summary=pd.DataFrame(summaries).sort_values(['goal_discovery_rate','mean_seconds_per_goal'],ascending=[False,True]);summary.to_csv(OUT/'summary.csv',index=False)
 pd.DataFrame(seedrows).to_csv(OUT/'metrics_by_seed.csv',index=False)
 dump(OUT/'summary.json',summaries)
 assert all(len(q)==147 for _,q in df.groupby('method'))
 expected=set(pd.read_csv(OUT/'data/test_goals.csv').goal_id)
 assert all(set(q.goal_id)==expected for _,q in df.groupby(['method','seed']))
 assert df.forward_calls.le(BUDGET).all()
 audit=dict(methods=len(METHODS),seeds=SEEDS,goals=49,goal_seed_method_records=len(df),shared_goal_ids=True,within_forward_budget=True,representatives=len(representatives),ansys_executed=False)
 dump(OUT/'evaluation_audit.json',audit)
 make_plots(df,summary)
 fmt=lambda x:f'{100*x:.1f}%'
 body=['<h1>통합 역설계: 학습 및 Forward 기준 평가</h1>','<p><strong>공통 Forward 예측과 공식 형상 제약에 따른 결과입니다. 새 ANSYS 해석은 실행하지 않았습니다.</strong></p>',f'<p>실제 시험 전환 형상의 응답 조합 49개 × 시드 3개. 목표별 최대 Forward 평가 {BUDGET:,}개. 안정각 ±{TOL[0]}°, 변위 ±{TOL[1]} mm, log₁₀ 장벽 ±{TOL[2]}, P(bi→mono) ≥{P_MIN}, 공식 형상 제약을 모두 적용했습니다.</p>','<h2>방법별 결과</h2>','<table><tr><th>방법</th><th>목표·시드 발견</th><th>발견률 평균 ± SD</th><th>고유 목표</th><th>개별 생성 후보 통과율</th><th>목표당 Forward 평가</th><th>목표당 시간 중앙값</th></tr>']
 for _,r in summary.iterrows():
  rate='—' if pd.isna(r.generated_candidate_pass_rate) else fmt(r.generated_candidate_pass_rate)
  body.append(f'<tr><td>{html.escape(r.label)}</td><td>{r.found_goal_seed}/147</td><td>{100*r.goal_discovery_rate:.1f}% ± {100*r.goal_discovery_std:.1f}%</td><td>{r.unique_goals_found}/49</td><td>{rate}</td><td>{r.mean_forward_evaluations:.0f}</td><td>{r.median_seconds_per_goal:.2f}초</td></tr>')
 body+=['</table>','<p>발견률은 정해진 예산 안에서 통과 후보를 하나 이상 찾은 비율입니다. 고유 목표는 세 시드를 합쳐 한 번이라도 성공한 목표 수입니다. 생성 모델의 개별 통과율은 필터 전 전체 생성 후보를 분모로 하며, 범위 밖 후보도 실패로 포함합니다. DE·SLSQP의 탐색점 비율은 적응적으로 모인 점의 비율이므로 생성 모델의 개별 후보 통과율과 같은 의미로 비교하지 않습니다.</p>','<p>모든 방법은 실제로 평가한 후보 중에서 허용오차로 나눈 세 응답 오차의 최댓값이 가장 작은 통과 후보를 1순위로 선택합니다. 같은 Forward로 선별하고 채점하므로 1순위 통과 여부는 발견 여부와 같습니다. 이를 독립적인 실제 성능 검증으로 해석하면 안 됩니다.</p>','<img src="method_comparison.png" alt="역설계 방법 비교"><img src="budget_discovery.png" alt="후보 예산별 발견률">','<h2>조건부 생성 성능</h2>','<p>아래는 학습 데이터의 분포가 아니라 고정한 목표 49개에 대한 개별 생성 후보의 최종 통과율입니다. 목표는 안정각 순으로 정렬했습니다.</p>','<img src="candidate_pass_heatmap.png" alt="목표별 후보 통과율">','<h2>기존 실제 형상에 대한 Forward 진단</h2>',f"<p>목표를 제공한 실제 시험 형상 49개는 모두 공식 형상 제약을 통과했습니다. 그 치수를 고정 Forward에 넣으면 세 응답 허용오차를 동시에 만족하는 것은 {diagnostic['response_ok']}개, 전환 확률 기준을 만족하는 것은 {diagnostic['probability_ok']}개, 모든 기준을 동시에 만족하는 것은 {diagnostic['joint_ok']}개였습니다. 이는 새 해석 결과가 아니라 기존 실제 형상에 대한 대리모델 진단입니다. 따라서 역설계 탐색의 성공·실패를 실제 형상의 성능이나 목표의 물리적 가능성과 동일시하지 않습니다.</p>",'<h2>실험 설정과 해석 범위</h2>','<p>Forward는 교차검증으로 선정한 승민 MLP의 시드 0을 사전에 고정했습니다. 분류와 회귀 각각 5개 신경망의 평균을 사용하며 다른 시드나 다른 Forward로 교체하지 않았습니다. 각 역설계 방법은 시드 0·1·2로 반복했습니다.</p>','<p>생성 모델에는 partial 쌍안정이고 full 판정이 확인된 Train 형상 788개를 제공합니다(전환 394, 두 상태 쌍안정 394). Train 내부 630/158개를 분리하고 검증 전환 형상 중 각도 범위를 고르게 고른 목표 20개에서 설정 3개를 비교했습니다. 설정 선택은 발견률, 동률이면 개별 후보 통과율, 다시 동률이면 목표 근접도로 결정합니다. 이후 전체 788개로 재학습했습니다. GMM은 전환 여부별로 혼합분포를 각각 학습하며 생성 시 전환 그룹을 조건화합니다. CVAE는 두 그룹 사이에 신경망 가중치를 공유합니다.</p>','<p>학습 손실은 진우 CVAE의 치수 MSE+KL, 승민 CVAE의 대각 가우시안 NLL+KL입니다. 승민 CVAE는 원래 구현처럼 디코더 평균으로 후보를 생성합니다. 두 CVAE 모두 Forward를 학습 손실에 넣지 않았습니다. 장벽은 모든 조건에서 log₁₀로 사용하고, 연속 조건의 평균·표준편차는 해당 학습 부분에서만 구했습니다.</p>','<p>DE는 빠른 해석적 형상 조건과 전환 확률 위반을 목적함수에 반영합니다. SLSQP는 전환 확률과 해석적 형상 조건을 부등식 제약으로 두고 8개 Sobol 시작점에서 유한차분으로 탐색합니다. 공식 다각형 교차·겹침·노드 수를 포함한 최종 형상 검사는 모든 방법에서 동일합니다. 공통 Forward 평가 상한은 4,096개이며 SLSQP는 먼저 수렴하면 적게 사용할 수 있습니다. 유한차분에 쓴 점도 실제 Forward 평가 횟수에 포함합니다.</p>','<p>효율을 위해 응답·확률 기준을 이미 실패한 점의 정밀 다각형 검사는 생략합니다. 최종 공동 통과 수는 정밀 형상 검사를 통과한 후보만 셉니다. 원래 시험 형상의 치수는 진단에만 사용하며 생성 모델의 학습이나 최적화 시작점에 쓰지 않았습니다. 목표 자체의 상관관계는 실제 응답 조합을 사용해 유지했습니다.</p>','<p>원본 ANSYS 결과로 실제 존재하는 목표를 사용했어도, 생성된 새 후보의 실제 응답이 맞는지는 이 평가로 확인되지 않습니다. Forward 근사 오차와 최적화의 대리모델 활용 효과가 함께 포함됩니다. 시간에는 전체 후보 통과율을 집계하기 위한 전수 형상 검사도 포함됩니다. 실제로 통과 후보 한 개를 찾고 즉시 중단하는 실행 시간과 다릅니다. summary.csv에 순수 탐색/Forward 시간과 필터 감사 시간을 분리했습니다. 시간은 같은 노트북에서 프로세스별 단일 스레드로 측정한 값이며 병렬 작업의 CPU 경합 영향을 받을 수 있습니다. 표준편차는 시드 변동이며 95% 신뢰구간이 아닙니다.</p>','<h2>결과 파일</h2>','<p>summary.csv: 방법별 집계 · goal_results.csv: 목표/시드별 결과 · representatives.csv: 통과 목표별 1순위 치수 · nearest_candidates.csv: 실패 목표를 포함한 가장 가까운 유효 후보 진단 · runs/: 모든 평가점과 개별 결과 · tuning/: 검증 결과 · models/: 학습 가중치. 원본 형상 진단은 data/forward_witness_diagnostic.json에 별도로 저장했습니다.</p>']
 (OUT/'평가 결과.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>통합 역설계 평가</title><style>body{font-family:Arial,"Apple SD Gothic Neo",sans-serif;max-width:1250px;margin:35px auto;padding:0 25px;color:#17202a;line-height:1.65}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:9px;border:1px solid #d5dce4;text-align:right}th{background:#eaf0f8}td:first-child,th:first-child{text-align:left}img{width:100%;height:auto;margin:18px 0}h2{margin-top:35px}</style>'+''.join(body),encoding='utf-8')
 p=json.loads((OUT/'protocol.json').read_text());p['status']='complete';p['ansys_executed']=False;p['code_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')};dump(OUT/'protocol.json',p)
 print(summary.to_string(index=False),flush=True)


def make_plots(df,summary):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 english={'jw_cvae':'Jinwoo CVAE','sm_cvae':'Seungmin CVAE','conditional_gmm':'Conditional GMM','differential_evolution':'Forward + DE','slsqp':'Forward + SLSQP'}
 order=summary.method.tolist();colors=['#3579b8' if m in METHODS[:3] else '#d17a35' for m in order]
 fig,axs=plt.subplots(1,3,figsize=(16,5));fig.subplots_adjust(left=.06,right=.98,bottom=.3,top=.85,wspace=.35)
 axs[0].bar(range(5),summary.goal_discovery_rate*100,color=colors);axs[0].set_ylim(0,105);axs[0].set_ylabel('Found goals (%)');axs[0].set_title('Surrogate goal discovery')
 gen=summary[summary.method.isin(METHODS[:3])];axs[1].bar(range(3),gen.generated_candidate_pass_rate*100,color='#3579b8');axs[1].set_ylim(0,100);axs[1].set_ylabel('Passing raw candidates (%)');axs[1].set_title('Generative candidate pass rate')
 axs[2].bar(range(5),summary.mean_search_seconds,color='#607d8b',label='Search + Forward');axs[2].bar(range(5),summary.mean_filter_audit_seconds,bottom=summary.mean_search_seconds,color='#cfd8dc',label='Full pool geometry audit');axs[2].set_ylabel('Seconds / goal');axs[2].set_title('Mean search and audit time');axs[2].legend(fontsize=8)
 for ax,values in [(axs[0],summary.goal_discovery_rate*100),(axs[1],gen.generated_candidate_pass_rate*100)]:
  ax.bar_label(ax.containers[0],labels=[f'{v:.1f}%' for v in values],padding=4,fontsize=9)
 for ax,methods in zip(axs,[order,gen.method.tolist(),order]):
  ax.set_xticks(range(len(methods)),[english[m] for m in methods],rotation=35,ha='right',fontsize=9);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['right','top']].set_visible(False)
 fig.suptitle('Fixed Forward and geometry checks | 49 goals x 3 seeds | No new ANSYS validation')
 fig.savefig(OUT/'method_comparison.png',dpi=170,bbox_inches='tight');fig.savefig(OUT/'method_comparison.svg',bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,5))
 for m in METHODS:
  q=df[df.method==m];ax.plot([512,1024,2048,4096],[100*q[f'found_by_{b}'].mean() for b in [512,1024,2048,4096]],marker='o',label=english[m])
 ax.set_xscale('log',base=2);ax.set_xticks([512,1024,2048,4096],['512','1,024','2,048','4,096']);ax.set_ylim(0,105);ax.set_xlabel('Forward evaluation cap (prefix of the recorded search)');ax.set_ylabel('Goals with at least one passing candidate (%)');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/'budget_discovery.png',dpi=170);plt.close(fig)
 goals=pd.read_csv(OUT/'data/test_goals.csv').sort_values('angle_deg');order=goals.goal_id.tolist();matrix=[]
 for m in METHODS[:3]:matrix.append(df[df.method==m].groupby('goal_id').pass_fraction.mean().reindex(order).to_numpy()*100)
 fig,ax=plt.subplots(figsize=(15,3.5));im=ax.imshow(np.array(matrix),vmin=0,vmax=100,cmap='RdYlGn',aspect='auto');ax.set_yticks(range(3),[english[m] for m in METHODS[:3]]);ticks=np.arange(0,49,6);ax.set_xticks(ticks,[f"{goals.iloc[i].angle_deg:.1f}°\n{order[i]}" for i in ticks]);ax.set_xlabel('Actual response targets, sorted by stable angle');ax.set_title('Raw generated candidate pass rate by target (all criteria, 0–100%)');fig.colorbar(im,ax=ax,label='Passing candidates (%)');fig.tight_layout();fig.savefig(OUT/'candidate_pass_heatmap.png',dpi=170);plt.close(fig)

if __name__=='__main__':make_report()
