"""Descriptive report for physical-unit tolerance review; Train OOF informs recommendation."""
from shared import *
import html
DEST=RUNS/'barrier-tolerance-review'

def main():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 cv=pd.read_csv(DEST/'train_oof_predictions.csv');stats=pd.read_csv(DEST/'forward_oof_error_summary.csv');inv=pd.read_csv(DEST/'inverse_sensitivity_summary.csv');examples=pd.read_csv(DEST/'physical_scale_examples.csv')
 fig,axes=plt.subplots(1,2,figsize=(13,4.8));fig.subplots_adjust(left=.07,right=.97,bottom=.16,top=.84,wspace=.32)
 x=np.linspace(0,1.5,301)
 axes[0].plot(x,x,color='#414c59',lw=1,label='Perfect match')
 axes[0].fill_between(x,x*10**(-.17),x*10**.17,color='#c98263',alpha=.4,label='log10 tolerance +/-0.17')
 axes[0].fill_between(x,np.maximum(x-.03,0),x+.03,color='#2169a8',alpha=.65,label='Absolute tolerance +/-0.03 mJ')
 axes[0].set_xlabel('Target barrier (mJ)');axes[0].set_ylabel('Allowed predicted barrier (mJ)');axes[0].set_title('The same log tolerance permits larger energy errors');axes[0].legend(fontsize=8,loc='upper left');axes[0].set_xlim(0,1.5);axes[0].set_ylim(0,2.25)
 for q,label,color in [(cv,'All partial-bi Train OOF (n=836)','#8195a9'),(cv[cv.label=='bi>mono'],'bi -> mono Train OOF (n=394)','#2169a8')]:
  e=np.sort(q.absolute_error_mJ);axes[1].plot(np.r_[0,e],np.r_[0,np.arange(1,len(e)+1)/len(e)*100],label=label,color=color,lw=2)
 for t,color in [(.02,'#b0b8c1'),(.03,'#2169a8'),(.05,'#b0b8c1')]:axes[1].axvline(t,color=color,ls='--',lw=1)
 axes[1].set_xlabel('Absolute Forward prediction error (mJ)');axes[1].set_ylabel('Predictions within the error (%)');axes[1].set_xlim(0,.08);axes[1].set_ylim(0,103);axes[1].set_title('Train OOF errors in physical units');axes[1].legend(fontsize=8,loc='lower right')
 for ax in axes:ax.grid(alpha=.18);ax.set_axisbelow(True);ax.spines[['right','top']].set_visible(False)
 fig.suptitle('Barrier tolerance review | Training remains in log10; acceptance can use mJ')
 fig.savefig(DEST/'barrier_tolerance_review.png',dpi=170);plt.close(fig)
 recommendation=dict(status='recommendation, not a new user-confirmed barrier threshold',angle_tolerance_deg=2.,displacement_tolerance_mm=.25,barrier_acceptance='abs(predicted_barrier_mJ - target_barrier_mJ) <= 0.03',barrier_absolute_tolerance_mJ=.03,predicted_bi_mono_probability_min=.9,official_geometry_required=True,stress_filter=False,barrier_learning_space='log10 unchanged',basis='Original selected-model Train OOF n836, relevant bi->mono subgroup n394; not maximizing Test inverse discovery',limitations=['Same mJ tolerance is proportionally looser for tiny barriers and stricter for large barriers','Forward OOF coverage is not inverse or ANSYS success probability'],ansys_executed=False)
 dump(DEST/'recommended_settings.json',recommendation)
 body=['<h1>장벽 허용오차: log₁₀에서 mJ 기준으로 검토</h1>','<p><strong>안정각 ±2°, 변위 ±0.25 mm, 전환 예측 확률 ≥0.9와 공식 형상 제약을 고정했습니다. 장벽은 학습 시 log₁₀를 유지하고, 후보 판정 시 10의 거듭제곱으로 mJ를 복원할 수 있습니다.</strong></p>','<h2>권장 시작 기준: 장벽 절대오차 ±0.03 mJ</h2>','<p>선정된 Forward의 원래 5-fold 교차검증을 같은 설정(config 0)과 시드 42로 복원했습니다. 저장된 기존 CV 지표와 일치하는지 확인했습니다. 전체 partial-bi Train OOF 836개에서 장벽 MAE는 0.01203 mJ, RMSE는 0.01869 mJ입니다. 실제 전환 형상 394개에서는 MAE 0.01476 mJ, RMSE 0.02169 mJ입니다. ±0.02 mJ는 엄격한 편이고, ±0.03 mJ는 중간 기준, ±0.05 mJ는 더 높은 Forward 오차 포괄률을 원하는 완화 기준입니다. 이 권장은 아래 Test 후보의 발견률을 최대화해서 정하지 않았습니다.</p>','<h2>Forward 예측 오차가 기준 안에 드는 비율</h2>','<p>이 표는 실제 형상에 대한 Train OOF 장벽 예측의 오차 포괄률입니다. 역설계 후보의 성공률이나 실제 ANSYS 검증 성공률을 뜻하지 않습니다.</p>','<table><tr><th>Train OOF 집단</th><th>건수</th><th>±0.02 mJ</th><th>±0.03 mJ</th><th>±0.05 mJ</th></tr>']
 for _,r in stats.iterrows():body.append(f'<tr><td>{html.escape(r.population)}</td><td>{r.n}</td>'+''.join(f'<td>{100*r[f"within_{t:.2f}mJ"]:.1f}%</td>' for t in [.02,.03,.05])+'</tr>')
 body+=['</table>','<img src="barrier_tolerance_review.png">','<h2>현재 log₁₀ ±0.17이 뜻하는 실제 허용 범위</h2>','<p>|log₁₀(Ê) − log₁₀(E*)| ≤ 0.17은 0.6761 E* ≤ Ê ≤ 1.4791 E*와 같습니다. 목표 대비 −32.4%~+47.9%이며, mJ 기준으로 대칭인 오차가 아닙니다.</p>','<table><tr><th>목표 장벽</th><th>log₁₀ ±0.17의 허용 범위</th><th>절대오차 ±0.03 mJ의 허용 범위</th></tr>']
 for _,r in examples.iterrows():body.append(f'<tr><td>{r.goal_mJ:g} mJ</td><td>{r.log10_017_low:.4f}–{r.log10_017_high:.4f} mJ</td><td>{r.abs_003_low:.4f}–{r.abs_003_high:.4f} mJ</td></tr>')
 body+=['</table>','<p>절대오차 ±0.03 mJ는 모든 목표에 같은 에너지 차이를 허용합니다. 따라서 목표 1 mJ에서는 ±3%지만 0.01 mJ에서는 ±300%에 해당합니다. 아주 작은 장벽까지 비율로 정밀하게 맞추려면 상대오차 기준이나 별도의 최소 장벽 기준이 필요하며, 원래 단위 허용오차만으로 상대적 정확도를 주장할 수 없습니다. 이 변경은 일괄적인 완화가 아니라 큰 장벽에는 더 엄격하고 작은 장벽에는 더 느슨한 판정 방식입니다.</p>','<h2>같은 역설계 후보를 재판정한 민감도</h2>','<p>실제 Test 응답 조합 49개 × 기존 역설계 시드 3개를 유지했습니다. 아래는 목표 발견률(통과 후보를 하나 이상 찾은 평가 비율)이며, 생성·재학습·최적화 재탐색은 하지 않았습니다. 최적화도 기존 log 기준으로 탐색한 점들을 새 기준으로 채점한 결과입니다. 새 통과 가능 후보에는 공식 형상 검사를 추가했습니다. 새 ANSYS 해석은 실행하지 않았습니다.</p>','<table><tr><th>방법</th><th>log₁₀ ±0.17</th><th>±0.02 mJ</th><th>±0.03 mJ</th><th>±0.05 mJ</th></tr>']
 criteria=['log10_0.17','abs_0.02mJ','abs_0.03mJ','abs_0.05mJ']
 for m in METHODS:
  q=inv[inv.method==m].set_index('criterion');body.append(f'<tr><td>{LABELS[m]}</td>'+''.join(f'<td>{100*q.loc[c,"discovery_rate"]:.1f}% ({int(q.loc[c,"found"])}/147)</td>' for c in criteria)+'</tr>')
 body+=['</table>','<h2>생성 모델의 개별 후보 통과율</h2>','<table><tr><th>방법</th><th>log₁₀ ±0.17</th><th>±0.02 mJ</th><th>±0.03 mJ</th><th>±0.05 mJ</th></tr>']
 for m in METHODS[:3]:
  q=inv[inv.method==m].set_index('criterion');body.append(f'<tr><td>{LABELS[m]}</td>'+''.join(f'<td>{100*q.loc[c,"generated_candidate_pass_rate"]:.1f}%</td>' for c in criteria)+'</tr>')
 body+=['</table>','<p>개별 후보 비율은 각 생성 모델의 필터 전 생성 후보 전체를 분모로 합니다. 같은 Forward로 판정한 대리모델 기준이며 독립적인 실제 응답 검증이 아닙니다.</p>','<h2>파일</h2>','<p>forward_oof_error_summary.csv: Train OOF 오차 요약 · train_oof_predictions.csv: OOF 개별 오차 · forward_error_by_energy_bin.csv: 장벽 구간별 오차 · physical_scale_examples.csv: 로그 허용오차의 물리 단위 환산 · inverse_sensitivity_summary.csv: 역설계 재판정 요약 · candidate_results.csv: 목표/시드별 판정 · representatives.csv: 기준별 대표 치수 · recommended_settings.json: 권장 시작 기준과 한계.</p>']
 (DEST/'검토 결과.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>장벽 허용오차 검토</title><style>body{font-family:Arial,"Apple SD Gothic Neo",sans-serif;max-width:1150px;margin:40px auto;padding:0 20px;line-height:1.7;color:#18202b}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border:1px solid #ccd4df;text-align:right}th{background:#e9eff8}td:first-child,th:first-child{text-align:left}img{width:100%;margin:25px 0}h2{margin-top:35px}</style>'+''.join(body),encoding='utf-8')
 print('REPORT COMPLETE',DEST/'검토 결과.html')
if __name__=='__main__':main()
