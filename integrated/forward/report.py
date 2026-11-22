"""Aggregate identical Test populations; no tolerance-based success metrics."""
import html,json
from pathlib import Path
import numpy as np
import pandas as pd
from common import *
LABELS={'jw_mlp':'진우 MLP','jw_xgboost':'진우 XGBoost','jw_gp':'진우 GP','jw_extra_trees':'진우 Extra Trees','jw_svm':'진우 RBF SVM','jw_mlp_xgboost':'진우 MLP+XGBoost','sm_mlp':'승민 MLP 앙상블','sm_gp':'승민 GP','sm_rom_gp':'승민 ROM+GP','sm_referee':'승민 Referee','sm_referee_nogp':'승민 Referee(GP 제외)'}

def generate():
 models=[];cv=[];preds=[]
 for p in sorted((OUT/'models').glob('*/complete.json')):
  r=json.loads(p.read_text());models.append(r);preds.append(pd.read_csv(p.parent/'test_predictions.csv'))
  for c in r['cv']:
   target=c['target']
   if target=='joint':
    for t in TARGETS:cv.append(dict(model=r['model'],target=t,config=c['config'],score=c[t]['mae']))
   else:cv.append(dict(model=r['model'],target=target,config=c['config'],score=c['f1' if target=='stability' else 'mae']))
 if not preds:return
 p=pd.concat(preds,ignore_index=True)
 p=p[p.target!='barrier_mJ'].copy()
 physical=p[p.target=='log10_barrier'].copy();physical['target']='barrier_mJ'
 physical['actual']=10**physical['actual'];physical['predicted']=10**physical['predicted']
 p=pd.concat([p,physical],ignore_index=True);scores=[]
 for (m,t,s),q in p.groupby(['model','target','seed']):
  if t=='barrier_mJ':continue
  assert not q.geom_id.duplicated().any()
  metric=metrics_cls(q.actual.astype(bool),q.predicted.astype(bool)) if t=='stability' else metrics_reg(q.actual,q.predicted)
  scores.append(dict(model=m,target=t,seed=s,**metric))
  if t=='log10_barrier':scores.append(dict(model=m,target='barrier_mJ',seed=s,**metrics_reg(10**q.actual,10**q.predicted)))
 p.to_csv(OUT/'all_test_predictions.csv',index=False)
 s=pd.DataFrame(scores);s.to_csv(OUT/'test_metrics_by_seed.csv',index=False)
 pd.DataFrame(cv).to_csv(OUT/'cv_configs.csv',index=False)
 aggregation=[]
 for (m,t),q in s.groupby(['model','target']):
  row=dict(model=m,label=LABELS[m],target=t,seeds=len(q),test_n=int(q.n.iloc[0]))
  for col in ['f1','precision','recall','balanced_accuracy','mae','rmse','r2']:
   if col in q and q[col].notna().any():row[col+'_mean']=float(q[col].mean());row[col+'_std']=float(q[col].std(ddof=0))
  selected=next((z['config'] for model in models if model['model']==m for z in model['test'] if z['target']==t),None)
  cvs=[x['score'] for x in cv if x['model']==m and x['target']==t and x['config']==selected]
  row['selected_config']=selected
  if cvs:row['selected_cv_score']=cvs[0]
  aggregation.append(row)
 agg=pd.DataFrame(aggregation);agg.to_csv(OUT/'test_summary.csv',index=False)
 winners=[]
 for t in ['stability',*TARGETS]:
  q=agg[agg.target==t].sort_values('f1_mean' if t=='stability' else 'mae_mean',ascending=t!='stability')
  if not len(q):continue
  w=q.iloc[0].dropna().to_dict();w['basis']='Test descriptive ranking; configurations were selected by CV only';winners.append(w)
 dump(OUT/'test_best_by_output.json',winners)
 cv_winners=[]
 for t in ['stability',*TARGETS]:
  q=agg[(agg.target==t)&agg.selected_cv_score.notna()].sort_values('selected_cv_score',ascending=t!='stability')
  if len(q):cv_winners.append(q.iloc[0].dropna().to_dict())
 dump(OUT/'cv_best_by_output.json',cv_winners)
 selection={}
 for r in cv_winners:
  t=r['target'];stem='joint' if r['model']=='sm_mlp' and t!='stability' else t
  selection[t]=dict(model=r['model'],criterion='CV F1' if t=='stability' else 'CV MAE',cv_score=r['selected_cv_score'],files={str(seed):f"models/{r['model']}/{stem}_s{seed}.joblib" for seed in SEEDS})
 dump(OUT/'forward_selection.json',dict(basis='Training cross-validation only; Test descriptive ranks are separate',seed_policy='Each seed is evaluated separately; the reported mean is not a new ensemble',outputs=selection))
 # Verify every method is evaluated against exactly the same rows and truth.
 checks={}
 for t in ['stability',*TARGETS]:
  groups=[q.sort_values('geom_id') for _,q in p[(p.target==t)&(p.seed==0)].groupby('model')]
  for q in groups[1:]:
   assert q.geom_id.tolist()==groups[0].geom_id.tolist()
   np.testing.assert_allclose(q.actual,groups[0].actual,atol=0,rtol=0)
  checks[t]=dict(models=len(groups),n=len(groups[0])) if groups else {}
 dump(OUT/'evaluation_checks.json',checks)
 content=['<h1>통합 Forward 평가 결과</h1>',f'<p>완료 모델: {len(models)}/11</p>','<p>공통 에너지 정답 · ok/15 mm 완료 · 장벽 log₁₀ 학습 · 응력 제외. 설정은 공통 5-fold CV로 선정하고 전체 Train으로 다시 학습했습니다. 아래 값은 고정 Test에서 시드 0–4의 평균 ± 표준편차입니다.</p>']
 for t,title,cols in [('stability','안정성: partial bi → full mono',[('f1','F1'),('precision','Precision'),('recall','Recall'),('balanced_accuracy','Balanced Accuracy')]),('angle_deg','안정각 [°]',[('mae','MAE'),('rmse','RMSE'),('r2','R²')]),('deformation_mm','두 번째 안정점 변위 [mm]',[('mae','MAE'),('rmse','RMSE'),('r2','R²')]),('log10_barrier','역방향 장벽 [log₁₀(mJ)]',[('mae','MAE'),('rmse','RMSE'),('r2','R²')]),('barrier_mJ','역방향 장벽: 원래 단위 복원 [mJ]',[('mae','MAE'),('rmse','RMSE'),('r2','R²')])]:
  q=agg[agg.target==t].sort_values('f1_mean' if t=='stability' else 'mae_mean',ascending=t!='stability')
  content+=[f'<h2>{title}</h2>','<table><thead><tr><th>모델</th><th>표본 수</th>'+''.join(f'<th>{v}</th>' for _,v in cols)+'</tr></thead><tbody>']
  for _,r in q.iterrows():
   cells=''
   for k,_ in cols:
    mean=r[k+'_mean'];std=r[k+'_std'];factor=100 if t=='stability' else 1;suffix='%' if t=='stability' else ''
    cells+=f'<td>{mean*factor:.4f} ± {std*factor:.4f}{suffix}</td>'
   content.append(f'<tr><td>{html.escape(r.label)}</td><td>{r.test_n}</td>{cells}</tr>')
  content.append('</tbody></table>')
 content+=['<h2>교차검증 선정과 Test 순위의 구분</h2>','<p>설정은 Train 내부 교차검증으로만 선택했습니다. Test 표의 순위는 시험 결과를 설명하는 순위이며, Test를 보고 설정을 바꾸지는 않았습니다. 교차검증 우승과 Test 우승이 다르면 두 결과가 일관되지 않은 것이므로 어느 모델이 항상 우수하다고 단정하지 않습니다.</p>','<table><tr><th>출력</th><th>교차검증 기준 우수 모델</th><th>Test 평균 기준 우수 모델</th></tr>']
 for c,w in zip(cv_winners,winners):
  content.append(f"<tr><td>{c['target']}</td><td>{html.escape(c['label'])}</td><td>{html.escape(w['label'])}</td></tr>")
 content.append('</table>')
 content+=['<h2>비교 범위와 해석</h2>','<p>안정성은 두 상태 판정이 모두 알려진 학습 854개/시험 105개 형상을 공통 사용합니다. 회귀는 partial 쌍안정이며 세 정답이 유효한 학습 836개/시험 143개 형상입니다. 안정성 분류의 예측은 진우 버전은 두 상태의 이진 예측을 결합하고, 승민 버전은 기존 3-class 구조의 최빈 조합을 bi→mono 여부로 변환했습니다.</p>','<p>모델군별 3개 설정을 고정하고 동일한 5-fold CV로 선택했습니다. 독립 회귀는 출력별 MAE, 동시 출력 MLP는 각 학습 fold의 출력 표준편차로 나눈 MAE를 동일 가중 평균한 값으로 공통 설정을 선택했습니다. 비교 예산에 맞춰 승민 모델의 기존 20회 무작위 탐색을 3개 고정 설정으로 바꾸고, GP는 설정당 한 시작점에서 최대 60회 L-BFGS 최적화, Referee는 내부 3-fold stacking을 사용했습니다. 따라서 기존 저장 가중치를 그대로 재평가한 실험과 다릅니다.</p>','<p>승민 MLP의 가우시안 손실·5개 신경망 앙상블, GP의 ARD Matérn 5/2, ROM 물리식과 GP 추가 입력, Referee의 다항식·LightGBM·GP 비음수 stacking 구조는 유지했습니다. ROM 계수는 각 fold의 학습 형상 중 완료된 힘 곡선만으로 다시 적합했습니다. 검증·Test 곡선으로 적합하지 않았습니다.</p>','<p>시드 표준편차는 학습 난수에 따른 변동이며, 새로운 시험 표본에 대한 신뢰구간은 아닙니다. 기존 프로젝트에서 사용한 고정 Test를 공통 비교에 재사용했습니다. 이번 실행에서는 Test를 설정 탐색에 사용하지 않았고, Test 순위는 결과 설명용입니다. 회귀 허용오차 통과율이나 역설계 성공률은 이 평가에 포함하지 않았습니다.</p>']
 doc='<!doctype html><html lang="ko"><meta charset="utf-8"><title>통합 Forward 평가</title><style>body{font-family:Arial,"Apple SD Gothic Neo",sans-serif;max-width:1200px;margin:40px auto;padding:0 25px;color:#17202a;line-height:1.65}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border:1px solid #dce1e7;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eaf0f8}tbody tr:first-child{background:#edf8f0}h2{margin-top:36px}</style>'+''.join(content)+'</html>'
 (OUT/'평가 결과.html').write_text(doc,encoding='utf-8')
 print('Report:',len(models),'models complete; winners',[(x['label'],x['target']) for x in winners],flush=True)
if __name__=='__main__':generate()
