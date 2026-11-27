"""Plot saved 50-trial history only; never fit, tune or score a model."""
import os
from pathlib import Path
import hashlib
import json
import sys
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from runtime_paths import RUNS
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

NAMES=['mlp','xgboost','gp','rom_mlp','rom_xgboost','rom_gp']
LABELS=['MLP','XGBoost','GP','ROM+MLP','ROM+XGBoost','ROM+GP']
COLORS=['#2166AC','#D97706','#178568','#9A4F96','#C54C54','#578C9C']
TASKS=['classification','angle_displacement','barrier','stress']
TITLES=['A  안정성 분류 · F1 ↑','B  안정각·변위 공동 회귀 · 오차 ↓',
        'C  에너지 장벽 회귀 · 오차 ↓','D  힌지 최대 응력 회귀 · 오차 ↓']
UNITS=['누적 최고 OOF F1','누적 최소 정규화 MAE','누적 최소 log10 장벽 MAE','누적 최소 MAE (MPa)']


def main():
    root=RUNS/'forward-refinement'
    source=root/'search_history_50trials.csv'
    h=pd.read_csv(source)
    assert len(h)==1200
    for name in NAMES:
        for task in TASKS:
            q=h[(h.candidate==name)&(h.task==task)].sort_values('completed_trial')
            np.testing.assert_array_equal(q.completed_trial,np.arange(1,51))
            recomputed=np.maximum.accumulate(q.objective) if task=='classification' else np.minimum.accumulate(q.objective)
            np.testing.assert_allclose(q.best_so_far,recomputed,atol=1e-12,rtol=1e-12)
    available={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ['Apple SD Gothic Neo','AppleGothic','Arial Unicode MS'] if f in available),'DejaVu Sans')
    plt.rcParams.update({'font.family':font,'font.size':12,'axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'path'})
    dest=root/'figures50';dest.mkdir(exist_ok=True)
    handles=[Line2D([0],[0],color=c,lw=2.5,ls='--' if n.startswith('rom_') else '-',label=l)
             for n,l,c in zip(NAMES,LABELS,COLORS)]
    outputs=[]
    for mode in ['full','late_improvement']:
        fig,axes=plt.subplots(2,2,figsize=(15.5,10.5))
        fig.subplots_adjust(left=.085,right=.97,bottom=.21,top=.80,hspace=.46,wspace=.25)
        full=mode=='full'
        fig.suptitle('Forward 모델 · 50회 하이퍼파라미터 튜닝 전체 추이' if full else
                     '후반부 튜닝 효과 · 20회 최고 성능 대비 추가 개선',fontsize=22,fontweight='bold',y=.97)
        fig.text(.5,.915,'6개 후보 × 4개 작업 · 동일한 Train 내부 5-fold · 각 후보·작업 50회',
                 ha='center',fontsize=13,color='#475569')
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.885),ncol=6,frameon=False,fontsize=12)
        for ax,task,title,unit in zip(axes.flat,TASKS,TITLES,UNITS):
            for name,c in zip(NAMES,COLORS):
                q=h[(h.candidate==name)&(h.task==task)].sort_values('completed_trial')
                if full:
                    x,y=q.completed_trial.to_numpy(),q.best_so_far.to_numpy()
                else:
                    baseline=q.loc[q.completed_trial.eq(20),'best_so_far'].item()
                    q=q[q.completed_trial.ge(20)]
                    x=q.completed_trial.to_numpy()
                    y=(q.best_so_far.to_numpy()-baseline)*100 if task=='classification' else (baseline-q.best_so_far.to_numpy())/baseline*100
                ax.step(x,y,where='post',color=c,lw=2.5 if name=='mlp' else 2,
                        ls='--' if name.startswith('rom_') else '-',alpha=.95,zorder=5 if name=='mlp' else 3)
                ax.plot(x[-1],y[-1],'o',color=c,ms=5,zorder=6)
            for x in ([10,20,40] if full else [40]):
                ax.axvline(x,color='#94A3B8',lw=1,ls=':',zorder=0)
            ax.set_title(title if full else title.replace('· F1 ↑','· F1 증가 ↑').replace('· 오차 ↓','· 오차 감소율 ↑'),
                         loc='left',fontweight='bold',fontsize=14,pad=12)
            ax.set_xlabel('완료한 trial 수 (하이퍼파라미터 설정 평가 횟수)')
            ax.set_ylabel(unit if full else ('F1 증가 (%p)' if task=='classification' else '20회 대비 오차 감소율 (%)'))
            ax.set_xlim(1 if full else 20,50)
            ax.set_xticks([1,10,20,30,40,50] if full else [20,25,30,35,40,45,50])
            ax.grid(axis='both',alpha=.18);ax.margins(y=.12)
            if not full:ax.axhline(0,color='#64748B',lw=.8)
            ax.tick_params(labelsize=11)
        caption=('선 = 해당 회차까지의 최고 검증 성능. 개별 trial 점수나 학습 epoch별 loss가 아님.\n'
                 '각도·변위: fold별 Train 표준편차로 정규화한 두 MAE의 평균. 장벽·응력: fold MAE 평균.\n'
                 '1–10회 = 기본 설정 1개 + 무작위 탐색 9개, 11–50회 = TPE 탐색. Test 평가값은 포함하지 않음.') if full else (
                 '각 모델의 20회 시점 최고 점수를 기준으로 계산. 모두 위로 갈수록 추가 개선이 큼.\n'
                 '모델별 출발 성능이 다르므로 이 그림은 최종 성능 순위가 아닌 추가 튜닝 효과를 비교함.\n'
                 '곡선 정체는 현재 탐색 범위의 관찰 결과이며 완전 수렴 또는 전역 최적해를 뜻하지 않음.')
        fig.text(.085,.025,caption,ha='left',va='bottom',fontsize=11,color='#475569',linespacing=1.6)
        for ext in ['png','pdf','svg']:
            path=dest/f'tuning50_{mode}.{ext}'
            fig.savefig(path,dpi=190,facecolor='white');outputs.append(path)
        plt.close(fig)
    (dest/'figure_audit.json').write_text(json.dumps({'source':str(source),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'records':len(h),
        'all_24_histories_have_exactly_50':True,'best_so_far_recomputed':True,
        'model_fits_performed':0,'test_data_used':False,'font':font,
        'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}},ensure_ascii=False,indent=2)+'\n')
    print(dest)


if __name__=='__main__':main()
