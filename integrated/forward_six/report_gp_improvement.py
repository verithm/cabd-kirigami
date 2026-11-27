"""Audit and visualize the separate GP classification diagnostic experiment."""
from pathlib import Path
import json
import hashlib
import sqlite3
import sys
BASE=Path(__file__).resolve().parents[1]
for p in [BASE,BASE/'forward',BASE.parent/'src']:sys.path.insert(0,str(p))
import numpy as np
import pandas as pd
from runtime_paths import RUNS
from common import dump
from forward_six.gp_improvement import DEST,NAMES,VARIANTS,aggregate,pilot_config,training,old_choice
from forward_six.gp_controls import CONTROLS


def main():
    protocol=json.loads((DEST/'protocol.json').read_text())
    for p,digest in protocol['source_data_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,p
    extended=(DEST/'all_pilot_summary.json').exists()
    decisions=json.loads((DEST/('controls_decisions.json' if extended else 'pilot_decisions.json')).read_text())
    summary=json.loads((DEST/('all_pilot_summary.json' if extended else 'pilot_summary.json')).read_text())
    if extended:
        controls_protocol=json.loads((DEST/'controls_protocol.json').read_text())
        for p,digest in controls_protocol['source_sha256'].items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,p
    data=training();fold_records=0;inner=[];rows=[]
    for n in NAMES:
        for variant in [*VARIANTS,*(CONTROLS if extended else [])]:
            saved=next(r for r in summary if r['candidate']==n and r['variant']==variant)
            cfg=saved['config']
            result=aggregate(n,cfg,DEST/('pilot' if variant in VARIANTS else 'controls')/n/variant)
            np.testing.assert_allclose(result['objective'],saved['pilot_f1'],atol=0,rtol=0)
            for f in result['folds']:
                assert set(f['eligible_training_ids'])==set(data.loc[data.fold.ne(f['fold'])&data.label.notna(),'geom_id'])
                for stage in f['diagnostics']['kernel_optimization']:
                    for rec in stage:inner.append({'candidate':n,'variant':variant,**rec})
                fold_records+=1
            rows.append({'candidate':n,'phase':'baseline','variant':'original_best_of_50','f1':old_choice(n)['objective']}) if variant==VARIANTS[0] else None
            rows.append({'candidate':n,'phase':'pilot','variant':variant,'f1':result['objective']})
    newstudies={}
    for n in NAMES:
        studyroot=DEST/'revised50' if extended else DEST
        path=studyroot/'studies'/n/'study.sqlite3'
        if path.exists():
            with sqlite3.connect(path) as db:
                states=dict(db.execute('select state,count(*) from trials group by state'))
            newstudies[n]=states
            best=json.loads((studyroot/'studies'/n/'best.json').read_text())
            rows.append({'candidate':n,'phase':'new_tpe','variant':'new_50_best','f1':best['objective']})
            if (DEST/'complete.json').exists():assert states.get('COMPLETE')==50 and states.get('RUNNING',0)==0
        if not decisions[n]['run_new_50']:assert not path.exists()
    pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False)
    pd.DataFrame(inner).to_csv(DEST/'pilot_inner_optimization.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ['Apple SD Gothic Neo','AppleGothic','Arial Unicode MS'] if f in names),'DejaVu Sans')
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'pdf.fonttype':42})
    groups=[(VARIANTS,['상한만 확대','상한 + 반복 확대','RBF + 상한 확대','Matern 0.5 + 상한 확대'])]
    if extended:groups.append((CONTROLS,['RBF · 기존 상한','Matern 0.5 · 기존 상한','신호 상한 100','신호 상한 10,000']))
    fig,axes=plt.subplots(len(groups),2,figsize=(13,10 if extended else 5.8),squeeze=False)
    fig.subplots_adjust(left=.08,right=.97,top=.83 if extended else .76,bottom=.24 if extended else .28,wspace=.25,hspace=.75)
    for ri,(variants,nlabels) in enumerate(groups):
        for ci,n in enumerate(NAMES):
            ax=axes[ri,ci]
            variants_full=['original_best_of_50',*variants];labels=['기존 50회 최고',*nlabels]
            y=[next(r['f1'] for r in rows if r['candidate']==n and r['variant']==v) for v in variants_full]
            ax.bar(np.arange(5),y,color=['#64748B','#D29A42','#A46EAD','#27879B','#3F875B'])
            ax.axhline(y[0],color='#475569',ls='--',lw=1)
            for j,v in enumerate(y):ax.text(j,v+.002,f'{v:.4f}',ha='center',fontsize=10)
            ax.set_xticks(np.arange(5),labels,rotation=25,ha='right',fontsize=10)
            ax.set_title(('GP' if n=='gp' else 'ROM+GP')+(' · 상한 확대 실험' if ri==0 else ' · 단일 요인 대조 실험'),fontsize=14)
            ax.set_ylabel('Train 내부 5-fold OOF F1 ↑')
            ax.set_ylim(max(0,min(r['f1'] for r in rows)-.035),max(r['f1'] for r in rows)+.025)
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('GP 분류 개선 파일럿 · 내부 경계 / 최적화 횟수 / 커널 비교',fontsize=19,y=.96)
    fig.text(.5,.90 if extended else .85,'같은 Train 분할·분류 구조·판정 규칙 · Test 미사용',ha='center',fontsize=12,color='#475569')
    fig.text(.08,.055,'기존: 50회 중 최고 설정. 파일럿: 변경 방법별 설정 1개를 동일 5-fold로 평가.\n'
             '내부 상한: 신호 1e3→1e6, 길이 1e3→1e5. 반복 확대: 최대 60→150회.\n'
             '파일럿은 새 탐색의 유망성을 진단하며, GP 전체의 최적 성능을 보장하지 않음.',fontsize=10,color='#475569',linespacing=1.5)
    fig.savefig(DEST/'pilot_comparison.png',dpi=180);fig.savefig(DEST/'pilot_comparison.pdf');plt.close(fig)
    original_studies=0
    for name in ['mlp','xgboost','gp','rom_mlp','rom_xgboost','rom_gp']:
        for task in ['classification','angle_displacement','barrier','stress']:
            parent='forward-six' if task=='classification' else 'forward-grouped'
            with sqlite3.connect(RUNS/parent/'tuning/studies'/name/task/'study.sqlite3') as db:
                states=dict(db.execute('select state,count(*) from trials group by state'))
            assert states.get('COMPLETE')==50 and states.get('RUNNING',0)==0
            original_studies+=1
    old_manifest=json.loads((RUNS/'forward-final50/artifact_manifest.json').read_text())
    for relative,digest in old_manifest.items():
        assert hashlib.sha256((RUNS/'forward-final50'/relative).read_bytes()).hexdigest()==digest,relative
    audit={'pilot_fold_records_recomputed':fold_records,'all_pilot_metrics_verified':True,
        'train_only_ids_verified':True,'source_data_hashes_verified':True,'test_scored':False,
        'original_studies_verified_at_50':original_studies,'original_final_test_artifacts_unchanged':True,
        'inner_optimization_calls':len(inner),'inner_calls_not_success':sum(not r['success'] for r in inner),
        'new_studies':newstudies,'decisions':{n:decisions[n]['run_new_50'] for n in NAMES},
        'claim':'pilot diagnostic; no universal model superiority or statistical significance claim'}
    dump(DEST/'audit.json',audit)
    # Exclude live logs from content integrity snapshots.
    dump(DEST/'manifest.json',{str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in DEST.rglob('*') if p.is_file() and p.suffix not in ['.log','.sqlite3'] and p.name!='manifest.json'})
    print(pd.DataFrame(rows).to_string(index=False));print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
