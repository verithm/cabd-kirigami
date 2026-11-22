"""Shared folds, eligible samples and metrics. Test is never used for tuning."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,precision_score,recall_score,balanced_accuracy_score,accuracy_score,mean_absolute_error,mean_squared_error,r2_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime_paths import ROOT,RUNS,reference,resolve_reference
OUT=RUNS/'unified-forward'; DATA=OUT/'data'
FEATURES=['t_h','L_h','t_a','L_a','L','t_f','c','rho']
TARGETS=['angle_deg','deformation_mm','log10_barrier']
CLASSES=['bi>bi','bi>mono','mono>mono']
SEEDS=[0,1,2,3,4]
CV_SEED=42

def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,default=lambda v:v.item() if hasattr(v,'item') else v));tmp.replace(path)

def load_data():
 s=pd.read_csv(DATA/'scalar_targets.csv');jobs=pd.read_csv(DATA/'jobs.csv').set_index('geom_id');rows=[]
 for geom,states in s.groupby('geom_id'):
  r=jobs.loc[geom].to_dict();r['geom_id']=geom;r['label']=None;r['is_reg']=False
  p=states[states.state=='partial'];f=states[states.state=='full']
  r['partial_label']=None if p.empty else p.iloc[0].stability
  r['full_label']=None if f.empty else f.iloc[0].stability
  if r['partial_label'] in ['bistable','monostable'] and r['full_label'] in ['bistable','monostable']:
   r['label']=('bi' if r['partial_label']=='bistable' else 'mono')+'>'+('bi' if r['full_label']=='bistable' else 'mono')
   assert r['label'] in CLASSES,r
  for k in TARGETS:r[k]=np.nan
  if r['partial_label']=='bistable':
   q=p.iloc[0];r.update(angle_deg=q.stable_angle_deg,deformation_mm=q.deformation_applied_mm,log10_barrier=q.log10_reverse_barrier)
   r['is_reg']=all(np.isfinite(r[k]) for k in TARGETS)
  rows.append(r)
 d=pd.DataFrame(rows).sort_values('geom_id').reset_index(drop=True)
 folds=DATA/'folds.csv'
 if not folds.exists():
  tr=d[d.role=='train'];strata=tr.label.fillna('unpaired_unknown')
  assignment={}
  for fold,(_,held) in enumerate(StratifiedKFold(5,shuffle=True,random_state=CV_SEED).split(tr,strata)):
   assignment.update({tr.iloc[i].geom_id:fold for i in held})
  pd.DataFrame([dict(geom_id=g,fold=f) for g,f in assignment.items()]).sort_values('geom_id').to_csv(folds,index=False)
 mapping=pd.read_csv(folds).set_index('geom_id').fold
 d['fold']=d.geom_id.map(mapping).fillna(-1).astype(int)
 assert not(set(d[d.role=='train'].geom_id)&set(d[d.role=='test'].geom_id))
 return d

def metrics_cls(y,p):
 y=np.asarray(y,dtype=bool);p=np.asarray(p,dtype=bool)
 return dict(f1=float(f1_score(y,p,zero_division=0)),precision=float(precision_score(y,p,zero_division=0)),recall=float(recall_score(y,p,zero_division=0)),balanced_accuracy=float(balanced_accuracy_score(y,p)),accuracy=float(accuracy_score(y,p)),n=len(y),tp=int((y&p).sum()),fp=int((~y&p).sum()),fn=int((y&~p).sum()),tn=int((~y&~p).sum()))

def metrics_reg(y,p):
 y=np.asarray(y);p=np.asarray(p)
 return dict(mae=float(mean_absolute_error(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))),r2=float(r2_score(y,p)),n=len(y))

def reg_report(y,p):
 r={t:metrics_reg(y[:,j],p[:,j]) for j,t in enumerate(TARGETS)}
 r['barrier_mJ']=metrics_reg(10**y[:,2],10**p[:,2]);return r

if __name__=='__main__':
 d=load_data();d.to_csv(DATA/'geometry_targets.csv',index=False)
 config_path=ROOT/'integrated/configs/forward_protocol.json'
 if not (OUT/'protocol.json').exists():dump(OUT/'protocol.json',json.loads(config_path.read_text()))
 summary={}
 for role in ['train','test']:
  q=d[d.role==role];summary[role]=dict(geometries=len(q),classification=int(q.label.notna().sum()),regression=int(q.is_reg.sum()),classes=q.label.value_counts().to_dict())
 dump(DATA/'evaluation_population.json',summary);print(summary)
