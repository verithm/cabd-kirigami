"""Fixed data, Forward, official geometry and evaluation rules for the inverse benchmark."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import json,sys,hashlib,time,math
from pathlib import Path
import numpy as np
import pandas as pd
import torch,joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime_paths import ROOT,RUNS,reference,resolve_reference
sys.path.insert(0,str(ROOT/'integrated/forward'))
sys.path.insert(0,str(ROOT/'src'))
from common import FEATURES,TARGETS,load_data
from cabd.geometry import params as G
OUT=RUNS/'unified-inverse'
FORWARD=RUNS/'unified-forward/models/sm_mlp'
BOUNDS=np.array([G.RANGES[k] for k in FEATURES],dtype=float)
LO,SPAN=BOUNDS[:,0],BOUNDS[:,1]-BOUNDS[:,0]
_protocol_path=OUT/'protocol.json'
PROTOCOL=json.loads((_protocol_path if _protocol_path.exists() else ROOT/'integrated/configs/inverse_initial.json').read_text())
TOL=np.array([PROTOCOL['tolerances'][k] for k in TARGETS])
P_MIN=PROTOCOL['minimum_predicted_bi_mono_probability']
SEEDS=[0,1,2]
BUDGET=4096
VALID_BUDGET=512
METHODS=['jw_cvae','sm_cvae','conditional_gmm','differential_evolution','slsqp']
LABELS={'jw_cvae':'진우 CVAE','sm_cvae':'승민 CVAE','conditional_gmm':'조건부 GMM','differential_evolution':'Forward + DE','slsqp':'Forward + SLSQP'}

def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda v:v.item() if hasattr(v,'item') else v));tmp.replace(path)

def to_phys(u):return LO+np.asarray(u)*SPAN

def normalized(x):return (np.asarray(x)-LO)/SPAN

def bound_mask(u):return np.isfinite(u).all(1)&(u>=0).all(1)&(u<=1).all(1)

def fast_margins(u):
 u=np.atleast_2d(u);th,lh,ta,la,L,tf,c,rho=to_phys(u).T
 num=12.7-tf-lh-th-la/2;den=L-ta;a=np.arctan2(num,den);k=np.tan((np.pi/2-a)/2)
 rs=rho*lh/2;du=rs*k;dl=rho*la/2
 lapex=la-2*dl;hyp=den/np.cos(a)-dl-du
 xt=np.where(du>=th,-rs+np.sqrt(np.maximum(rs**2-(du-th)**2,0)),0)
 return np.column_stack([num,lh*(1-rho),lapex,hyp,(xt+th+1e-9)/.1,(th+rs-du+1e-9)/.1,L-th-rs,lapex-.1+1e-9])

class Forward:
 def __init__(self):
  torch.set_num_threads(1)
  selection_path=RUNS/'unified-forward/forward_selection.json'
  if not selection_path.exists():raise FileNotFoundError('Run report-forward or provide forward_selection.json with the selected weights')
  selected=json.loads(selection_path.read_text())['outputs']
  if any(selected[t]['model']!='sm_mlp' for t in ['stability',*TARGETS]):raise ValueError('This frozen inverse adapter requires the CV-selected sm_mlp for all outputs; review adapters for a different selection')
  self.reg=joblib.load(FORWARD/'joint_s0.joblib');self.cls=joblib.load(FORWARD/'stability_s0.joblib')
  for m in self.reg['nets']+self.cls['nets']:m.eval();m.requires_grad_(False)
 @torch.inference_mode()
 def predict(self,u):
  u=np.atleast_2d(u);ys=[];ps=[]
  for start in range(0,len(u),4096):
   x=torch.tensor(2*u[start:start+4096]-1,dtype=torch.float32)
   mean=torch.stack([m(x)[:,:3] for m in self.reg['nets']]).mean(0).numpy()
   probs=torch.stack([torch.softmax(m(x),1) for m in self.cls['nets']]).mean(0).numpy()
   ys.append(mean*self.reg['sd']+self.reg['mean']);ps.append(probs[:,1])
  return np.concatenate(ys),np.concatenate(ps)

class Geometry:
 def __init__(self):self.memo={};self.checked=0
 def one(self,u):
  key=np.asarray(u,dtype=float).tobytes()
  if key in self.memo:return self.memo[key]
  if not bound_mask(np.atleast_2d(u))[0]:self.memo[key]=False;return False
  p=dict(zip(FEATURES,to_phys(u)));p['t_w']=G.T_W
  original=G._dedup;cache={}
  def dedup(points,tol=1e-9):
   k=(id(points),tol)
   if k not in cache:
    keep=[0];coords=points.tolist()
    for i,(x,z) in enumerate(coords[1:],1):
     a,b=coords[keep[-1]]
     if math.hypot(x-a,z-b)>tol:keep.append(i)
    x,z=coords[keep[-1]];a,b=coords[keep[0]]
    if len(keep)>1 and math.hypot(x-a,z-b)<=tol:keep.pop()
    cache[k]=(points,points[keep])
   return cache[k][1]
  try:
   G._dedup=dedup;ok=all(G.check_constraints(p).values())
  except (ValueError,ZeroDivisionError,FloatingPointError):ok=False
  finally:G._dedup=original
  self.memo[key]=bool(ok);self.checked+=1;return bool(ok)
 def many(self,u):return np.array([self.one(row) for row in u],dtype=bool)

def population():
 d=load_data();train=d[(d.role=='train')&d.is_reg&d.label.isin(['bi>bi','bi>mono'])].copy()
 test=d[(d.role=='test')&d.is_reg&d.label.eq('bi>mono')].copy()
 assert len(train)==788 and len(test)==49
 assert not set(train.geom_id)&set(test.geom_id)
 return train.reset_index(drop=True),test.reset_index(drop=True)

def prepare():
 from sklearn.model_selection import train_test_split
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'data').mkdir(parents=True,exist_ok=True)
 tr,te=population();flag=tr.label.eq('bi>mono').to_numpy(int)
 a,b=train_test_split(np.arange(len(tr)),test_size=.2,stratify=flag,random_state=42)
 tr['inverse_split']='fit';tr.loc[b,'inverse_split']='validation';tr.to_csv(OUT/'data/training.csv',index=False)
 val=tr.iloc[b];val=val[val.label=='bi>mono'].sort_values('angle_deg')
 val=val.iloc[np.linspace(0,len(val)-1,20).round().astype(int)].copy()
 val[['geom_id',*TARGETS]].rename(columns={'geom_id':'goal_id'}).to_csv(OUT/'data/validation_goals.csv',index=False)
 te[['geom_id',*TARGETS]].rename(columns={'geom_id':'goal_id'}).to_csv(OUT/'data/test_goals.csv',index=False)
 te[['geom_id',*FEATURES,*TARGETS]].to_csv(OUT/'data/known_test_witnesses.csv',index=False)
 settings=dict(methods=METHODS,seeds=SEEDS,budget_per_goal=BUDGET,validation_budget=VALID_BUDGET,forward_seed=0,forward_members=5,goal_count=len(te),training_shapes=len(tr),training_positive=int(flag.sum()),training_negative=int((1-flag).sum()),fit_shapes=len(a),validation_shapes=len(b),validation_goals=len(val),tuning_candidates=3,ansys=False,goal_basis='All 49 held-out confirmed bi->mono physical response triples; no Test geometry used for fitting or search warm starts',top_rank_rule='minimum maximum tolerance-normalized response error among jointly passing candidates')
 if 'run_settings' in PROTOCOL:assert PROTOCOL['run_settings']==settings,'Existing run settings differ; use a fresh result folder'
 PROTOCOL['run_settings']=settings;PROTOCOL['status']='running';PROTOCOL['forward_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [FORWARD/'joint_s0.joblib',FORWARD/'stability_s0.joblib']}
 dump(OUT/'protocol.json',PROTOCOL)
 # A fresh diagnostic replaces the historical hard-coded 22/35/17 counts.
 fwd=Forward();units=normalized(te[FEATURES].to_numpy(float));pred,prob=fwd.predict(units)
 response=(np.abs((pred-te[TARGETS].to_numpy(float))/TOL).max(1)<=1);stable=prob>=np.float32(P_MIN);geometry=Geometry().many(units)
 dump(OUT/'data/forward_witness_diagnostic.json',dict(goals=len(te),geometry_valid=int(geometry.sum()),response_ok=int(response.sum()),probability_ok=int(stable.sum()),joint_ok=int((geometry&response&stable).sum())))
 audit=dict(train=len(tr),test_goals=len(te),train_test_overlap=0,train_bi_mono=int(flag.sum()),train_bi_bi=int((1-flag).sum()),data_sha256=hashlib.sha256((RUNS/'unified-forward/data/scalar_targets.csv').read_bytes()).hexdigest())
 dump(OUT/'data/audit.json',audit);print('DATA',audit,flush=True)


def score_pool(u,y,p,goal,geometry):
 errors=np.abs((y-goal)/TOL);distance=errors.max(1);bounds=bound_mask(u)
 surrogate=bounds&(p>=P_MIN)&(distance<=1)
 geom=np.full(len(u),-1,dtype=np.int8);geom[~bounds]=0
 idx=np.flatnonzero(surrogate)
 if len(idx):geom[idx]=geometry.many(u[idx]).astype(np.int8)
 passed=surrogate&(geom==1);selected=None
 if passed.any():
  j=np.flatnonzero(passed)[np.argmin(distance[passed])]
  selected=dict(unit=u[j].tolist(),dimensions=dict(zip(FEATURES,to_phys(u[j]).tolist())),predicted=dict(zip(TARGETS,y[j].tolist())),predicted_barrier_mJ=float(10**y[j,2]),p_bi_mono=float(p[j]),normalized_max_error=float(distance[j]))
 return dict(passed=passed,geometry=geom,distance=distance,errors=errors,selected=selected,surrogate=surrogate)
