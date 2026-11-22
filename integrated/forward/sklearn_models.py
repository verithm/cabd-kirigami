"""Original Jinwoo model families; three prespecified configs, shared five-fold selection."""
import warnings,time
from pathlib import Path
import numpy as np
import joblib
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor,VotingClassifier,VotingRegressor
from sklearn.gaussian_process import GaussianProcessClassifier,GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.svm import SVC,SVR
from xgboost import XGBClassifier,XGBRegressor
from threadpoolctl import threadpool_limits
from common import *
NAMES=['jw_mlp','jw_xgboost','jw_gp','jw_extra_trees','jw_svm','jw_mlp_xgboost']

def estimator(name,k,seed,cls):
 if name=='jw_mlp':
  cfg=[dict(hidden_layer_sizes=(32,16),alpha=.05),dict(hidden_layer_sizes=(64,32),alpha=.05),dict(hidden_layer_sizes=(32,16),alpha=.5)][k]
  m=(MLPClassifier if cls else MLPRegressor)(**cfg,solver='lbfgs',max_iter=1500,random_state=seed)
  return make_pipeline(StandardScaler(),m if cls else TransformedTargetRegressor(regressor=m,transformer=StandardScaler()))
 if name=='jw_xgboost':
  cfg=[dict(n_estimators=300,max_depth=3,learning_rate=.05),dict(n_estimators=500,max_depth=2,learning_rate=.03),dict(n_estimators=300,max_depth=4,learning_rate=.05)][k]
  return (XGBClassifier if cls else XGBRegressor)(**cfg,subsample=.85,colsample_bytree=.9,random_state=seed,n_jobs=1)
 if name=='jw_extra_trees':return (ExtraTreesClassifier if cls else ExtraTreesRegressor)(n_estimators=300,min_samples_leaf=[2,1,4][k],random_state=seed,n_jobs=1)
 if name=='jw_svm':
  if cls:return make_pipeline(StandardScaler(),SVC(C=[3,10,30][k],probability=True,random_state=seed))
  return make_pipeline(StandardScaler(),TransformedTargetRegressor(regressor=SVR(C=[3,10,30][k],epsilon=.1),transformer=StandardScaler()))
 if name=='jw_gp':
  kernel=Matern(length_scale=[2.,1.,4.][k],nu=1.5)
  m=GaussianProcessClassifier(kernel=kernel,optimizer=None,random_state=seed) if cls else GaussianProcessRegressor(kernel=kernel,alpha=.01,normalize_y=True,optimizer=None)
  return make_pipeline(StandardScaler(),m)
 if name=='jw_mlp_xgboost':
  members=[('mlp',estimator('jw_mlp',k,seed,cls)),('xgb',estimator('jw_xgboost',k,seed,cls))]
  return VotingClassifier(members,voting='soft') if cls else VotingRegressor(members)
 raise ValueError(name)

def arrays(d,target):
 if target=='stability':
  q=d[d.label.notna()];x=q[FEATURES].to_numpy(float)
  xx=np.concatenate([np.column_stack([x,np.zeros(len(x))]),np.column_stack([x,np.ones(len(x))])])
  yy=np.r_[q.partial_label.eq('bistable'),q.full_label.eq('bistable')].astype(int)
  return q,xx,yy
 q=d[d.is_reg];return q,np.column_stack([q[FEATURES].to_numpy(float),np.zeros(len(q))]),q[target].to_numpy(float)

def evaluate(model,q,x,target):
 if target=='stability':
  labels=model.predict(x);n=len(q);return (labels[:n]==1)&(labels[n:]==0)
 return model.predict(x)

def run(name):
 warnings.filterwarnings('ignore');d=load_data();tr=d[d.role=='train'];te=d[d.role=='test'];out=OUT/'models'/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():return name+' exists'
 allcv=[];results=[];predrows=[]
 with threadpool_limits(limits=1):
  for target in ['stability',*TARGETS]:
   cls=target=='stability';cv=[]
   for k in range(3):
    cache=out/f'cv_{target}_{k}.json'
    if cache.exists():record=json.loads(cache.read_text())
    else:
     actual=[];pred=[];fs=[]
     for fold in range(5):
      q,x,y=arrays(tr[tr.fold!=fold],target);qh,xh,yh=arrays(tr[tr.fold==fold],target)
      model=estimator(name,k,42,cls);model.fit(x,y);ph=evaluate(model,qh,xh,target)
      yh=qh.label.eq('bi>mono').to_numpy() if cls else yh
      actual.extend(yh.tolist());pred.extend(ph.tolist());fs.append((metrics_cls if cls else metrics_reg)(yh,ph))
     metric=(metrics_cls if cls else metrics_reg)(actual,pred)
     record=dict(model=name,target=target,config=k,folds=fs,**metric);dump(cache,record)
    cv.append(record);allcv.append(record);print('CV',name,target,k,record.get('f1',record.get('mae')),flush=True)
   best=(max(cv,key=lambda z:(z['f1'],z['precision'])) if cls else min(cv,key=lambda z:z['mae']))['config']
   q,x,y=arrays(tr,target);qt,xt,yt=arrays(te,target)
   for seed in SEEDS:
    model=estimator(name,best,seed,cls);model.fit(x,y);p=evaluate(model,qt,xt,target)
    actual=qt.label.eq('bi>mono').to_numpy() if cls else yt
    score=(metrics_cls if cls else metrics_reg)(actual,p)
    results.append(dict(model=name,target=target,seed=seed,config=best,**score))
    for geom,a,b in zip(qt.geom_id,actual,p):predrows.append(dict(model=name,target=target,seed=seed,geom_id=geom,actual=float(a),predicted=float(b)))
    joblib.dump(model,out/f'{target}_s{seed}.joblib')
    print('TEST',name,target,seed,score,flush=True)
 pd.DataFrame(predrows).to_csv(out/'test_predictions.csv',index=False)
 dump(out/'complete.json',dict(model=name,cv=allcv,test=results))
 return name+' complete'
