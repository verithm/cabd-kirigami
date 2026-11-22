"""Seungmin architecture adapters, using common energy targets and CV F1/MAE selection.
Original source files are untouched. Fixed three configurations per family replace legacy tuning.
"""
import warnings,time
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize,nnls,least_squares
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LassoCV,LogisticRegression
from sklearn.model_selection import StratifiedKFold,KFold,train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor,GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern,ConstantKernel,WhiteKernel,RBF,DotProduct
from threadpoolctl import threadpool_limits
from common import *
sys.path.insert(0,str(ROOT/'src'))
from cabd.models import rom
from cabd.models import mlp as original_mlp
BOUNDS=np.array([[.1,.4],[2,4],[.1,.4],[2,4],[5,8],[.8,1.8],[2,4],[.3,.8]])
NAMES=['sm_mlp','sm_gp','sm_rom_gp','sm_referee','sm_referee_nogp']

def scale(d):return (d[FEATURES].to_numpy(float)-BOUNDS[:,0])/(BOUNDS[:,1]-BOUNDS[:,0])

def gp_optimizer(fun,initial,bounds):
 r=minimize(fun,initial,method='L-BFGS-B',jac=True,bounds=bounds,options=dict(maxiter=60,ftol=1e-7))
 return r.x,r.fun

def gp_estimator(k,seed,cls=False,ref=False,n=8):
 length=[1.,.3,3.][k]
 if ref:base=RBF(np.full(n,length),(1e-2,1e3))+ConstantKernel(.1,(1e-4,1e2))*DotProduct(1.,(1e-3,1e2))
 else:base=Matern(np.full(n,length),(1e-2,1e3),nu=2.5)
 kernel=ConstantKernel(1.,(1e-3,1e3))*base
 if cls:return GaussianProcessClassifier(kernel=kernel,n_restarts_optimizer=0,optimizer=gp_optimizer,random_state=seed,max_iter_predict=60)
 return GaussianProcessRegressor(kernel=kernel+WhiteKernel(1e-3,(1e-8,1)),normalize_y=True,n_restarts_optimizer=0,optimizer=gp_optimizer,random_state=seed)

def mlp_fit(d,k,seed,cls):
 import torch
 torch.set_num_threads(1)
 cfg=[dict(width=32,depth=2,lr=.001,wd=1e-4,act='silu'),dict(width=64,depth=2,lr=.001,wd=1e-4,act='silu'),dict(width=128,depth=3,lr=.0003,wd=1e-4,act='tanh')][k]
 q=d[d.label.notna()] if cls else d[d.is_reg]
 X=(2*scale(q)-1).astype('float32')
 y=q.label.map({c:i for i,c in enumerate(CLASSES)}).to_numpy(int) if cls else q[TARGETS].to_numpy(float)
 rng=np.random.default_rng(seed)
 tr,va=train_test_split(np.arange(len(q)),test_size=.2,random_state=seed,stratify=y if cls else None)
 mu=np.zeros(3) if cls else y[tr].mean(0);sd=np.ones(3) if cls else y[tr].std(0)
 yy=y if cls else ((y-mu)/sd).astype('float32')
 torch.manual_seed(seed);net=original_mlp.net(3 if cls else 6,cfg)
 optimizer=torch.optim.Adam(net.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
 tx=torch.tensor(X);ty=torch.tensor(yy,dtype=torch.long if cls else torch.float32)
 best=float('inf');epoch=100;wait=0
 for ep in range(1,1201):
  optimizer.zero_grad();loss=original_mlp.loss_fn('cls' if cls else 'reg',net(tx[tr]),ty[tr]);loss.backward();optimizer.step()
  if ep%20==0:
   with torch.no_grad():o=net(tx[va]).numpy()
   score=-metrics_cls(y[va]==1,o.argmax(1)==1)['f1'] if cls else float(np.abs(o[:,:3]-yy[va]).mean())
   if score<best-1e-6:best=score;epoch=ep;wait=0
   else:wait+=20
   if wait>=160:break
 # Refit ensemble on ALL rows of this training partition, with full-partition scalers.
 mu=np.zeros(3) if cls else y.mean(0);sd=np.ones(3) if cls else y.std(0)
 yy=y if cls else ((y-mu)/sd).astype('float32');nets=[]
 for member in range(5):
  net,_,_=original_mlp.train('cls' if cls else 'reg',X,yy,cfg,seed*100+member,epochs=epoch)
  nets.append(net)
 return dict(kind='joint_mlp',cls=cls,nets=nets,mean=mu,sd=sd,epochs=epoch,cfg=cfg)

def mlp_predict(m,d):
 mean=original_mlp.ensemble('cls' if m['cls'] else 'reg',m['nets'],(2*scale(d)-1).astype('float32'))
 if m['cls']:return mean.argmax(1)==1
 mu,sd=mean;return mu*m['sd']+m['mean']

# Vectorized fit of the original ROM equations. Only completed training curves participate.
def fit_rom(d,tag):
 path=OUT/'rom'/f'{tag}.json'
 if path.exists():return np.array(json.loads(path.read_text())['theta'])
 curves=json.loads((DATA/'completed_force_curves.json').read_text());rows=[]
 for _,r in d.iterrows():
  for state in ['partial','full']:
   key=f'{r.geom_id}|{state}'
   if key in curves:rows.append((rom.geometry(r),state,curves[key]))
 u=rom.U_GRID[None,:]
 geo={k:np.array([r[0][k] for r in rows])[:,None] for k in rows[0][0]}
 ef=np.array([rom.E_DRY if r[1]=='partial' else rom.E_WET for r in rows])[:,None]
 us=[np.array(r[2]['u']) for r in rows];fs=[np.array(r[2]['F']) for r in rows]
 scales=[max(np.max(np.abs(f)),1e-3) for f in fs]
 def residual(theta):
  ct,ca,cf,ch=np.exp(theta[:4]);g=theta[4]
  kt=ct*rom.E_WET*6*geo['t_h']**3/(12*geo['l_tip'])
  ka=ca*rom.E_WET*6*geo['t_a']**3/(12*geo['l_apex'])
  kf=cf*3*ef*(6*geo['t_f']**3/12)/geo['L_fl']**3
  kh=ch*rom.E_WET*6*geo['t_h']/geo['l_tip'];kl=1/(1/kf+1/kh)
  x0=-(geo['L']-geo['t_a']/2)*g;z0=geo['z_t0'];R=np.hypot(x0,z0);xa=x0+u
  zt=np.sqrt(np.maximum(R**2-xa**2,0));phi=np.arctan2(zt,-xa);dp=phi-phi[:,0,None]
  U=kl*(zt-z0)**2+kt*dp**2+.5*ka*(2*dp)**2
  F=np.gradient(U,rom.U_GRID,axis=1)
  return np.concatenate([(np.interp(a,rom.U_GRID,f)-b)/s for a,f,b,s in zip(us,F,fs,scales)])
 r=least_squares(residual,np.array([0,0,0,-5,1.]),bounds=([-8]*4+[.5],[8]*4+[1.5]))
 theta=r.x
 # Validate vectorized residual against the unmodified physical equations.
 a,state,c=rows[0];rr=(np.interp(c['u'],rom.U_GRID,rom.curve(a,state,theta)[1])-c['F'])/scales[0]
 np.testing.assert_allclose(residual(theta)[:len(rr)],rr,atol=1e-10,rtol=1e-9)
 dump(path,dict(theta=theta.tolist(),rms=float(np.sqrt(np.mean(r.fun**2))),states=len(rows),training_ids=d.geom_id.tolist()))
 return theta

def rom_features(d,theta):
 import hashlib
 key=','.join(d.geom_id).encode()+np.ascontiguousarray(d[FEATURES].to_numpy(float)).tobytes()+np.asarray(theta,dtype=float).tobytes()
 p=OUT/'rom'/('features_'+hashlib.sha256(key).hexdigest()[:20]+'.npz')
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  with np.load(p) as z:return z['reg'],z['fp'],z['ff']
 f=rom.features(d,theta)
 b=np.column_stack([f.rom_partial_theta2,f.rom_partial_u2,np.maximum(f.rom_partial_logE,-4.2)])
 fp=f.rom_partial_Fmin.to_numpy();ff=f.rom_full_Fmin.to_numpy()
 np.savez(p,reg=b,fp=fp,ff=ff);return b,fp,ff

def gp_inputs(d,theta=None):
 X=scale(d)
 if theta is None:return X,X,X
 B,fp,ff=rom_features(d,theta)
 return np.column_stack([X,B]),np.column_stack([X,fp]),np.column_stack([X,ff])

# Same member families and nonnegative stacking rule as the original Referee.
# Inner OOF stacking never sees an outer validation fold or final Test.
def member_fit(kind,X,y,k,seed,cls):
 if kind=='gpx':return gp_estimator(k,seed,cls,True,X.shape[1]).fit(X,y)
 if kind=='poly':
  if cls:m=make_pipeline(PolynomialFeatures(3),StandardScaler(),LogisticRegression(C=[.1,1,10][k],max_iter=1000,random_state=seed))
  else:m=make_pipeline(PolynomialFeatures(3),StandardScaler(),LassoCV(cv=3,max_iter=5000,random_state=seed))
  return m.fit(X,y)
 from lightgbm import LGBMClassifier,LGBMRegressor
 return (LGBMClassifier if cls else LGBMRegressor)(n_estimators=[200,400,400][k],max_depth=[2,3,4][k],num_leaves=[4,7,15][k],learning_rate=.03,min_child_samples=10,reg_lambda=1.,random_state=seed,verbosity=-1,n_jobs=1).fit(X,y)

def member_pred(m,X,cls):return m.predict_proba(X)[:,list(m.classes_).index(True)] if cls else m.predict(X)

def stack_fit(X,y,k,seed,cls,nogp=False):
 members=['poly','gbm']+([] if nogp else ['gpx']);P=np.zeros((len(y),len(members)))
 splitter=StratifiedKFold(3,shuffle=True,random_state=seed).split(X,y) if cls else KFold(3,shuffle=True,random_state=seed).split(X)
 for a,b in splitter:
  for j,name in enumerate(members):P[b,j]=member_pred(member_fit(name,X[a],y[a],k,seed,cls),X[b],cls)
 if cls:
  def loss(z):
   w=np.exp(z-z.max());w/=w.sum();p=np.clip(P@w,1e-9,1-1e-9);return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
  z=minimize(loss,np.zeros(len(members)),method='Nelder-Mead').x;w=np.exp(z-z.max());w/=w.sum()
 else:w,_=nnls(P,y);w=w/w.sum() if w.sum() else np.ones(len(w))/len(w)
 return dict(weights=w,models=[member_fit(n,X,y,k,seed,cls) for n in members],cls=cls)

def stack_predict(m,X):return np.column_stack([member_pred(q,X,m['cls']) for q in m['models']])@m['weights']

def fit(name,d,target,k,seed,theta=None):
 cls=target=='stability'
 if name=='sm_mlp':return mlp_fit(d,k,seed,cls)
 q=d[d.label.notna()] if cls else d[d.is_reg]
 a,b,c=gp_inputs(q,theta if name=='sm_rom_gp' else None)
 ref=name.startswith('sm_referee')
 if ref:a=b=c=2*scale(q)-1
 def train(X,y):
  scaler=StandardScaler() if name=='sm_rom_gp' else None
  xx=scaler.fit_transform(X) if scaler is not None else X
  model=stack_fit(xx,y,k,seed,cls,name.endswith('nogp')) if ref else gp_estimator(k,seed,cls,n=xx.shape[1]).fit(xx,y)
  return dict(scaler=scaler,model=model)
 if cls:
  y1=q.partial_label.eq('bistable').to_numpy();idx=y1
  return dict(kind='two_stage',ref=ref,first=train(b,y1),second=train(c[idx],q.full_label.eq('monostable').to_numpy()[idx]),theta=theta if name=='sm_rom_gp' else None)
 return dict(kind='gp_reg',ref=ref,reg=train(a,q[target].to_numpy(float)),theta=theta if name=='sm_rom_gp' else None)

def predict(m,d):
 if m['kind']=='joint_mlp':return mlp_predict(m,d)
 a,b,c=gp_inputs(d,m['theta'])
 if m['ref']:a=b=c=2*scale(d)-1
 def pred(m,X,cls):
  if m['scaler'] is not None:X=m['scaler'].transform(X)
  q=m['model'];return stack_predict(q,X) if mref else (q.predict_proba(X)[:,list(q.classes_).index(True)] if cls else q.predict(X))
 mref=m['ref']
 if m['kind']=='two_stage':
  p1=pred(m['first'],b,True);p2=pred(m['second'],c,True);P=np.column_stack([p1*(1-p2),p1*p2,1-p1]);return P.argmax(1)==1
 return pred(m['reg'],a,False)

def run(name):
 warnings.filterwarnings('ignore');d=load_data();tr=d[d.role=='train'];te=d[d.role=='test'];out=OUT/'models'/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():return name+' exists'
 allcv=[];results=[];predrows=[]
 targets=['stability','joint'] if name=='sm_mlp' else ['stability',*TARGETS]
 with threadpool_limits(limits=1):
  for target in targets:
   cls=target=='stability';cv=[]
   for k in range(3):
    cache=out/f'cv_{target}_{k}.json'
    if cache.exists():record=json.loads(cache.read_text())
    else:
     actual=[];preds=[];fs=[]
     for fold in range(5):
      training=tr[tr.fold!=fold];held=tr[tr.fold==fold];q=held[held.label.notna()] if cls else held[held.is_reg]
      theta=fit_rom(training,f'fold{fold}') if name=='sm_rom_gp' else None
      model=fit(name,training,'stability' if cls else target,k,42,theta);p=predict(model,q)
      y=q.label.eq('bi>mono').to_numpy() if cls else q[TARGETS].to_numpy(float) if target=='joint' else q[target].to_numpy(float)
      if target=='joint':score=reg_report(y,p);score['scaled_mae']=float(np.mean(np.abs(y-p).mean(0)/training[training.is_reg][TARGETS].std(ddof=0).to_numpy()))
      else:score=(metrics_cls if cls else metrics_reg)(y,p)
      fs.append(score);actual.extend(y.tolist());preds.extend(p.tolist());print('FOLD',name,target,k,fold,flush=True)
     if target=='joint':metric=reg_report(np.array(actual),np.array(preds));metric['scaled_mae']=float(np.mean([r['scaled_mae'] for r in fs]))
     else:metric=(metrics_cls if cls else metrics_reg)(actual,preds)
     record=dict(model=name,target=target,config=k,folds=fs,**metric);dump(cache,record)
    cv.append(record);allcv.append(record);print('CV',name,target,k,record.get('f1',record.get('mae',record.get('scaled_mae'))),flush=True)
   best=(max(cv,key=lambda z:(z['f1'],z['precision'])) if cls else min(cv,key=lambda z:z['scaled_mae' if target=='joint' else 'mae']))['config']
   qt=te[te.label.notna()] if cls else te[te.is_reg]
   theta=fit_rom(tr,'full_train') if name=='sm_rom_gp' else None
   for seed in SEEDS:
    model=fit(name,tr,'stability' if cls else target,best,seed,theta);p=predict(model,qt)
    y=qt.label.eq('bi>mono').to_numpy() if cls else qt[TARGETS].to_numpy(float) if target=='joint' else qt[target].to_numpy(float)
    report={target:(metrics_cls if cls else metrics_reg)(y,p)} if target!='joint' else reg_report(y,p)
    for t,score in report.items():
     results.append(dict(model=name,target=t,seed=seed,config=best,**score))
     a=y if target!='joint' else y[:,TARGETS.index(t)] if t in TARGETS else 10**y[:,2]
     b=p if target!='joint' else p[:,TARGETS.index(t)] if t in TARGETS else 10**p[:,2]
     for geom,aa,bb in zip(qt.geom_id,a,b):predrows.append(dict(model=name,target=t,seed=seed,geom_id=geom,actual=float(aa),predicted=float(bb)))
    joblib.dump(model,out/f'{target}_s{seed}.joblib');print('TEST',name,target,seed,report,flush=True)
 pd.DataFrame(predrows).to_csv(out/'test_predictions.csv',index=False)
 dump(out/'complete.json',dict(model=name,cv=allcv,test=results));return name+' complete'
