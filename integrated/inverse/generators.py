"""Two original CVAE families and categorical conditional GMM; Forward-free training losses."""
import copy,warnings,time
import numpy as np
import torch
from torch import nn
from scipy.special import expit,logit,logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from shared import *
from cabd.models.cvae import CVAE as OriginalSMCVAE

CONFIGS={
 'jw_cvae':[dict(beta=.001,epochs=300),dict(beta=.005,epochs=300),dict(beta=.02,epochs=300)],
 'sm_cvae':[dict(width=64,depth=2,latent=2,beta=.5,lr=.001,wd=1e-4,epochs=2000),dict(width=128,depth=2,latent=3,beta=1.,lr=.001,wd=1e-4,epochs=2000),dict(width=128,depth=3,latent=4,beta=2.,lr=.0003,wd=1e-4,epochs=2000)],
 'conditional_gmm':[dict(components=2),dict(components=4),dict(components=8)]
}

class JWCVAE(nn.Module):
 def __init__(self):
  super().__init__();self.enc=nn.Sequential(nn.Linear(12,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,4));self.dec=nn.Sequential(nn.Linear(6,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,8),nn.Sigmoid())
 def loss(self,x,c,beta):
  mu,lv=self.enc(torch.cat([x,c],1)).chunk(2,1);lv=lv.clamp(-10,5);z=mu+torch.exp(.5*lv)*torch.randn_like(mu)
  reconstruction=self.dec(torch.cat([z,c],1));rec=(reconstruction-x).square().mean();kl=.5*(lv.exp()+mu.square()-1-lv).mean()
  return rec+beta*kl,rec,kl

def arrays(d,method,mu=None,sd=None):
 x=normalized(d[FEATURES].to_numpy(float));y=d[TARGETS].to_numpy(float)
 mu=y.mean(0) if mu is None else mu;sd=y.std(0) if sd is None else sd
 c=np.column_stack([(y-mu)/sd,d.label.eq('bi>mono').to_numpy(float)])
 return (x if method=='jw_cvae' else 2*x-1).astype('float32'),c.astype('float32'),mu,sd

def torch_read(path):return torch.load(path,map_location='cpu',weights_only=False)

def fit_neural(method,d,cfg,seed,path,validation=None,fixed_epochs=None):
 torch.set_num_threads(1);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  saved=torch_read(path)
  if saved.get('done'):return saved
 x,c,mu,sd=arrays(d,method);x=torch.tensor(x);c=torch.tensor(c)
 vx=vc=None
 if validation is not None:
  vx,vc,_,_=arrays(validation,method,mu,sd);vx=torch.tensor(vx);vc=torch.tensor(vc)
 torch.manual_seed(seed);model=JWCVAE() if method=='jw_cvae' else OriginalSMCVAE(cfg)
 optimizer=torch.optim.Adam(model.parameters(),lr=.002 if method=='jw_cvae' else cfg['lr'],weight_decay=0 if method=='jw_cvae' else cfg['wd'])
 last=0;history=[];best=float('inf');best_ep=0;best_state=None;wait=0
 if path.exists():
  saved=torch_read(path);assert saved['cfg']==cfg and saved['seed']==seed
  model.load_state_dict(saved['state']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);last=saved['epoch'];history=saved['history'];best=saved['best'];best_ep=saved['best_epoch'];best_state=saved['best_state'];wait=saved['wait']
 epochs=int(fixed_epochs or cfg['epochs']);started=time.perf_counter()
 def save(ep,done=False):
  payload=dict(method=method,cfg=cfg,seed=seed,state=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),epoch=ep,history=history,best=best,best_epoch=best_ep,best_state=best_state,wait=wait,mu=mu,sd=sd,training_ids=d.geom_id.tolist(),done=done)
  if done and method=='sm_cvae' and validation is not None and best_state is not None:payload['state']=best_state
  tmp=path.with_suffix(f'.{os.getpid()}.tmp');torch.save(payload,tmp);tmp.replace(path);return payload
 ep=last
 for ep in range(last+1,epochs+1):
  if method=='jw_cvae':
   losses=[];recs=[];kls=[]
   for ids in torch.randperm(len(x)).split(32):
    loss,rec,kl=model.loss(x[ids],c[ids],cfg['beta']*min(1,ep/60));optimizer.zero_grad();loss.backward();optimizer.step();losses.append(loss.item());recs.append(rec.item());kls.append(kl.item())
   train_loss=float(np.mean(losses))
  else:
   optimizer.zero_grad();loss=model.loss(x,c);loss.backward();optimizer.step();train_loss=float(loss.item())
  assert np.isfinite(train_loss),f'Nonfinite {method} training loss'
  if ep%10==0 or ep==epochs:
   val_loss=None
   if vx is not None:
    with torch.no_grad(),torch.random.fork_rng():
     torch.manual_seed(123)
     vals=[(model.loss(vx,vc,cfg['beta'])[0] if method=='jw_cvae' else model.loss(vx,vc)).item() for _ in range(3)]
     val_loss=float(np.mean(vals));assert np.isfinite(val_loss)
    if val_loss<best-1e-5:best=val_loss;best_ep=ep;best_state=copy.deepcopy(model.state_dict());wait=0
    else:wait+=10
   history.append(dict(epoch=ep,train_loss=train_loss,validation_loss=val_loss,elapsed_seconds=time.perf_counter()-started))
  if ep%50==0:save(ep)
  if method=='sm_cvae' and validation is not None and wait>=200 and ep>=300:break
 payload=save(ep,True);print('FIT',method,seed,'epochs',ep,'best epoch',best_ep,flush=True);return payload


def fit_gmm(d,cfg,seed,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():return joblib.load(path)
 y=d[TARGETS].to_numpy(float);mu=y.mean(0);sd=y.std(0)
 x=logit(np.clip(normalized(d[FEATURES].to_numpy(float)),.001,.999));joint=np.column_stack([x,(y-mu)/sd]);flag=d.label.eq('bi>mono').to_numpy(int)
 models={}
 with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  for label in [0,1]:models[label]=GaussianMixture(n_components=cfg['components'],reg_covar=.01,n_init=5,random_state=seed,max_iter=500).fit(joint[flag==label])
 payload=dict(method='conditional_gmm',cfg=cfg,seed=seed,mu=mu,sd=sd,models=models,training_ids=d.geom_id.tolist(),done=True)
 tmp=path.with_suffix(f'.{os.getpid()}.tmp');joblib.dump(payload,tmp);tmp.replace(path);print('FIT GMM',seed,cfg,flush=True);return payload

class Generator:
 def __init__(self,path):
  self.path=Path(path)
  self.saved=joblib.load(path) if self.path.suffix=='.joblib' else torch_read(path)
  self.method=self.saved['method'];self.mu=self.saved['mu'];self.sd=self.saved['sd']
  if self.method!='conditional_gmm':
   self.model=JWCVAE() if self.method=='jw_cvae' else OriginalSMCVAE(self.saved['cfg'])
   self.model.load_state_dict(self.saved['state']);self.model.eval();self.model.requires_grad_(False)
 @torch.inference_mode()
 def sample(self,target,n,seed):
  rng=np.random.default_rng(seed);condition=np.r_[(np.asarray(target)-self.mu)/self.sd,1.].astype('float32')
  if self.method=='conditional_gmm':
   g=self.saved['models'][1];t=condition[:3];means=[];covs=[];lw=[]
   for w,m,C in zip(g.weights_,g.means_,g.covariances_):
    cross=np.linalg.solve(C[8:,8:],C[8:,:8]).T
    means.append(m[:8]+cross@(t-m[8:]));v=C[:8,:8]-cross@C[8:,:8];v=(v+v.T)/2;ev,evec=np.linalg.eigh(v);covs.append((evec*np.maximum(ev,1e-9))@evec.T)
    lw.append(np.log(w)+multivariate_normal.logpdf(t,m[8:],C[8:,8:]))
   components=rng.choice(len(lw),n,p=np.exp(lw-logsumexp(lw)));a=np.empty((n,8))
   for j in range(len(lw)):
    mask=components==j;a[mask]=rng.multivariate_normal(means[j],covs[j],size=mask.sum())
   return expit(a)
  latent=2 if self.method=='jw_cvae' else self.saved['cfg']['latent']
  z=torch.tensor(rng.normal(size=(n,latent)),dtype=torch.float32);c=torch.tensor(np.tile(condition,(n,1)))
  if self.method=='jw_cvae':u=self.model.dec(torch.cat([z,c],1)).numpy()
  else:u=(self.model.decode(z,c)[0].numpy()+1)/2
  assert np.isfinite(u).all();return u.astype(float)


def train_method(method):
 from threadpoolctl import threadpool_limits
 torch.set_num_threads(1)
 with threadpool_limits(limits=1):
  d=pd.read_csv(OUT/'data/training.csv');fit=d[d.inverse_split=='fit'];val=d[d.inverse_split=='validation'];goals=pd.read_csv(OUT/'data/validation_goals.csv')
  selection_path=OUT/'tuning'/method/'selection.json';records=[];fwd=Forward();geometry=Geometry()
  for k,cfg in enumerate(CONFIGS[method]):
   metrics_path=OUT/'tuning'/method/f'config{k}.json';ext='.joblib' if method=='conditional_gmm' else '.pt';path=OUT/'tuning'/method/f'config{k}{ext}'
   if metrics_path.exists():records.append(json.loads(metrics_path.read_text()));continue
   start=time.perf_counter()
   if method=='conditional_gmm':fit_gmm(fit,cfg,42,path)
   else:fit_neural(method,fit,cfg,42,path,validation=val)
   gen=Generator(path);hits=[];fractions=[];best=[]
   for i,r in goals.iterrows():
    goal=r[TARGETS].to_numpy(float);u=gen.sample(goal,VALID_BUDGET,12345+i);y,p=fwd.predict(u);score=score_pool(u,y,p,goal,geometry)
    hits.append(bool(score['passed'].any()));fractions.append(float(score['passed'].mean()));eligible=bound_mask(u)&(p>=P_MIN)
    best.append(float(score['distance'][eligible].min()) if eligible.any() else float('inf'))
   record=dict(method=method,config=k,cfg=cfg,goal_success_rate=float(np.mean(hits)),candidate_pass_rate=float(np.mean(fractions)),median_best_error=float(np.median(best)),seconds=time.perf_counter()-start)
   dump(metrics_path,record);records.append(record);print('TUNE',record,flush=True)
  winner=max(records,key=lambda r:(r['goal_success_rate'],r['candidate_pass_rate'],-r['median_best_error']))
  dump(selection_path,winner)
  k=winner['config'];cfg=CONFIGS[method][k]
  tune=Generator(OUT/'tuning'/method/f"config{k}{'.joblib' if method=='conditional_gmm' else '.pt'}")
  epochs=(cfg['epochs'] if method=='jw_cvae' else tune.saved['best_epoch']) if method!='conditional_gmm' else None
  folder=OUT/'models'/method;folder.mkdir(parents=True,exist_ok=True)
  for seed in SEEDS:
   path=folder/f"seed{seed}{'.joblib' if method=='conditional_gmm' else '.pt'}"
   if method=='conditional_gmm':fit_gmm(d,cfg,seed,path)
   else:fit_neural(method,d,cfg,seed,path,fixed_epochs=epochs)
  dump(folder/'complete.json',dict(method=method,selected=winner,training_shapes=len(d),seeds=SEEDS,epochs=epochs))
  return method
