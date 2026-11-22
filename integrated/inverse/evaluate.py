"""Equal Forward-evaluation cap, common final geometry/response filtering; no ANSYS."""
import warnings
from scipy.optimize import differential_evolution,minimize
from scipy.special import logsumexp
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from shared import *
from generators import Generator

class BudgetReached(Exception):pass

class Oracle:
 def __init__(self,fwd,goal):self.fwd=fwd;self.goal=goal;self.count=0;self.limit=BUDGET;self.us=[];self.ys=[];self.ps=[]
 def evaluate(self,u):
  u=np.atleast_2d(u)
  if self.count+len(u)>min(BUDGET,self.limit):raise BudgetReached()
  y,p=self.fwd.predict(u);assert np.isfinite(y).all() and np.isfinite(p).all()
  self.us.append(u.copy());self.ys.append(y);self.ps.append(p);self.count+=len(u);return y,p
 def arrays(self):return np.concatenate(self.us),np.concatenate(self.ys),np.concatenate(self.ps)
 def value(self,u):
  y,p=self.evaluate(u);e=np.abs((y-self.goal)/TOL);smooth=logsumexp(20*e,axis=1)/20
  return smooth+20*np.maximum(P_MIN-p,0)+20*np.maximum(-fast_margins(u),0).sum(1)

def initial_points(seed):
 u=qmc.Sobol(8,scramble=True,seed=seed).random_base2(11)
 u=u[(fast_margins(u)>=0).all(1)]
 assert len(u)>=128
 return u[:128]

def optimize(method,fwd,goal,seed):
 oracle=Oracle(fwd,goal);info={}
 initial=initial_points(seed)
 if method=='differential_evolution':
  def objective(a):return oracle.value(np.asarray(a).T)
  result=differential_evolution(objective,[(0.,1.)]*8,init=initial,strategy='best1bin',mutation=(.5,1),recombination=.7,maxiter=31,polish=False,vectorized=True,updating='deferred',seed=seed,tol=0,atol=-1)
  info=dict(iterations=int(result.nit),optimizer_success=bool(result.success),message=str(result.message))
 else:
  records=[]
  for start,x0 in enumerate(initial[:8]):
   if BUDGET-oracle.count<9:break
   oracle.limit=min(BUDGET,oracle.count+512);cache={}
   def batch(x):
    key=x.tobytes()
    if key not in cache:
     step=np.where(x+1e-4<=1,1e-4,-1e-4);points=np.vstack([x,x+np.diag(step)])
     y,p=oracle.evaluate(points);e=np.sqrt(((y-goal)/TOL)**2+1e-12);v=logsumexp(20*e,axis=1)/20
     g=np.column_stack([p-P_MIN,fast_margins(points)])
     cache.clear();cache[key]=(v[0],(v[1:]-v[0])/step,g[0],((g[1:]-g[0])/step[:,None]).T)
    return cache[key]
   try:
    r=minimize(lambda x:batch(x)[0],x0,jac=lambda x:batch(x)[1],method='SLSQP',bounds=[(0.,1.)]*8,constraints=[dict(type='ineq',fun=lambda x:batch(x)[2],jac=lambda x:batch(x)[3])],options=dict(maxiter=160,ftol=1e-7))
    records.append(dict(start=start,status=int(r.status),message=str(r.message)))
   except BudgetReached:records.append(dict(start=start,status='budget',message='Per-start evaluation budget exhausted'))
  info=dict(starts=len(records),start_results=records)
 assert oracle.count<=BUDGET
 return (*oracle.arrays(),info)


def evaluate_method_seed(method,seed):
 warnings.filterwarnings('ignore');torch.set_num_threads(1)
 with threadpool_limits(limits=1):
  goals=pd.read_csv(OUT/'data/test_goals.csv');folder=OUT/'runs'/method/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
  fwd=Forward();geometry=Geometry();gen=None
  if method in ['jw_cvae','sm_cvae','conditional_gmm']:
   ext='.joblib' if method=='conditional_gmm' else '.pt';gen=Generator(OUT/'models'/method/f'seed{seed}{ext}')
  for i,row in goals.iterrows():
   goal_id=str(row.goal_id);path=folder/f'{goal_id}.json'
   if path.exists():continue
   started=time.perf_counter();goal=row[TARGETS].to_numpy(float);before=geometry.checked
   if gen is not None:
    u=gen.sample(goal,BUDGET,seed*100000+i);y,p=fwd.predict(u);info={};forward_calls=len(u)
   else:
    u,y,p,info=optimize(method,fwd,goal,seed);forward_calls=len(u)
   search_seconds=time.perf_counter()-started
   score=score_pool(u,y,p,goal,geometry)
   bounds=bound_mask(u);eligible=np.flatnonzero(bounds&(p>=P_MIN));fallback=False
   if not len(eligible):eligible=np.flatnonzero(bounds);fallback=True
   nearest=None
   for j in eligible[np.argsort(score['distance'][eligible],kind='stable')]:
    ok=geometry.one(u[j]);score['geometry'][j]=int(ok)
    if ok:
     nearest=dict(dimensions=dict(zip(FEATURES,to_phys(u[j]).tolist())),predicted=dict(zip(TARGETS,y[j].tolist())),p_bi_mono=float(p[j]),normalized_max_error=float(score['distance'][j]),passes_stability=bool(p[j]>=P_MIN),passes_all=bool(p[j]>=P_MIN and score['distance'][j]<=1));break
   trace=folder/f'{goal_id}.npz';tmp=trace.with_suffix(f'.{os.getpid()}.tmp.npz')
   np.savez_compressed(tmp,u=u,predicted=y,p_bi_mono=p,passed=score['passed'],geometry_checked=score['geometry'],normalized_error=score['errors']);tmp.replace(trace)
   result=dict(method=method,seed=seed,goal_id=goal_id,goal=dict(zip(TARGETS,goal.tolist())),goal_barrier_mJ=float(10**goal[2]),forward_calls=forward_calls,candidate_points=len(u),success=bool(score['passed'].any()),pass_count=int(score['passed'].sum()),pass_fraction=float(score['passed'].mean()),fraction_scope='generated candidates' if gen is not None else 'adaptive evaluated points; not a generative success fraction',bounds_pass_count=int(bounds.sum()),stability_passing_count=int((p>=P_MIN).sum()),property_pass_fractions=(score['errors']<=1).mean(0).tolist(),surrogate_joint_count=int(score['surrogate'].sum()),official_geometry_checks=geometry.checked-before,unique_designs_1e4=int(len(np.unique(np.round(u,4),axis=0))),search_seconds=search_seconds,filter_seconds=time.perf_counter()-started-search_seconds,total_seconds=time.perf_counter()-started,selected=score['selected'],nearest=nearest,optimizer=info,budget_prefix={str(b):bool(score['passed'][:b].any()) for b in [512,1024,2048,4096]})
   dump(path,result);print('EVAL',method,seed,i+1,'/49','success',result['success'],'passing',result['pass_count'],'forward',forward_calls,'seconds',round(result['total_seconds'],2),flush=True)
  dump(folder/'complete.json',dict(method=method,seed=seed,goals=len(goals),done=True));return f'{method} seed{seed}'
