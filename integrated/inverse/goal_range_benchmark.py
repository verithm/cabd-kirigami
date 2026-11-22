"""Five target-generation strategies, fixed generators/Forward and physical-unit search.
Only newly predicted candidates are evaluated; no ANSYS or generator retraining.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
import warnings
from scipy.spatial import cKDTree
from scipy.special import logsumexp,softmax
from scipy.stats import qmc
from scipy.optimize import differential_evolution,minimize
from threadpoolctl import threadpool_limits
from shared import *
from generators import Generator
from evaluate import initial_points,BudgetReached
DEST=RUNS/'inverse-target-strategies'
TARGET_SEED=20261122
SEED=0
PHYSICAL_TOL=np.array([2.,.25,.03])
STRATEGIES=['1_observed','2_interpolation','4_angle_conditional','5_sparse_boundary','6_independent']
STRATEGY_LABELS={'1_observed':'1. 실제 Test 조합','2_interpolation':'2. 가까운 조합 사이 보간','4_angle_conditional':'4. 각도별 조건부 샘플링','5_sparse_boundary':'5. 희소 구간·경계','6_independent':'6. 독립적인 넓은 조합'}

def now():return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
def physical(y):
 y=np.asarray(y);return np.column_stack([y[:,:2],10**y[:,2]])
def error_matrix(y,g):return np.abs((physical(y)-physical(np.atleast_2d(g)))/PHYSICAL_TOL)

def target_sets():
 from sklearn.neighbors import KernelDensity
 DEST.mkdir(parents=True,exist_ok=True);(DEST/'goals').mkdir(parents=True,exist_ok=True)
 training=pd.read_csv(OUT/'data/training.csv');tr=training[training.label=='bi>mono'].copy();assert len(tr)==394
 a=tr[TARGETS].to_numpy();phys=physical(a);mu=a.mean(0);sd=a.std(0);z=(a-mu)/sd;tree=cKDTree(z);ptree=cKDTree(phys/PHYSICAL_TOL)
 lo=a.min(0);hi=a.max(0);plo=phys.min(0);phi=phys.max(0);rng=np.random.default_rng(TARGET_SEED)
 H=(.98*np.cov(z,rowvar=False)+.02*np.eye(3))*len(z)**(-2/7)
 candidates={};provenance={}
 test=pd.read_csv(OUT/'data/test_goals.csv');candidates['1_observed']=test[TARGETS].to_numpy();provenance['1_observed']=[dict(source_geom_id=str(g),subtype='actual held-out response') for g in test.goal_id]
 # Nearby pairs selected in standardized joint angle/displacement/log-energy space.
 rows=[];meta=[];neighbors=tree.query(z,k=9)[1]
 while len(rows)<100:
  i=int(rng.integers(len(a)));j=int(rng.choice(neighbors[i,1:]));weight=float(rng.uniform(.25,.75));p=(1-weight)*phys[i]+weight*phys[j];row=np.r_[p[:2],np.log10(p[2])]
  if tree.query((row-mu)/sd)[0]<1e-8:continue
  rows.append(row);meta.append(dict(source_a=str(tr.iloc[i].geom_id),source_b=str(tr.iloc[j].geom_id),weight=weight,subtype='physical-unit convex interpolation'))
 candidates['2_interpolation']=np.array(rows);provenance['2_interpolation']=meta
 # Conditional Gaussian KDE: jointly sample displacement/log-barrier at an angle.
 cross=H[1:,0]/H[0,0];conditional_cov=H[1:,1:]-np.outer(H[1:,0],H[0,1:])/H[0,0]
 rows=[];meta=[]
 for i in range(100):
  angle=float(lo[0]+(hi[0]-lo[0])*(i+rng.random())/100);az=(angle-mu[0])/sd[0]
  weights=softmax(-.5*(az-z[:,0])**2/H[0,0])
  for _ in range(10000):
   j=int(rng.choice(len(z),p=weights));rest=rng.multivariate_normal(z[j,1:]+cross*(az-z[j,0]),conditional_cov);row=mu+sd*np.r_[az,rest]
   if (row>=lo).all() and (row<=hi).all():break
  else:raise RuntimeError('Conditional rejection failed')
  rows.append(row);meta.append(dict(source_center=str(tr.iloc[j].geom_id),subtype='joint residual conditional KDE',angle_stratum=i))
 candidates['4_angle_conditional']=np.array(rows);provenance['4_angle_conditional']=meta
 # Boundary/sparse probes from the JOINT KDE, not the independent marginal box.
 centers=rng.integers(len(z),size=16000);pool=z[centers]+rng.multivariate_normal(np.zeros(3),H,size=len(centers));poolraw=mu+sd*pool
 mask=((poolraw>=lo)&(poolraw<=hi)).all(1);pool=pool[mask];poolraw=poolraw[mask];assert len(pool)>1000
 nn=tree.query(pool,k=5)[0].mean(1);margin=np.minimum((poolraw-lo)/(hi-lo),(hi-poolraw)/(hi-lo)).min(1)
 chosen=[];metas=[]
 for label,rank in [('sparse',np.argsort(-nn)),('boundary',np.argsort(margin))]:
  n=0
  for j in rank:
   if chosen and np.linalg.norm(pool[j]-pool[chosen],axis=1).min()<.12:continue
   chosen.append(int(j));metas.append(dict(subtype=label,k5_standardized_distance=float(nn[j]),box_edge_fraction=float(margin[j])));n+=1
   if n==50:break
  assert n==50
 candidates['5_sparse_boundary']=poolraw[chosen];provenance['5_sparse_boundary']=metas
 # Latin hypercube in ORIGINAL physical energy units; angles/displacements independent.
 lhs=qmc.LatinHypercube(d=3,seed=TARGET_SEED).random(100);wide=plo+lhs*(phi-plo)
 candidates['6_independent']=np.column_stack([wide[:,:2],np.log10(wide[:,2])]);provenance['6_independent']=[dict(subtype='independent physical-unit Latin hypercube') for _ in wide]
 hashes={}
 for strategy in STRATEGIES:
  y=candidates[strategy];p=physical(y);_,near=tree.query((y-mu)/sd);pdistance,pi=ptree.query(p/PHYSICAL_TOL,k=min(30,len(phys)))
  # Exact nearest Chebyshev tolerance distance across all training bi->mono responses.
  tol_distance=np.abs((p[:,None,:]-phys[None,:,:])/PHYSICAL_TOL).max(2).min(1)
  df=pd.DataFrame(y,columns=TARGETS);df['goal_barrier_mJ']=p[:,2];df.insert(0,'goal_id',[f'{strategy}_G{i:03d}' for i in range(len(df))]);df['nearest_train_geom_id']=tr.iloc[near].geom_id.to_numpy();df['nearest_train_tolerance_distance']=tol_distance;df['within_train_response_tolerances']=tol_distance<=1
  metadata=pd.DataFrame(provenance[strategy]);df=pd.concat([df,metadata],axis=1)
  if strategy!='1_observed':assert (tree.query((y-mu)/sd)[0]>1e-8).all()
  assert df.goal_id.nunique()==len(df) and len(np.unique(np.round(y,10),axis=0))==len(df)
  path=DEST/'goals'/f'{strategy}.csv'
  content=df.to_csv(index=False)
  if path.exists():assert path.read_text()==content,'Changed target generation; use a fresh destination'
  else:path.write_text(content)
  hashes[strategy]=hashlib.sha256(path.read_bytes()).hexdigest()
 tr[['geom_id',*TARGETS]].to_csv(DEST/'target_distribution_training.csv',index=False)
 config=dict(strategies=STRATEGY_LABELS,goal_counts={s:len(candidates[s]) for s in STRATEGIES},target_seed=TARGET_SEED,inverse_seed=SEED,forward_seed=0,budget_per_goal=BUDGET,tolerances=dict(angle_deg=2.,deformation_mm=.25,barrier_mJ=.03),p_min=P_MIN,official_geometry_required=True,stress_filter=False,models_retrained=False,optimizers_rerun=True,ansys_executed=False,goal_distribution_basis='394 Train bi->mono responses only; actual held-out 49 used solely for baseline strategy1',new_target_exact_matches_forbidden=True,goals_sha256=hashes,forward_sha256=PROTOCOL['forward_sha256'],notes=['Target novelty is not a physical feasibility guarantee','Strategy6 barrier sampled uniformly in physical mJ, not log-uniform','Optimization visited-point pass rates are not reported; representative is best official-valid joint-passing point','Generative pass counts include all 4096 raw candidates, including invalid proposals'])
 path=DEST/'protocol.json'
 if path.exists():assert json.loads(path.read_text())==config
 dump(path,config)
 return config

class Oracle:
 def __init__(self,fwd,goal):self.fwd=fwd;self.goal=goal;self.count=0;self.limit=BUDGET;self.us=[];self.ys=[];self.ps=[]
 def evaluate(self,u):
  u=np.atleast_2d(u)
  if self.count+len(u)>min(BUDGET,self.limit):raise BudgetReached()
  y,p=self.fwd.predict(u);assert np.isfinite(y).all() and np.isfinite(p).all()
  self.us.append(u.copy());self.ys.append(y);self.ps.append(p);self.count+=len(u);return y,p
 def arrays(self):return np.concatenate(self.us),np.concatenate(self.ys),np.concatenate(self.ps)
 def value(self,u):
  y,p=self.evaluate(u);e=error_matrix(y,self.goal);return logsumexp(20*e,axis=1)/20+20*np.maximum(P_MIN-p,0)+20*np.maximum(-fast_margins(u),0).sum(1)

def optimize(method,fwd,goal):
 oracle=Oracle(fwd,goal);initial=initial_points(SEED)
 if method=='differential_evolution':
  result=differential_evolution(lambda a:oracle.value(np.asarray(a).T),[(0.,1.)]*8,init=initial,strategy='best1bin',mutation=(.5,1),recombination=.7,maxiter=31,polish=False,vectorized=True,updating='deferred',seed=SEED,tol=0,atol=-1)
  info=dict(iterations=int(result.nit),message=str(result.message))
 else:
  records=[]
  for start,x0 in enumerate(initial[:8]):
   if BUDGET-oracle.count<9:break
   oracle.limit=min(BUDGET,oracle.count+512);cache={}
   def batch(x):
    key=x.tobytes()
    if key not in cache:
     step=np.where(x+1e-4<=1,1e-4,-1e-4);points=np.vstack([x,x+np.diag(step)]);y,p=oracle.evaluate(points)
     e=np.sqrt(error_matrix(y,goal)**2+1e-12);v=logsumexp(20*e,axis=1)/20;g=np.column_stack([p-P_MIN,fast_margins(points)])
     cache.clear();cache[key]=(v[0],(v[1:]-v[0])/step,g[0],((g[1:]-g[0])/step[:,None]).T)
    return cache[key]
   try:
    r=minimize(lambda x:batch(x)[0],x0,jac=lambda x:batch(x)[1],method='SLSQP',bounds=[(0.,1.)]*8,constraints=[dict(type='ineq',fun=lambda x:batch(x)[2],jac=lambda x:batch(x)[3])],options=dict(maxiter=160,ftol=1e-7));records.append(dict(start=start,status=int(r.status),message=str(r.message)))
   except BudgetReached:records.append(dict(start=start,status='budget'))
  info=dict(starts=len(records),start_results=records)
 return (*oracle.arrays(),info)

def score(u,y,p,goal,geometry,generative):
 errors=error_matrix(y,goal);distance=errors.max(1);bounds=bound_mask(u);promising=bounds&(p>=np.float32(P_MIN))&(distance<=1);geo=np.full(len(u),-1,dtype=np.int8);geo[~bounds]=0
 idx=np.flatnonzero(promising);selected_index=None
 if generative:
  if len(idx):geo[idx]=geometry.many(u[idx]).astype(np.int8)
  valid=idx[geo[idx]==1]
  if len(valid):selected_index=int(valid[np.argmin(distance[valid])])
 else:
  for j in idx[np.argsort(distance[idx],kind='stable')]:
   geo[j]=int(geometry.one(u[j]))
   if geo[j]==1:selected_index=int(j);break
 known=np.zeros(len(u),dtype=np.int8);known[promising]=-1;known[promising&(geo>=0)]=(geo[promising&(geo>=0)]==1).astype(np.int8)
 selected=None
 if selected_index is not None:
  j=selected_index;selected=dict(index=j,dimensions=dict(zip(FEATURES,to_phys(u[j]).tolist())),predicted=dict(zip(TARGETS,y[j].tolist())),predicted_barrier_mJ=float(10**y[j,2]),p_bi_mono=float(p[j]),normalized_max_error=float(distance[j]))
 return dict(errors=errors,geometry=geo,known_pass=known,promising=promising,selected=selected)

def run_chunk(strategy,method,start,stop):
 warnings.filterwarnings('ignore');torch.set_num_threads(1)
 with threadpool_limits(limits=1):
  goals=pd.read_csv(DEST/'goals'/f'{strategy}.csv');folder=DEST/'runs'/strategy/method;folder.mkdir(parents=True,exist_ok=True)
  if all((folder/f'{goals.iloc[i].goal_id}.json').exists() for i in range(start,stop)):return strategy,method,start,stop
  fwd=Forward();geometry=Geometry();gen=None
  if method in METHODS[:3]:gen=Generator(OUT/'models'/method/('seed0.joblib' if method=='conditional_gmm' else 'seed0.pt'))
  for i in range(start,stop):
   row=goals.iloc[i];path=folder/f'{row.goal_id}.json'
   if path.exists():continue
   begun=time.perf_counter();goal=row[TARGETS].to_numpy(float)
   if gen is not None:
    # Matching target index uses matching random stream across strategies/models.
    u=gen.sample(goal,BUDGET,i);y,p=fwd.predict(u);info={}
   else:u,y,p,info=optimize(method,fwd,goal)
   search_seconds=time.perf_counter()-begun;before=geometry.checked;s=score(u,y,p,goal,geometry,gen is not None)
   tmp=path.with_suffix(f'.{os.getpid()}.tmp.npz');np.savez_compressed(tmp,u=u,predicted=y,p_bi_mono=p,geometry_checked=s['geometry'],known_pass=s['known_pass'],normalized_error=s['errors']);tmp.replace(path.with_suffix('.npz'))
   record=dict(strategy=strategy,method=method,goal_id=str(row.goal_id),goal=dict(zip(TARGETS,goal.tolist())),goal_barrier_mJ=float(10**goal[2]),seed=SEED,forward_evaluations=len(u),success=s['selected'] is not None,selected=s['selected'],generated_pass_count=int((s['known_pass']==1).sum()) if gen is not None else None,generated_pass_fraction=float((s['known_pass']==1).mean()) if gen is not None else None,surrogate_joint_count=int(s['promising'].sum()),official_geometry_checks=geometry.checked-before,geometry_scope='all promising candidates' if gen is not None else 'error-ordered promising candidates through first valid representative',search_seconds=search_seconds,filter_seconds=time.perf_counter()-begun-search_seconds,total_seconds=time.perf_counter()-begun,optimizer=info)
   dump(path,record)
  print('CHUNK COMPLETE',strategy,method,start,stop,flush=True)
 return strategy,method,start,stop

def main():
 config=target_sets()
 for name,h in PROTOCOL['forward_sha256'].items():assert hashlib.sha256((FORWARD/name).read_bytes()).hexdigest()==h
 dump(DEST/'status.json',dict(phase='evaluating',updated=now(),pid=os.getpid(),planned_evaluations=sum(config['goal_counts'].values())*len(METHODS),ansys=False))
 tasks=[(strategy,method,start,min(start+10,n)) for start in range(0,100,10) for strategy,n in config['goal_counts'].items() if start<n for method in METHODS]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(run_chunk,*args) for args in tasks]):f.result()
 import importlib.util
 spec=importlib.util.spec_from_file_location('target_strategy_report',Path(__file__).with_name('goal_range_report.py'));r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);r.main()
 dump(DEST/'status.json',dict(phase='complete',updated=now(),ansys=False))
 print('TARGET STRATEGY BENCHMARK COMPLETE; NO ANSYS',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  dump(DEST/'status.json',dict(phase='failed',updated=now(),error=repr(e),ansys=False));raise
