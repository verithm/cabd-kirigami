"""Parallelize independent regression CV tasks while the classifier finishes.
Same configurations, fits, data, seed and metrics as run(); only scheduling differs.
"""
import os,warnings
from concurrent.futures import ProcessPoolExecutor,as_completed
from threadpoolctl import threadpool_limits
from seungmin_models import *

def precompute(target):
 warnings.filterwarnings('ignore');name='sm_referee';d=load_data();tr=d[d.role=='train'];out=OUT/'models'/name
 with threadpool_limits(limits=1):
  for k in range(3):
   cache=out/f'cv_{target}_{k}.json'
   if cache.exists():continue
   actual=[];preds=[];folds=[]
   for fold in range(5):
    training=tr[tr.fold!=fold];q=tr[(tr.fold==fold)&tr.is_reg]
    model=fit(name,training,target,k,42);p=predict(model,q);y=q[target].to_numpy(float)
    folds.append(metrics_reg(y,p));actual.extend(y.tolist());preds.extend(p.tolist())
    print('PRECOMPUTE',target,k,fold,flush=True)
   record=dict(model=name,target=target,config=k,folds=folds,**metrics_reg(actual,preds))
   temp=cache.with_suffix(f'.precompute-{os.getpid()}.tmp')
   temp.write_text(json.dumps(record,indent=2));temp.replace(cache)
 return target
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(precompute,t) for t in TARGETS]):print('DONE',f.result(),flush=True)
