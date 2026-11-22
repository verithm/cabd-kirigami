"""Predict with a saved final model: python predict.py MODEL INPUT.csv OUTPUT.csv --seed 0."""
import argparse
import joblib,numpy as np,pandas as pd
from common import *

def predict_frame(name,d,seed=0):
 folder=OUT/'models'/name
 if 'geom_id' not in d:d=d.copy();d['geom_id']=[f'q{i}' for i in range(len(d))]
 if name.startswith('jw_'):
  x=d[FEATURES].to_numpy(float);partial=np.column_stack([x,np.zeros(len(x))]);full=partial.copy();full[:,-1]=1
  clf=joblib.load(folder/f'stability_s{seed}.joblib');switch=(clf.predict(partial)==1)&(clf.predict(full)==0)
  y=np.column_stack([joblib.load(folder/f'{t}_s{seed}.joblib').predict(partial) for t in TARGETS])
 else:
  from seungmin_models import predict
  switch=predict(joblib.load(folder/f'stability_s{seed}.joblib'),d)
  if name=='sm_mlp':y=predict(joblib.load(folder/f'joint_s{seed}.joblib'),d)
  else:y=np.column_stack([predict(joblib.load(folder/f'{t}_s{seed}.joblib'),d) for t in TARGETS])
 out=d[['geom_id']].copy();out['predicted_switch']=switch
 for j,t in enumerate(TARGETS):out['predicted_'+t]=y[:,j]
 out['predicted_barrier_mJ']=10**y[:,2]
 return out

if __name__=='__main__':
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('model');a.add_argument('input',type=Path);a.add_argument('output',type=Path);a.add_argument('--seed',type=int,default=0);args=a.parse_args()
 d=pd.read_csv(args.input)
 if not all(c in d for c in FEATURES):a.error('Eight design variables are required')
 result=predict_frame(args.model,d,args.seed);result.to_csv(args.output,index=False)
 print(args.output,len(result))
