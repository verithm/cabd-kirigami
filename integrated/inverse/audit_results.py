"""Independent integrity checks on saved study data; no new ANSYS calculations."""
from shared import *
from generators import Generator,torch_read

def audit():
 protocol=json.loads((OUT/'protocol.json').read_text())
 assert protocol['ansys_executed'] is False
 for name,h in protocol['forward_sha256'].items():assert hashlib.sha256((FORWARD/name).read_bytes()).hexdigest()==h
 d=pd.read_csv(OUT/'data/training.csv');expected=set(d.geom_id);test=set(pd.read_csv(OUT/'data/test_goals.csv').goal_id);assert not expected&test
 checkpoints=[]
 for method in METHODS[:3]:
  for seed in SEEDS:
   ext='.joblib' if method=='conditional_gmm' else '.pt';gen=Generator(OUT/'models'/method/f'seed{seed}{ext}')
   assert set(gen.saved['training_ids'])==expected
   np.testing.assert_allclose(gen.mu,d[TARGETS].mean().to_numpy(),rtol=1e-10,atol=1e-10)
   np.testing.assert_allclose(gen.sd,d[TARGETS].std(ddof=0).to_numpy(),rtol=1e-10,atol=1e-10)
   assert gen.saved['done']
   checkpoints.append(dict(method=method,seed=seed,train_shapes=len(expected),gmm_converged=None if method!='conditional_gmm' else {str(k):bool(v.converged_) for k,v in gen.saved['models'].items()}))
 total=0;representatives=[]
 for method in METHODS:
  for seed in SEEDS:
   folder=OUT/'runs'/method/f'seed{seed}';files=sorted(folder.glob('TE*.json'));assert len(files)==49
   for path in files:
    r=json.loads(path.read_text());assert r['method']==method and r['seed']==seed
    with np.load(path.with_suffix('.npz')) as z:
     u,y,p,passed,geo,errors=(z[k] for k in ['u','predicted','p_bi_mono','passed','geometry_checked','normalized_error'])
     assert len(u)==r['forward_calls']<=BUDGET
     assert np.isfinite(u).all() and np.isfinite(y).all() and np.isfinite(p).all()
     e=np.abs((y-np.array([r['goal'][t] for t in TARGETS]))/TOL)
     np.testing.assert_allclose(e,errors,rtol=1e-10,atol=1e-10)
     surrogate=bound_mask(u)&(p>=np.float32(P_MIN))&(e.max(1)<=1)
     assert (geo[surrogate]>=0).all()
     np.testing.assert_array_equal(passed,surrogate&(geo==1))
     assert int(passed.sum())==r['pass_count'] and bool(passed.any())==r['success']
     if r['selected']:
      s=r['selected'];x=np.array([s['dimensions'][k] for k in FEATURES]);pp=dict(zip(FEATURES,x));pp['t_w']=G.T_W
      assert all(G.check_constraints(pp).values()),(method,seed,r['goal_id'])
      assert np.float32(s['p_bi_mono'])>=np.float32(P_MIN)
      assert s['normalized_max_error']<=1
      np.testing.assert_allclose(s['normalized_max_error'],e[passed].max(1).min(),atol=1e-10)
      representatives.append((normalized(x),np.array([s['predicted'][t] for t in TARGETS]),s['p_bi_mono']))
    total+=1
 fwd=Forward()
 if representatives:
  u=np.array([r[0] for r in representatives]);y,p=fwd.predict(u)
  np.testing.assert_allclose(y,np.array([r[1] for r in representatives]),rtol=1e-6,atol=5e-5)
  np.testing.assert_allclose(p,np.array([r[2] for r in representatives]),rtol=1e-6,atol=1e-6)
 assert total==735
 record=dict(goal_evaluations=total,generator_checkpoints=len(checkpoints),training_test_disjoint=True,forward_weights_unchanged=True,trace_counts_and_thresholds_verified=True,all_passing_representatives_official_geometry_rechecked=len(representatives),representative_forward_reload_matches_with_numeric_tolerance=True,probability_comparison='native classifier float32',checkpoints=checkpoints,ansys_executed=False)
 dump(OUT/'integrity_audit.json',record);print('INVERSE INTEGRITY AUDIT PASSED',total,'evaluations;',len(representatives),'representatives',flush=True)
if __name__=='__main__':audit()
