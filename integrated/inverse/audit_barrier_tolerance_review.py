"""Validate physical-unit sensitivity records and representative candidates."""
from shared import *
DEST=RUNS/'barrier-tolerance-review'

def main():
 reps=pd.read_csv(DEST/'representatives.csv');goals=pd.read_csv(OUT/'data/test_goals.csv').set_index('goal_id');protocol=json.loads((DEST/'protocol.json').read_text())
 assert protocol['angle_tolerance_deg']==2 and protocol['displacement_tolerance_mm']==.25
 checked=set();representative_y=reps[[f'predicted_{k}' for k in TARGETS]].to_numpy()
 x=reps[FEATURES].to_numpy();u=normalized(x);assert bound_mask(u).all()
 for dims in x:
  key=tuple(dims)
  if key not in checked:assert all(G.check_constraints(dict(zip(FEATURES,dims),t_w=G.T_W)).values());checked.add(key)
 target=goals.loc[reps.goal_id,TARGETS].to_numpy();err=np.abs(representative_y-target)
 assert (err[:,0]<=2+1e-10).all() and (err[:,1]<=.25+1e-10).all()
 p_saved=reps.p_bi_mono.to_numpy(dtype=np.float32);assert (p_saved>=np.float32(P_MIN)).all()
 for key,q in reps.groupby('criterion'):
  if key=='log10_0.17':assert (np.abs(q.predicted_log10_barrier-goals.loc[q.goal_id,'log10_barrier'].to_numpy())<=.17+1e-10).all()
  else:
   t=float(key[4:8]);assert (np.abs(q.predicted_barrier_mJ-q.goal_barrier_mJ)<=t+1e-10).all()
 fwd=Forward();y,p=fwd.predict(u);np.testing.assert_allclose(y,representative_y,rtol=1e-6,atol=5e-5);np.testing.assert_allclose(p,p_saved,rtol=1e-6,atol=1e-6)
 n=0
 for method in METHODS:
  for seed in SEEDS:
   paths=sorted((DEST/'candidate_runs'/method/f'seed{seed}').glob('TE*.json'));assert len(paths)==49
   assert {p.stem for p in paths}==set(goals.index)
   for path in paths:
    r=json.loads(path.read_text());assert r['method']==method and r['seed']==seed
    assert hashlib.sha256((resolve_reference(r['source_trace'])).read_bytes()).hexdigest()==r['source_sha256']
    s=r['scores'];counts=[s[f'abs_{t:.2f}mJ']['pass_count'] for t in [.02,.03,.05]];assert counts==sorted(counts)
    for q in s.values():assert 0<=q['pass_count']<=r['candidate_points'] and q['success']==(q['pass_count']>0) and (q['selected'] is not None)==q['success']
    n+=1
 assert n==735
 for name,h in PROTOCOL['forward_sha256'].items():assert hashlib.sha256((FORWARD/name).read_bytes()).hexdigest()==h
 a=json.loads((DEST/'audit.json').read_text());a.update(source_trace_hashes_verified=n,representatives_criteria_and_forward_verified=len(reps),unique_representative_geometries_original_checker_verified=len(checked),absolute_threshold_pass_counts_monotonic=True);dump(DEST/'audit.json',a)
 print('PHYSICAL BARRIER TOLERANCE AUDIT PASSED',n,'candidate records;',len(reps),'representatives;',len(checked),'unique dimensions',flush=True)
if __name__=='__main__':main()
