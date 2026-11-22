"""Check saved model reloads against stored Test predictions on a common subset."""
from common import *
from predict import predict_frame

def audit():
 d=load_data();q=d[(d.role=='test')&d.is_reg&d.label.notna()].head(8)
 checks=[]
 for path in sorted((OUT/'models').glob('*/complete.json')):
  model=path.parent.name;saved=pd.read_csv(path.parent/'test_predictions.csv')
  assert set(saved.seed)==set(SEEDS)
  pred=predict_frame(model,q,0)
  for t in ['stability',*TARGETS]:
   expected=saved[(saved.seed==0)&(saved.target==t)].set_index('geom_id').loc[q.geom_id,'predicted'].to_numpy()
   actual=pred['predicted_switch' if t=='stability' else 'predicted_'+t].to_numpy()
   np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-5)
  checks.append(dict(model=model,reload_matches_saved_predictions=True,geometries=len(q),seed_count=len(SEEDS)))
 dump(OUT/'saved_model_audit.json',checks);print('Reload audit passed:',len(checks),'models')
if __name__=='__main__':audit()
