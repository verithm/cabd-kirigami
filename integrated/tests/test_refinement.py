"""The requested 50-trial cap overrides any desire to continue improving."""
from concurrent.futures import Future
import json
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
BASE=Path(__file__).resolve().parents[1]
for folder in [BASE,BASE/'forward',BASE.parent/'src']:
    sys.path.insert(0,str(folder))
from forward_six import refine


class RequestedCapTests(unittest.TestCase):
    def test_stops_at_50_even_if_no_study_has_plateaued(self):
        targets=[]
        class Executor:
            def __init__(self,**kwargs):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def submit(self,fn,*args):
                targets.append(args[2]);f=Future();f.set_result(args);return f
        def state(n,previous=None):
            return {'budget':n,'objectives':{},'streaks':{},'plateau_studies':0,'all_plateau':False}
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);(dest/'stage_20.json').write_text(json.dumps(state(20)))
            with patch.object(refine,'DEST',dest),patch.object(refine,'verify_protocol'),\
                 patch.object(refine,'counts',return_value={'all':40}),patch.object(refine,'snapshot',side_effect=state),\
                 patch.object(refine,'ProcessPoolExecutor',Executor),patch.object(sys,'argv',['refine','--max-trials','50']):
                refine.main()
            self.assertEqual(len(targets),24)
            self.assertEqual(set(targets),{50})
            final=json.loads((dest/'complete.json').read_text())
            self.assertEqual(final['budget_per_study'],50)
            self.assertEqual(final['reason'],'explicit_user_requested_50_cap')
            self.assertFalse(final['additional_trials_after_50'])
            self.assertTrue(final['user_requested_stop'])
