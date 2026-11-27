"""Continue only selected candidates/tasks with frozen data/code/search space."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parents[1]
for folder in [BASE,BASE/'forward',BASE.parent/'src']:
    sys.path.insert(0,str(folder))
from common import dump
from forward_six.models import NAMES
from forward_six.single_models import GROUPS
from forward_six import single_tune as tuning


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=NAMES,required=True)
    p.add_argument('--tasks',nargs='+',choices=list(GROUPS),required=True)
    p.add_argument('--total-trials',type=int,required=True)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if args.total_trials<1 or args.workers<1:
        p.error('Trials and workers must be positive')
    tuning.prepare()
    started=tuning.now()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures=[ex.submit(tuning.study_worker,n,g,args.total_trials) for n in dict.fromkeys(args.models) for g in dict.fromkeys(args.tasks)]
        for f in as_completed(futures):
            print('SELECTED_CONTINUATION_DONE',f.result(),flush=True)
            tuning.summarize()
    dump(tuning.DEST/'selected-extension-complete.json',{'started_at':started,'finished_at':tuning.now(),
        'models':args.models,'tasks':args.tasks,'requested_total_per_selected_study':args.total_trials,
        'original_20_trial_comparison_preserved':True,'test_scored':False,
        'note':'Extended studies have larger budgets; do not compare their best.json as equal-budget 20-trial results.'})


if __name__=='__main__':
    main()
