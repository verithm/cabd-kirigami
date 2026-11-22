"""Resume-safe inverse training and surrogate evaluation through step 7 only."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
from shared import *
from generators import train_method
from evaluate import evaluate_method_seed

def status(phase,**extra):dump(OUT/'status.json',dict(phase=phase,updated=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),pid=os.getpid(),ansys=False,**extra))

def main():
 started=time.perf_counter();prepare();status('training')
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(train_method,m) for m in METHODS[:3]]):print('TRAINING COMPLETE',f.result(),flush=True)
 status('evaluating',planned_method_seed_runs=len(METHODS)*len(SEEDS))
 with ProcessPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(evaluate_method_seed,m,s) for m in METHODS for s in SEEDS]
  for f in as_completed(futures):print('EVALUATION COMPLETE',f.result(),flush=True)
 status('reporting')
 import importlib.util
 spec=importlib.util.spec_from_file_location('unified_inverse_report',Path(__file__).with_name('report.py'))
 report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
 report.make_report()
 status('complete',seconds_this_invocation=time.perf_counter()-started)
 print('ALL INVERSE TRAINING AND STEP-7 EVALUATION COMPLETE; NO ANSYS RUN',flush=True)

if __name__=='__main__':
 try:main()
 except BaseException as e:status('failed',error=repr(e));raise
