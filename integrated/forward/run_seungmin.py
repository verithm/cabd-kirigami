from concurrent.futures import ProcessPoolExecutor,as_completed
from seungmin_models import run,NAMES
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=3) as pool:
  futures={pool.submit(run,n):n for n in NAMES}
  for f in as_completed(futures):
   try:print('DONE',f.result(),flush=True)
   except Exception as e:print('FAILED',futures[f],repr(e),flush=True);raise
