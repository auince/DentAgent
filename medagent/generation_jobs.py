"""Resume successful rows and isolate invalid rows from the rest of a stage."""
import concurrent.futures
import contextvars
import time
from collections import deque
from .common import write_json,read_json
from .generation_teacher import TeacherTransientError

FEEDBACK=contextvars.ContextVar('generation_feedback',default=None)

class GenerationDeferred(RuntimeError):
    pass

def run_rows(view,count,produce,folder,status,workers,max_attempts=6):
    folder.mkdir(parents=True,exist_ok=True)
    pending=deque(i for i in range(count) if not (folder/f'{i:06}.json').exists())
    accepted=count-len(pending);attempts={};errors={};blocked={}
    def execute(i):
        token=FEEDBACK.set(errors.get(i))
        try:return produce(i)
        finally:FEEDBACK.reset(token)
    def progress(active):
        write_json(status,{'phase':'views','view':view,'completed':accepted,'target':count,
            'in_flight':len(active),'remaining_queued':len(pending),'blocked_count':len(blocked),
            'concurrency':workers,'time':time.time()})
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        active={};progress(active)
        while pending or active:
            while pending and len(active)<workers:
                i=pending.popleft();active[pool.submit(execute,i)]=i
            # Publish the submitted work before waiting. Otherwise a slow final
            # episode can leave a stale zero-in-flight snapshot for minutes.
            progress(active)
            done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                i=active.pop(future)
                try:
                    future.result()
                    if not (folder/f'{i:06}.json').exists():raise ValueError('accepted_row_artifact_missing')
                    accepted+=1
                except TeacherTransientError as exc:
                    # Infrastructure failures are handled by the queue supervisor.
                    raise
                except Exception as exc:
                    attempts[i]=attempts.get(i,0)+1;errors[i]=str(exc)[:400]
                    write_json(folder.parent.parent/'view_rejections'/view/f'{i:06}.json',
                        {'id':i,'attempts_this_run':attempts[i],'error':errors[i],'time':time.time()})
                    if attempts[i]<max_attempts:pending.append(i)
                    else:blocked[str(i)]=errors[i]
            progress(active)
    write_json(folder.parent.parent/'deferred'/('view_'+view+'.json'),{'view':view,'blocked':blocked,'time':time.time()})
    if blocked:raise GenerationDeferred('view_quality_review_required:'+view+':'+str(len(blocked)))
    return {'completed':accepted,'target':count}
