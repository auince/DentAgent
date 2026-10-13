"""One service process owns a bounded executor and durable request statuses."""
import concurrent.futures
import fcntl
import json
import threading
import time
import uuid
from .common import digest,read_json,write_json

class QueueFull(RuntimeError):pass
class IdempotencyConflict(ValueError):pass

class JobManager:
    def __init__(self,settings,store):
        self.settings,self.store=settings,store
        self.directory=settings.path('jobs_dir');self.directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        lock_path=settings.path('runtime_dir')/'service.lock'
        lock_path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.lock_file=lock_path.open('a+');lock_path.chmod(0o600)
        try:fcntl.flock(self.lock_file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_file.close();raise RuntimeError('service_already_running_for_runtime_directory') from None
        h=settings.data['http']
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=h['workers'],thread_name_prefix='medagent')
        self.capacity=threading.BoundedSemaphore(h['max_pending_runs']);self.guard=threading.Lock()
        self.accepting=True
        for path in self.directory.glob('*.json'):
            if path.name.startswith('idempotency-'):continue
            state=read_json(path)
            if state.get('status') in {'queued','running'}:
                state.update(status='interrupted',error='service_process_ended_before_completion',finished_at=time.time())
                write_json(path,state)
                if state.get('conversation_id'):self.store.release(state['conversation_id'],state['run_id'])

    def get(self,rid):
        p=self.directory/(rid+'.json')
        if not p.exists():raise KeyError('run_not_found')
        state=read_json(p)
        if state.get('status') in {'completed','failed'}:
            directory=self.settings.path({'assessment':'assessment_runs_dir','consultation':'conversation_runs_dir','retrieval_index':'index_runs_dir'}[state['kind']])
            result=directory/(rid+'.json')
            if result.exists():state['result']=read_json(result)
        return state

    def submit(self,kind,request,runner,cid=None,key=None):
        from .sessions import SessionBusy
        body_sha=digest(json.dumps(request,sort_keys=True,ensure_ascii=False))
        cache=self.directory/('idempotency-'+digest(str(self.settings.data['version'])+':'+kind+':'+str(cid)+':'+key)+'.json') if key else None
        with self.guard:
            if not self.accepting:raise QueueFull('service_shutting_down')
            if cache and cache.exists():
                old=read_json(cache)
                if old['body_sha256']!=body_sha:raise IdempotencyConflict('idempotency_key_reused_for_different_body')
                return self.get(old['run_id'])
            if not self.capacity.acquire(blocking=False):raise QueueFull('request_queue_full')
            rid=uuid.uuid4().hex;reserved=None;submitted=False
            try:
                if cid:reserved=self.store.reserve(cid,rid)
                state={'run_id':rid,'kind':kind,'status':'queued','created_at':time.time()}
                if cid:state['conversation_id']=cid
                path=self.directory/(rid+'.json');write_json(path,state)
                def execute():
                    try:
                        state.update(status='running',started_at=time.time());write_json(path,state)
                        result=runner(rid,reserved)
                        state.update(status='failed' if result.get('status')=='failed' else 'completed',result_status=result.get('status'),finished_at=time.time())
                        write_json(path,state)
                    except Exception as exc:
                        if cid:self.store.release(cid,rid)
                        state.update(status='failed',error=str(exc)[:300],finished_at=time.time());write_json(path,state)
                    finally:self.capacity.release()
                accepted=dict(state)
                if cache:write_json(cache,{'run_id':rid,'body_sha256':body_sha})
                self.pool.submit(execute);submitted=True
                return accepted
            except Exception:
                if not submitted:
                    if cid and reserved is not None:self.store.release(cid,rid)
                    if cache and cache.exists():cache.unlink()
                    if 'path' in locals() and path.exists():
                        state.update(status='failed',error='request_submission_failed',finished_at=time.time());write_json(path,state)
                    self.capacity.release()
                raise

    def close(self):
        with self.guard:self.accepting=False
        self.pool.shutdown(wait=True)
        fcntl.flock(self.lock_file,fcntl.LOCK_UN);self.lock_file.close()
