"""Take over after the existing supervisor; skip blocked branches, never fake gates."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
from medagent.training_queue import snapshot,complete,external_jobs
from small_pipeline import Pipeline,OUT
from medagent.gpu import try_training_pair,idle_gpu


def run(after_pid=None):
    os.environ['MEDAGENT_EXECUTOR_VERSION']='dual-v1'
    folder=ROOT/'runtime/small_model/queue';folder.mkdir(parents=True,exist_ok=True)
    failures={};p=folder/'failures.json'
    if p.exists():failures=read_json(p)
    retry_path=folder/'generation_retries.json'
    retries=read_json(retry_path) if retry_path.exists() else {}
    retry_cfg=read_json(ROOT/'configs/training_queue.json').get('generation_retry',{})
    retry_limit=retry_cfg.get('max_stage_attempts',8)
    cooldown=retry_cfg.get('cooldown_seconds',120)
    def defer(job,error):
        failures[job['id']]=error
        if job['kind']=='generation':
            old=retries.get(job['id'],{})
            attempts=old.get('attempts',0)+1
            retries[job['id']]={'attempts':attempts,'next_at':time.time()+cooldown,
                'exhausted':attempts>=retry_limit,'error':error}
            write_json(retry_path,retries)
        write_json(p,failures)
    # Wait before taking the worker lock when replacing a queue worker itself.
    if after_pid:
        proc=Path('/proc',str(after_pid),'cmdline');stat=Path('/proc',str(after_pid),'stat')
        if proc.exists() and b'scripts/training_queue_worker.py' in proc.read_bytes():
            started=stat.read_text().split()[21]
            while stat.exists() and stat.read_text().split()[21]==started and stat.read_text().split()[2]!='Z':time.sleep(20)
            after_pid=None
    with (folder/'worker.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        expected=None
        if after_pid:
            stat=Path('/proc',str(after_pid),'stat');cmd=Path('/proc',str(after_pid),'cmdline')
            if cmd.exists():
                if b'scripts/small_pipeline.py' not in cmd.read_bytes():raise ValueError('not_medagent_supervisor')
                expected=stat.read_text().split()[21]
            while stat.exists() and expected is not None and stat.read_text().split()[21]==expected and stat.read_text().split()[2]!='Z':
                current=read_json(OUT/'status.json')
                mapping={'generate_cards_balanced_v2':'data_cards','generate_specialist_v2':'data_specialist'}
                snapshot(current=mapping.get(current.get('job'),current.get('job')),failures=failures,waiting_for=after_pid)
                time.sleep(20)
        with (OUT/'supervisor.lock').open('a') as lock:
            # Do not compete with a supervisor restarted by the user.
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            pipeline=Pipeline();cfg=read_json(ROOT/'configs/small_training.json')
            active={}
            maximum=read_json(ROOT/'configs/training_queue.json').get('execution',{}).get('max_concurrent_training_jobs',1)
            def cleanup(item,terminate=False):
                proc=item['process']
                if terminate and proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                if item['job']['kind']=='generation' and (terminate or any(s.poll() is not None for s in pipeline.servers)):
                    pipeline.stop_server()
                item['resources'].close()
            def stop(*_):raise KeyboardInterrupt
            signal.signal(signal.SIGTERM,stop)
            try:
                while True:
                    for key,item in list(active.items()):
                        if item['job']['kind']=='generation' and pipeline.servers and any(s.poll() is not None for s in pipeline.servers):
                            try:os.killpg(item['process'].pid,signal.SIGTERM)
                            except ProcessLookupError:pass
                        code=item['process'].poll()
                        if code is None:continue
                        if code or not complete(item['job']):
                            defer(item['job'],'exit_code:'+str(code) if code else 'completion_artifact_missing')
                        else:
                            retries.pop(key,None);write_json(retry_path,retries)
                        cleanup(item);del active[key]
                    for key,retry in retries.items():
                        if not retry.get('exhausted') and key in failures and retry['next_at']<=time.time():
                            failures.pop(key,None);write_json(p,failures)
                    requests=folder/'retry_requests.json'
                    if requests.exists():
                        with (folder/'requests.lock').open('a') as request_lock:
                            fcntl.flock(request_lock,fcntl.LOCK_EX)
                            retry=read_json(requests)
                            for item in retry:
                                if not item.get('applied_at'):
                                    failures.pop(item['job'],None)
                                    if item.get('reset_attempts'):
                                        if not item.get('reason'):raise ValueError('retry_reset_requires_reason')
                                        item['previous_retry']=retries.pop(item['job'],None)
                                    item['applied_at']=time.time()
                            write_json(p,failures);write_json(retry_path,retries);write_json(requests,retry)
                    report=snapshot(current=list(active),failures=failures)
                    ready=sorted((j for j in report['jobs'] if j['status']=='ready' and j['id'] not in active),key=lambda j:(j['priority'],j['id']))
                    # Reuse warm replicas across consecutive generation stages. Release
                    # them before training or when no runnable teacher work remains.
                    generation_active=any(i['job']['kind']=='generation' for i in active.values())
                    teacher_ready=any(j['kind']=='generation' and j.get('teacher_required',True) for j in ready)
                    # Quality retry cooldowns are brief. Retain healthy replicas so
                    # the next missing-row batch does not reload all teacher weights.
                    teacher_cooling=any(j['kind']=='generation' and j.get('teacher_required',True)
                        and j['id'] in failures and j['id'] in retries
                        and not retries[j['id']].get('exhausted')
                        for j in report['jobs'])
                    training_ready=any(j['kind']=='training' for j in ready)
                    keep_for_retry=teacher_cooling and not training_ready
                    if pipeline.server is not None and (any(s.poll() is not None for s in pipeline.servers) or (not generation_active and not teacher_ready and not keep_for_retry)):
                        pipeline.stop_server()
                    launched=False
                    for job in ready:
                        training=job['kind']=='training'
                        if training and sum(i['job']['kind']=='training' for i in active.values())>=maximum:continue
                        if not training and any(i['job']['kind']=='generation' for i in active.values()):continue
                        resources=ExitStack();gpu=None
                        try:
                            env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONUNBUFFERED':'1','NCCL_CUMEM_HOST_ENABLE':'0','NCCL_CUMEM_ENABLE':'0'}
                            if training:
                                gpu=resources.enter_context(try_training_pair())
                                if gpu is None:resources.close();continue
                                env['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpu))
                                args=['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
                                    'scripts/train_small.py','--stage',job['stage'],'--seed',str(job['seed']),'--resume']
                            else:
                                if job.get('teacher_required',True) and read_json(ROOT/'configs/generation_teacher.json')['backend']!='remote_api':
                                    if pipeline.server is None:
                                        if idle_gpu() is None:resources.close();continue
                                        pipeline.status('queued',job=job['id'],child_pid=None)
                                        pipeline.start_server(cfg['teacher'],'oral-teacher')
                                    gpu=[int(i) for i in str(pipeline.state['gpu']).split(',')]
                                    env['ORAL_TEACHER_ENDPOINTS']=','.join(pipeline.teacher_endpoints)
                                args=['scripts/generate_oral_data.py',job['phase']]
                                if job.get('views'):args+=['--views',*job['views']]
                            log=resources.enter_context((folder/(job['id'].replace('@','_')+'.log')).open('ab'))
                            proc=subprocess.Popen([sys.executable,*args],cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
                            active[job['id']]={'job':job,'process':proc,'resources':resources,'gpus':gpu}
                            launched=True
                        except Exception as exc:
                            if not training:pipeline.stop_server()
                            resources.close()
                            if str(exc)=='teacher_gpu_pair_unavailable':continue
                            defer(job,str(exc)[:300])
                    running=[{'job':key,'pid':item['process'].pid,'gpus':item['gpus'],'kind':item['job']['kind']} for key,item in active.items()]
                    external=external_jobs()
                    pending_retry=any(not r.get('exhausted') and key in failures for key,r in retries.items())
                    if not active and not ready and not external and not pending_retry:
                        pipeline.status('queue_waiting_for_gates_or_implementation',job=None,child_pid=None,running_jobs=[])
                        snapshot(failures=failures);return
                    status='running' if active else 'waiting_for_external_data' if external else 'waiting_for_generation_retry' if pending_retry else 'waiting_for_available_gpus'
                    pipeline.status(status,job=running[0]['job'] if len(running)==1 else None,
                        child_pid=running[0]['pid'] if len(running)==1 else None,running_jobs=running)
                    snapshot(current=list(active),failures=failures)
                    time.sleep(2 if launched else 10)
            except KeyboardInterrupt:
                pipeline.status('interrupted')
            finally:
                for item in active.values():cleanup(item,terminate=True)
                pipeline.stop_server();snapshot(failures=failures)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int);p.add_argument('--snapshot',action='store_true');a=p.parse_args()
    if a.snapshot:print(snapshot())
    else:run(a.after_pid)
