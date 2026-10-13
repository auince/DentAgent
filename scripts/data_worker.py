"""Reserve GPU pairs for a local tensor-parallel generation teacher."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
import urllib.request
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
from medagent.gpu import reserve_specific,reserve_many,allowed_indices


def main(gpu):
    teacher_config=ROOT/'configs/generation_teacher.json'
    if teacher_config.exists() and read_json(teacher_config)['backend']=='remote_api':
        raise RuntimeError('local_teacher_disabled; use training_queue_worker for remote API generation')
    indices=[int(i) for i in gpu.split(',')]
    if not set(indices)<=allowed_indices() or len(set(indices))!=len(indices):raise ValueError('invalid_gpu_pool')
    folder=ROOT/'runtime/small_model/queue/external';folder.mkdir(parents=True,exist_ok=True)
    status=folder/'data_cards.json';server=None;servers=[];child=None
    def update(state,**extra):
        write_json(status,{'job':'data_cards','status':state,'pid':os.getpid(),'gpu':gpu,
            'child_pid':child.pid if child else None,'server_pid':server.pid if server else None,'server_pids':[p.pid for p in servers],'updated_at':time.time(),**extra})
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    with (folder/'data_cards.lock').open('a') as task_lock:
        fcntl.flock(task_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            reservation=reserve_many(indices,lambda:update('waiting_for_gpu'))
            with reservation:
                cfg=read_json(teacher_config)
                width=cfg.get('tensor_parallel_size',1)
                if len(indices)%width:raise ValueError('incomplete_teacher_gpu_group')
                groups=[','.join(map(str,indices[i:i+width])) for i in range(0,len(indices),width)]
                with (folder/'teacher_gpu0.log').open('ab') as log:
                    endpoints=[]
                    for rank,device in enumerate(groups):
                        port=18129+rank;endpoints.append(f'http://deployment.invalid')
                        servers.append(subprocess.Popen([sys.executable,'scripts/serve_small.py','--model',cfg['model_path'],'--name','oral-teacher','--port',str(port),'--gpu',device,'--max-model-len',str(cfg.get('max_model_len',8192))],cwd=ROOT,stdout=log,stderr=log,start_new_session=True))
                    server=servers[0]
                    update('loading_teacher')
                    for _ in range(600):
                        if any(p.poll() is not None for p in servers):raise RuntimeError('teacher_start_failed')
                        try:
                            ready=[]
                            for endpoint in endpoints:
                                with urllib.request.urlopen(endpoint+'/models',timeout=2) as response:models=json.load(response)
                                ready.append(any(m['id']=='oral-teacher' for m in models['data']))
                            if all(ready):break
                        except Exception:pass
                        time.sleep(2)
                    else:raise RuntimeError('teacher_start_timeout')
                    with (folder/'data_cards.log').open('ab') as tasklog:
                        child=subprocess.Popen([sys.executable,'scripts/generate_oral_data.py','cards'],cwd=ROOT,stdout=tasklog,stderr=tasklog,
                            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'ORAL_TEACHER_ENDPOINTS':','.join(endpoints),'ORAL_GENERATION_WORKERS':str(4*len(indices)),'PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
                        while child.poll() is None:
                            progress=ROOT/'runtime/small_modelLOCAL_RESOURCE'
                            update('running',progress=read_json(progress) if progress.exists() else None);time.sleep(10)
                        if child.returncode:raise RuntimeError('card_generation_exit:'+str(child.returncode))
                    update('complete')
        except KeyboardInterrupt:update('interrupted')
        except Exception as exc:update('failed',error=str(exc));raise
        finally:
            if child and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            for owned in servers:
                if owned.poll() is None:
                    os.killpg(owned.pid,signal.SIGTERM)
                    try:owned.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(owned.pid,signal.SIGKILL);owned.wait()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',choices=['0','1','2','3','0,1','2,3','0,1,2,3'],required=True);a=p.parse_args();main(a.gpu)
