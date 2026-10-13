"""Validated deployment of the three serving processes on physical GPUs 0 and 1."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
from medagent.gpu import snapshot
from serve_small import command

def main(selection):
    chosen=read_json(selection);report=read_json(chosen['evaluation_report'])
    if report.get('samples')!=1000 or report.get('symptom_f1',0)<.90 or report.get('route_recall',0)<.95 or report.get('route_exact',0)<.85 or report.get('text_p95_seconds',999)>20:
        raise ValueError('release_acceptance_not_met')
    if report.get('workflow_schema_success_rate',0)<.99 or report.get('urgent_misses'):
        raise ValueError('schema_or_urgent_review_gate_not_met')
    if report.get('failed'):raise ValueError('unresolved_failed_cases')
    roles=['frontend','leader','specialist','answer','vision']
    paths={role:Path(chosen[role]).resolve() for role in roles}
    for role,path in paths.items():
        complete=read_json(path.parent/'complete.json')
        if complete.get('status')!='complete':raise ValueError('incomplete_checkpoint:'+role)
    locks=[];processes=[]
    folder=ROOT/'runtime/small_model/deployment';folder.mkdir(parents=True,exist_ok=True)
    try:
        for gpu in [0,1]:
            lock=(ROOT/'runtime'/f'gpu-{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        if any(g['has_process'] for g in snapshot() if g['index'] in (0,1)):raise ValueError('deployment_gpus_occupied')
        base=ROOT/'runtime/small_model/models/Qwen3.5-4B'
        adapters={'oral-'+r:str(paths[r]) for r in ['leader','specialist','answer']}
        jobs=[(0,paths['frontend'],'oral-frontend',18120,None,.28),
              (0,base,'oral-vision-base',18122,{'oral-vision':str(paths['vision'])},.45),
              (1,base,'oral-base',18121,adapters,.85)]
        for gpu,model,name,port,loras,memory in jobs:
            log=(folder/(name+'.log')).open('a')
            proc=subprocess.Popen(command(model,name,port,loras,memory),cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1'},stdout=log,stderr=log,start_new_session=True)
            processes.append(proc)
            for _ in range(240):
                if proc.poll() is not None:raise RuntimeError(name+'_startup_failed')
                try:
                    urllib.request.urlopen(f'http://deployment.invalid',timeout=2).close();break
                except Exception:time.sleep(2)
            else:raise RuntimeError(name+'_startup_timeout')
            for adapter in list(loras or {name:None}):
                body={'model':adapter,'messages':[{'role':'user','content':'仅回复：就绪'}],'max_tokens':16,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
                req=urllib.request.Request(f'http://deployment.invalid',json.dumps(body).encode(),{'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=90) as response:json.load(response)
        write_json(folder/'status.json',{'status':'ready','pids':[p.pid for p in processes],'selection':str(selection),'adapters':adapters,'prewarmed':True,'physical_gpus':[0,1]})
        while all(p.poll() is None for p in processes):time.sleep(10)
        raise RuntimeError('deployment_worker_exited')
    finally:
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for lock in locks:lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);a=p.parse_args()
    main(a.selection)
