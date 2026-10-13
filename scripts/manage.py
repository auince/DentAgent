"""Start/read status for owned processes. No scheduler suggestion or implicit approval."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json


def start(name,args):
    directory=ROOT/'runtime/processes'; directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    pidfile=directory/(name+'.json')
    if pidfile.exists():
        old=read_json(pidfile); proc=Path('/proc')/str(old['pid'])/'cmdline'
        if proc.exists() and all(x.encode() in proc.read_bytes() for x in old['args']):
            return {'name':name,'status':'already_running',**old}
    log=(directory/(name+'.log')).open('ab')
    proc=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,env={**os.environ,'PYTHONUNBUFFERED':'1','LANGSMITH_TRACING':'false','LANGCHAIN_TRACING_V2':'false'})
    result={'pid':proc.pid,'args':args,'started_at':time.time()};write_json(pidfile,result)
    time.sleep(.4)
    if proc.poll() is not None: raise RuntimeError(f'{name} exited; inspect {name}.log')
    return {'name':name,'status':'started',**result}


if __name__=='__main__':
    os.umask(0o077)
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','status','evaluate','full-test','scenarios','stop']);parser.add_argument('--name',choices=['api','gateway','evaluation','full_test','scenarios']);a=parser.parse_args()
    if a.action=='start':
        print(json.dumps([start('gateway',[str(ROOT/'scripts/model_gateway.py')]),start('api',['-m','medagent.server'])],ensure_ascii=False))
    elif a.action=='evaluate': print(json.dumps(start('evaluation',[str(ROOT/'scripts/eval_worker.py')]),ensure_ascii=False))
    elif a.action=='full-test': print(json.dumps(start('full_test',[str(ROOT/'scripts/full_test.py'),'all']),ensure_ascii=False))
    elif a.action=='scenarios': print(json.dumps(start('scenarios',[str(ROOT/'scripts/scenario_eval.py'),'run']),ensure_ascii=False))
    elif a.action=='stop':
        if not a.name: raise SystemExit('--name required')
        p=ROOT/'runtime/processes'/(a.name+'.json')
        info=read_json(p);proc=Path('/proc')/str(info['pid'])/'cmdline'
        if proc.exists() and all(x.encode() in proc.read_bytes() for x in info['args']):
            os.killpg(info['pid'],signal.SIGTERM)
            print(json.dumps({'stopped':a.name,'pid':info['pid']}))
    else:
        result={}
        for p in (ROOT/'runtime/processes').glob('*.json'):
            info=read_json(p); proc=Path('/proc')/str(info['pid'])/'cmdline'
            result[p.stem]={**info,'alive':proc.exists() and all(x.encode() in proc.read_bytes() for x in info['args'])}
        p=ROOT/'runtime/evaluation/status.json'
        if p.exists():result['evaluation_progress']=read_json(p)
        print(json.dumps(result,ensure_ascii=False,indent=2))
