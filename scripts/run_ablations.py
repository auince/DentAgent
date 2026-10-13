"""Detached ablation supervisor: independent cases, fixed prompts, resumable stages."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,write_json
out=ROOT/'runtime/ablation';out.mkdir(parents=True,exist_ok=True,mode=0o700)
os.umask(0o077)
lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

def stage(name,jobs):
    write_json(out/'status.json',{'status':'running','stage':name,'updated_at':time.time(),'pid':os.getpid()})
    children=[]
    for args in jobs:
        log=(out/('_'.join(args).replace('/','_')+'.log')).open('ab')
        children.append(subprocess.Popen([sys.executable,str(ROOT/'scripts/ablation_worker.py'),*args],stdout=log,stderr=log));log.close()
    try:
        codes=[p.wait() for p in children]
        if any(codes):raise RuntimeError(name+':'+str(codes))
    except BaseException:
        for p in children:
            if p.poll() is None:p.terminate()
        raise
try:
    stage('freeze_selection',[['select']])
    stage('contract_and_repair', [['extract','--task','root_canal','--gpu','3'],['extract','--task','crack','--gpu','2']])
    stage('text_review',[['review','--gpu','2']])
    stage('review_reflow',[['reflow','--task','root_canal','--gpu','3'],['reflow','--task','crack','--gpu','2']])
    stage('evidence_grounding',[['ground','--gpu','2']])
    stage('rag_explanation',[['explain','--gpu','2']])
    write_json(out/'status.json',{'status':'completed','updated_at':time.time()})
except BaseException as exc:
    write_json(out/'status.json',{'status':'failed','error':str(exc),'updated_at':time.time()});raise
