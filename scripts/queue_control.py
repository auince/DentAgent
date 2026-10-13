"""Request a retry at a safe job boundary without killing running jobs."""
import argparse
import fcntl
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
from medagent.training_queue import complete

def requeue(job_id,reason):
    cfg=read_json(ROOT/'configs/training_queue.json')
    job=next((j for j in cfg['jobs'] if j['id']==job_id),None)
    if job is None:raise ValueError('unknown_job')
    if complete(job):raise ValueError('already_complete; explicit new experiment required')
    folder=ROOT/'runtime/small_model/queue';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'requests.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=folder/'retry_requests.json';rows=read_json(path) if path.exists() else []
        if not any(r['job']==job_id and not r.get('applied_at') for r in rows):
            rows.append({'job':job_id,'reason':reason,'requested_at':time.time()});write_json(path,rows)
    return {'job':job_id,'status':'requeue_requested','checkpoint_policy':'resume latest saved checkpoint; otherwise restart','reason':reason}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('--reason',required=True);a=p.parse_args();print(requeue(a.job,a.reason))
