"""Run dual-GPU early Agentic RL with a fixed offline teacher on the other pair."""
import os
import sys
import signal
import time
import json
import subprocess
import urllib.request
from pathlib import Path
from contextlib import ExitStack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
from medagent.gpu import reserve_many,snapshot
from medagent.frozen_models import checkpoint_fingerprint,validate_frozen_bindings

RUNTIME=ROOT/'runtime/small_model'

def run():
    os.environ['MEDAGENT_EARLY_RL_EXECUTOR']='dual-v1'
    teacher=None;training=None;interruption_reason='signal_or_interrupt'
    job='leader_base_grpo'
    status_file=RUNTIME/'queue/external'/('agentic_'+job+'.json')
    stage_file=RUNTIME/'pipeline/status.json'
    def status(value,**details):
        entry={'job':job,'kind':'training','status':value,'pid':os.getpid(),
            'world_size':2,'policy_gpus':[0,1],'offline_teacher_gpus':[2,3],
            'initialization':'base_model_no_sft','updated_at':time.time(),**details}
        write_json(status_file,entry)
        active=value in {'loading_teacher','loading_policy','running','retrying_training'}
        teacher_pid=teacher.pid if active and teacher and teacher.poll() is None else None
        training_pid=training.pid if active and training and training.poll() is None else None
        write_json(stage_file,{**entry,'allowed_physical_gpus':[0,1,2,3],
            'server_pid':teacher_pid,'server_pids':[teacher_pid] if teacher_pid else [],
            'child_pid':training_pid,
            'running_jobs':[{'job':job,'pid':training_pid,'gpus':[0,1],'kind':'training'}]
                if value=='running' and training_pid else []})
    def terminate(proc):
        if proc and proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:return
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        with ExitStack() as resources:
            resources.enter_context(reserve_many([2,3],lambda:status('waiting_for_gpu')))
            resources.enter_context(reserve_many([0,1],lambda:status('waiting_for_gpu')))
            source=read_json(ROOT/'configs/generation_teacher.json')['model_path']
            env={**os.environ,'NCCL_CUMEM_HOST_ENABLE':'0','NCCL_CUMEM_ENABLE':'0',
                'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONUNBUFFERED':'1'}
            teacher_log=resources.enter_context((RUNTIME/'queue/early_rl_teacher.log').open('ab'))
            teacher=subprocess.Popen([sys.executable,'scripts/serve_small.py','--model',source,
                '--name','oral-rl-offline-teacher','--port','18130','--gpu','2,3',
                '--max-model-len','16384','--max-num-seqs','8'],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=teacher_log,stderr=teacher_log,start_new_session=True)
            status('loading_teacher')
            # Fingerprint while the server loads; this is an integrity check,
            # not a fabricated checkpoint-completion or medical performance gate.
            fingerprint=checkpoint_fingerprint(source)
            endpoint='http://deployment.invalid'
            for _ in range(600):
                if teacher.poll() is not None:raise RuntimeError('offline_teacher_server_failed')
                try:
                    with urllib.request.urlopen(endpoint+'/models',timeout=2) as response:
                        models=json.load(response)['data']
                    if any(m['id']=='oral-rl-offline-teacher' for m in models):break
                except (OSError,ValueError):pass
                time.sleep(2)
            else:raise RuntimeError('offline_teacher_start_timeout')
            binding={'checkpoint':source,'checkpoint_sha256':fingerprint,
                'checkpoint_kind':'offline_teacher_base','endpoint':endpoint,
                'model':'oral-rl-offline-teacher','temperature':0,'timeout':180,'retries':1}
            frozen={'validated':True,'validation_kind':'checkpoint_integrity_and_endpoint_identity',
                'clinical_validation':False,'offline_teacher_only':True,'online_deployment_eligible':False,
                'models':{'specialist':{**binding,'max_tokens':1536},'needs':{**binding,'max_tokens':2048}}}
            validate_frozen_bindings(frozen)
            write_json(RUNTIME/'frozen_offline_teacher.json',frozen)
            train_env={**env,'CUDA_VISIBLE_DEVICES':'0,1'}
            log=resources.enter_context((RUNTIME/'queue/leader_base_grpo.log').open('ab'))
            for attempt in range(5):
                policy_states=[g for g in snapshot() if g['index'] in (0,1)]
                if len(policy_states)!=2 or any(g['has_process'] or g['free_mib']<40000
                        for g in policy_states):
                    interruption_reason='gpu_conflict_reappeared_before_policy_load'
                    status('waiting_for_gpu',reason=interruption_reason)
                    raise KeyboardInterrupt
                attempt_started_at=time.time()
                training=subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone',
                    '--nnodes=1','--nproc_per_node=2','scripts/train_small.py','--stage',job,
                    '--seed','42','--resume'],cwd=ROOT,env=train_env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=log,start_new_session=True)
                while training.poll() is None:
                    if teacher.poll() is not None:raise RuntimeError('offline_teacher_exited_during_rl')
                    progress=RUNTIME/'checkpoints'/job/'42/progress.json'
                    saved=read_json(progress) if progress.exists() else None
                    fresh=bool(saved and saved.get('status')=='training'
                        and saved.get('updated_at',0)>=attempt_started_at)
                    status('running' if fresh else 'loading_policy',attempt=attempt+1,
                        training_progress=saved if fresh else None,
                        saved_training_progress=saved,optimization_started=fresh)
                    time.sleep(10)
                if training.returncode==0:break
                if attempt==4:raise RuntimeError('agentic_rl_exit_code:'+str(training.returncode))
                status('retrying_training',attempt=attempt+1,last_exit_code=training.returncode)
                time.sleep(30)
            proof=RUNTIME/'checkpoints'/job/'42/complete.json'
            if not proof.exists():raise RuntimeError('agentic_rl_completion_proof_missing')
            status('complete',completion=read_json(proof))
    except KeyboardInterrupt:
        status('interrupted',reason=interruption_reason)
    except Exception as exc:
        status('failed',error=str(exc))
        raise
    finally:
        terminate(training);terminate(teacher)

if __name__=='__main__':run()
