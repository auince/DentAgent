"""Persistent, resumable job supervisor; never touches foreign GPU processes."""
import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from contextlib import ExitStack
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
from medagent.gpu import snapshot,reserve_training_pair,reserve_specific,allowed_indices,try_training_pair

OUT=ROOT/'runtime/small_model/pipeline'

class Pipeline:
    def __init__(self,priority_specialties=False):
        OUT.mkdir(parents=True,exist_ok=True)
        self.child=None;self.server=None;self.server_resources=None;self.servers=[];self.teacher_endpoints=[]
        self.priority_specialties=priority_specialties
        self.state=read_json(OUT/'status.json') if (OUT/'status.json').exists() else {'completed':[]}
        if self.state.get('error'):
            self.state['previous_error']=self.state.pop('error')
        self.state.update(pid=os.getpid(),allowed_physical_gpus=sorted(allowed_indices()),status='starting',server_pid=None,child_pid=None,model=None,gpu=None,occupied=[],training_progress=None,running_jobs=[])
    def status(self,status,**fields):
        self.state.update(status=status,updated_at=time.time(),**fields);write_json(OUT/'status.json',self.state)
    def run_job(self,name,args,gpu=None):
        if name in self.state['completed']:return
        training=bool(args and Path(args[0]).name=='train_small.py')
        if training and self.server is not None:
            raise RuntimeError('stop_owned_inference_server_before_dual_gpu_training')
        self.state.pop('error',None)
        self.status('running',job=name,server_pid=self.server.pid if self.server else None,model=None if not self.server else self.state.get('model'),gpu=gpu if gpu is not None else self.state.get('gpu'))
        env={**os.environ,'PYTHONUNBUFFERED':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','LANGSMITH_TRACING':'false','LANGCHAIN_TRACING_V2':'false'}
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        if self.teacher_endpoints:env['ORAL_TEACHER_ENDPOINTS']=','.join(self.teacher_endpoints)
        with ExitStack() as resources:
            if training:
                pair=resources.enter_context(reserve_training_pair(lambda:self.status('waiting_for_both_gpus',job=name,gpu=None,child_pid=None)))
                env['CUDA_VISIBLE_DEVICES']=','.join(map(str,pair))
                args=['-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',*args]
                self.status('running',job=name,gpu=pair,world_size=2)
            log=resources.enter_context((OUT/(name+'.log')).open('a'))
            self.child=subprocess.Popen([sys.executable,*args],cwd=ROOT,env=env,stdout=log,stderr=log)
            self.status('running',job=name,child_pid=self.child.pid)
            while self.child.poll() is None:
                progress=ROOT/'runtime/small_model/checkpoints'/name/'42/progress.json'
                self.status('running',job=name,training_progress=read_json(progress) if progress.exists() else None)
                time.sleep(10)
            code=self.child.returncode;self.child=None
        if code:raise RuntimeError(name+' failed with exit '+str(code)+'; see '+name+'.log')
        self.state['completed'].append(name);self.status('running',job=name)
    def adopt_training(self,pid):
        """Wait for a verified existing training process without interrupting it."""
        proc=Path('/proc',str(pid),'cmdline')
        if proc.exists():
            cmd=proc.read_bytes()
            if b'scripts/train_small.py' not in cmd or b'legacy_only_sft' not in cmd:
                raise RuntimeError('refusing_to_adopt_unrelated_process')
        marker=ROOT/'runtime/small_model/checkpoints/legacy_only_sft/42/complete.json'
        log=OUT/'legacy_only_sft.log'
        while proc.exists():
            tail=''
            if log.exists():
                with log.open('rb') as f:
                    f.seek(max(0,log.stat().st_size-131072));tail=f.read().decode(errors='replace')
            matches=re.findall(r'(\d+)/(536)',tail)
            self.status('running',job='legacy_only_sft',child_pid=pid,server_pid=None,model=None,gpu=1,
                training_progress={'step':int(matches[-1][0]),'total_steps':536} if matches else None)
            if marker.exists():break
            time.sleep(15)
        if not marker.exists():raise RuntimeError('adopted_training_exited_without_completion')
        if 'legacy_only_sft' not in self.state['completed']:self.state['completed'].append('legacy_only_sft')
        self.status('running',child_pid=None)
    def available_gpu(self):
        while True:
            # Inference/preparation may use one GPU; training reserves the pair.
            states={g['index']:g for g in snapshot()}
            for gpu in sorted(allowed_indices()):
                g=states[gpu]
                if not g['has_process'] and g['free_mib']>=40000:return gpu
            self.status('waiting_for_gpu',occupied=[g for i,g in states.items() if i in allowed_indices()]);time.sleep(20)
    def start_server(self,model,name,adapters=None):
        teacher_config=ROOT/'configs/generation_teacher.json'
        if name=='oral-teacher' and teacher_config.exists() and read_json(teacher_config)['backend']=='remote_api':
            self.status('remote_teacher',model=read_json(teacher_config)['model'],gpu=None,server_pid=None)
            return
        if self.server is not None:raise RuntimeError('inference_server_already_owned')
        self.server_resources=ExitStack()
        generation=read_json(teacher_config) if name=='oral-teacher' and teacher_config.exists() else None
        if generation and generation.get('backend')=='local':
            pair=self.server_resources.enter_context(try_training_pair())
            if pair is None:
                self.server_resources.close();self.server_resources=None
                raise RuntimeError('teacher_gpu_pair_unavailable')
            groups=[','.join(map(str,pair))]
            second=self.server_resources.enter_context(try_training_pair())
            if second is not None:groups.append(','.join(map(str,second)))
            gpu=','.join(groups);model=generation['model_path']
        else:
            gpu=self.available_gpu()
            self.server_resources.enter_context(reserve_specific(gpu,lambda:self.status('waiting_for_gpu',gpu=gpu)))
            groups=[str(gpu)]
        endpoints=[]
        with (OUT/'server.log').open('a') as log:
            for rank,group in enumerate(groups):
                port=18129+rank;endpoints.append(f'http://deployment.invalid')
                args=[sys.executable,'scripts/serve_small.py','--model',str(model),'--name',name,'--port',str(port),'--gpu',group]
                if generation:args+=['--max-model-len',str(generation.get('max_model_len',8192)),
                    '--max-num-seqs',str(generation.get('max_num_seqs',8))]
                if adapters:
                    path=OUT/'serving_adapters.json';write_json(path,adapters);args+=['--adapters',str(path)]
                self.servers.append(subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=log,start_new_session=True))
        self.server=self.servers[0]
        self.teacher_endpoints=endpoints if generation else []
        self.status('loading_model',server_pid=self.server.pid,server_pids=[p.pid for p in self.servers],model=name,gpu=gpu)
        for _ in range(600):
            if any(p.poll() is not None for p in self.servers):raise RuntimeError('inference_server_failed; see server.log')
            try:
                ready=[]
                for endpoint in endpoints:
                    with urllib.request.urlopen(endpoint+'/models',timeout=2) as r:models=json.load(r)
                    ready.append(name in [m['id'] for m in models['data']])
                if all(ready):return
            except Exception:pass
            time.sleep(2)
        raise RuntimeError('inference_server_start_timeout')
    def stop_server(self):
        for server in self.servers:
            try:
                # The API process can exit while its TP workers still own GPU memory.
                os.killpg(server.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            if server.poll() is None:
                try:server.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
        self.server=None;self.servers=[];self.teacher_endpoints=[]
        if self.server_resources is not None:
            self.server_resources.close();self.server_resources=None
        self.state.update(server_pid=None,server_pids=[],model=None)
    def prepare_specialties(self,cfg):
        if 'generate_specialist_v2' not in self.state['completed']:
            self.status('queued',job='generate_cards_balanced_v2',child_pid=None,server_pid=None,model=None)
            self.start_server(cfg['teacher'],'oral-teacher')
            self.run_job('generate_cards_balanced_v2',['scripts/generate_oral_data.py','cards'])
            self.run_job('generate_specialist_v2',['scripts/generate_oral_data.py','views','--views','specialist'])
            self.stop_server()
        if 'specialist_sft' not in self.state['completed']:
            self.status('queued',job='specialist_sft')
            self.run_job('specialist_sft',['scripts/train_small.py','--stage','specialist_sft','--seed','42','--resume'])
    def run(self):
        cfg=read_json(ROOT/'configs/small_training.json')
        try:
            while True:
                manifest=ROOT/'runtime/small_model/models/manifest.json'
                models=read_json(manifest) if manifest.exists() else {}
                if all(models.get(n,{}).get('status')=='downloaded' for n in ['Qwen3.5-2B','Qwen3.5-4B']):break
                self.status('waiting_for_model_download',job='models');time.sleep(20)
            self.run_job('prepare',['scripts/prepare_small_data.py'])
            if 'preflight_gpu' not in self.state['completed']:
                self.run_job('preflight_gpu',['scripts/preflight_small.py','--gpu-check'],self.available_gpu())
            if self.priority_specialties:self.prepare_specialties(cfg)
            if 'baseline_aligned_full121' not in self.state['completed']:
                self.start_server(ROOT/cfg['models']['shared'],'oral-baseline')
                self.run_job('baseline_aligned_full121',['scripts/evaluate_small.py','legacy','--label','qwen35_4b_base_aligned','--model','oral-baseline','--policy','training_aligned'])
                self.stop_server()
            # This real-data control can train before the new synthetic corpus exists.
            if 'legacy_only_sft' not in self.state['completed']:
                self.run_job('legacy_only_sft',['scripts/train_small.py','--stage','legacy_only_sft','--seed','42','--resume'])
            if 'legacy_sft_full121' not in self.state['completed']:
                adapter=ROOT/'runtime/small_model/checkpoints/legacy_only_sft/42/final'
                self.start_server(ROOT/cfg['models']['shared'],'oral-baseline',{'oral-ft':str(adapter)})
                self.run_job('baseline_explicit_full121',['scripts/evaluate_small.py','legacy','--label','base_prompt','--model','oral-baseline','--policy','explicit'])
                self.run_job('legacy_sft_full121',['scripts/evaluate_small.py','legacy','--label','legacy_only_sft','--model','oral-ft','--policy','explicit'])
                self.stop_server()
            self.prepare_specialties(cfg)
            if 'generate_remaining_v2' not in self.state['completed']:
                self.start_server(cfg['teacher'],'oral-teacher')
                self.run_job('generate_remaining_v2',['scripts/generate_oral_data.py','views','--views','leader','frontend','answer_dpo','vision','retrieval'])
                self.stop_server()
            for stage in cfg['stages']:
                name=stage['id']
                if name in self.state['completed']:continue
                if stage['method']=='opd' and not (ROOT/'runtime/small_model/evaluation/teacher_gate.json').exists():
                    self.status('needs_validation_gate',job=name,reason='OPD requires measured teacher > student on validation');return
                if stage['method'].startswith('agent_') and not (ROOT/'runtime/small_model/frozen_models.json').exists():
                    self.status('needs_validation_gate',job=name,reason='Agent RL requires evaluated frozen experts');return
                self.run_job(name,['scripts/train_small.py','--stage',name,'--seed','42','--resume'])
            self.status('training_complete_evaluation_pending',job='full_evaluation_and_three_seed_selection')
        except Exception as exc:
            self.status('failed',error=str(exc));raise
        finally:self.stop_server()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--adopt-training-pid',type=int);parser.add_argument('--priority-specialties',action='store_true');args=parser.parse_args()
    os.umask(0o077);OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pipeline=Pipeline(args.priority_specialties)
        def stop(*_):
            if pipeline.child and pipeline.child.poll() is None:pipeline.child.terminate()
            pipeline.stop_server();pipeline.status('interrupted');raise SystemExit(0)
        signal.signal(signal.SIGTERM,stop)
        if args.adopt_training_pid:
            try:pipeline.adopt_training(args.adopt_training_pid)
            except Exception as exc:pipeline.status('failed',error=str(exc));raise
        pipeline.run()
