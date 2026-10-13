"""Use GPU 1 after the existing SFT for the already requested full comparison."""
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
from medagent.gpu import reserve_specific


def main():
    folder=ROOT/'runtime/small_model/queue/external';folder.mkdir(parents=True,exist_ok=True)
    status=folder/'legacy_comparison.json';server=None;child=None
    checkpoint=ROOT/'runtime/small_model/checkpoints/legacy_only_sft/42'
    labels={'oral-baseline':'queue_base_explicit_full121','oral-ft':'queue_ft_explicit_full121'}
    def update(state,**extra):
        write_json(status,{'job':'legacy_comparison','status':state,'pid':os.getpid(),'gpu':1,
            'child_pid':child.pid if child else None,'server_pid':server.pid if server else None,
            'updated_at':time.time(),'samples_per_model':121,'labels':labels,**extra})
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    with (folder/'legacy_comparison.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while not (checkpoint/'complete.json').exists():
                update('waiting_dependency',dependency='legacy_only_sft');time.sleep(20)
            if read_json(checkpoint/'complete.json').get('status')!='complete':raise ValueError('checkpoint_not_complete')
            if not (checkpoint/'final/adapter_config.json').exists():raise ValueError('adapter_missing')
            with reserve_specific(1,lambda:update('waiting_for_gpu')):
                cfg=read_json(ROOT/'configs/small_training.json')
                adapters=folder/'legacy_comparison_adapters.json'
                write_json(adapters,{'oral-ft':str(checkpoint/'final')})
                with (folder/'legacy_comparison_server.log').open('ab') as log:
                    server=subprocess.Popen([sys.executable,'scripts/serve_small.py','--model',str(ROOT/cfg['models']['shared']),
                        '--name','oral-baseline','--port','18130','--gpu','1','--adapters',str(adapters)],
                        cwd=ROOT,stdout=log,stderr=log,start_new_session=True)
                    update('loading_model')
                    for _ in range(240):
                        if server.poll() is not None:raise RuntimeError('comparison_server_exited')
                        try:
                            with urllib.request.urlopen('http://deployment.invalid',timeout=2) as response:models=json.load(response)
                            if set(labels)<={m['id'] for m in models['data']}:break
                        except Exception:pass
                        time.sleep(2)
                    else:raise RuntimeError('comparison_server_start_timeout')
                    for model,label in labels.items():
                        with (folder/(label+'.log')).open('ab') as output:
                            child=subprocess.Popen([sys.executable,'scripts/evaluate_small.py','legacy','--endpoint','http://deployment.invalid',
                                '--model',model,'--label',label,'--policy','explicit'],cwd=ROOT,stdout=output,stderr=output)
                            while child.poll() is None:
                                progress=ROOT/'runtime/small_model/evaluation'/label/'status.json'
                                update('running',model=model,progress=read_json(progress) if progress.exists() else None);time.sleep(10)
                            if child.returncode:raise RuntimeError('comparison_exit:'+str(child.returncode))
                            result=ROOT/'runtime/small_model/evaluation'/label/'status.json'
                            if read_json(result).get('status')!='complete':raise RuntimeError('comparison_incomplete')
                    for task in ('root_canal','crack'):
                        manifests=[read_json(ROOT/'runtime/small_model/evaluation'/label/(task+'_manifest.json')) for label in labels.values()]
                        if manifests[0]['requests_sha256']!=manifests[1]['requests_sha256']:raise ValueError('comparison_inputs_differ')
                    update('complete')
        except KeyboardInterrupt:update('interrupted')
        except Exception as exc:update('failed',error=str(exc));raise
        finally:
            if child and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            if server and server.poll() is None:
                os.killpg(server.pid,signal.SIGTERM)
                try:server.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()

if __name__=='__main__':main()
