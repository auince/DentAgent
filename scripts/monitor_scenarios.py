"""Stay attached until all cases finish; record physical GPU use and process liveness."""
import json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json

directory=ROOT/'runtime/scenario_eval_v1'
last_count=-1;last_notice=0;last_error=''
audit=directory/'gpu_audit.jsonl'
while True:
    state=read_json(directory/'status.json')
    gateway=read_json(ROOT/'runtime/gateway_status.json') if (ROOT/'runtime/gateway_status.json').exists() else {}
    pid=gateway.get('worker_pid')
    if pid:
        p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        devices=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        mapping={line.split(',')[1].strip():int(line.split(',')[0]) for line in devices.stdout.splitlines()}
        used=[mapping[line.split(',')[1].strip()] for line in p.stdout.splitlines() if int(line.split(',')[0])==pid]
        with audit.open('a') as f:f.write(json.dumps({'time':time.time(),'worker_pid':pid,'physical_gpu_indices':used})+'\n')
        if any(g not in (0,1) for g in used):raise RuntimeError('GPU policy violation')
    completed=state.get('completed',0)
    if completed!=last_count:
        metrics=read_json(directory/'metrics.json') if (directory/'metrics.json').exists() else {}
        cases=metrics.get('cases',[])
        print(json.dumps({'completed':completed,'total':60,'counts':metrics.get('counts',{}),
                          'last_case':cases[-1] if cases else None,'gpu':gateway.get('gpu')},ensure_ascii=False),flush=True)
        last_count=completed;last_notice=time.time()
    elif time.time()-last_notice>=55:
        print(json.dumps({'still_running':True,'completed':completed,'pending':state.get('pending'),
                          'gpu':gateway.get('gpu'),'model_calls':gateway.get('completed_requests')},ensure_ascii=False),flush=True)
        last_notice=time.time()
    error=ROOT/'runtime/gateway_error.json'
    if error.exists():
        value=read_json(error);fingerprint=json.dumps(value,sort_keys=True)
        if fingerprint!=last_error:
            print('GATEWAY_ERROR '+fingerprint,flush=True);last_error=fingerprint
    if state['status']=='completed':
        assert completed==60 and len(list((directory/'reports').glob('*.json')))==60
        assert read_json(directory/'metrics.json')['completed']==60
        write_json(directory/'monitor_verification.json',{'completed':60,'gpu_policy_verified':True,'finished_at':time.time()})
        print('ALL_60_COMPLETE_AND_GPU_POLICY_VERIFIED',flush=True);break
    if state['status']=='failed':raise RuntimeError('Scenario supervisor failed: '+state.get('error',''))
    supervisor=state.get('pid')
    if supervisor and not (Path('/proc')/str(supervisor)).exists():raise RuntimeError('Scenario supervisor disappeared')
    time.sleep(10)
