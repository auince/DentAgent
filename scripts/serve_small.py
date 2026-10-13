"""Launch one vLLM process with a shared base and request-selected role adapters."""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json

def command(model,name,port,adapters=None,memory=.85,max_model_len=8192,max_num_seqs=8):
    args=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',str(model),
        '--served-model-name',name,'--host','127.0.0.1','--port',str(port),'--dtype','bfloat16',
        '--gpu-memory-utilization',str(memory),'--max-model-len',str(max_model_len),'--max-num-seqs',str(max_num_seqs),
        '--enable-prefix-caching','--generation-config','vllm','--enforce-eager']
    if adapters:
        args+=['--enable-lora','--max-loras',str(len(adapters)),'--max-cpu-loras',str(len(adapters)),
            '--max-lora-rank','32','--lora-modules']+[k+'='+str(v) for k,v in adapters.items()]
    return args

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--name',required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--gpu',choices=['0','1','2','3','0,1','2,3','0,1,2,3'],required=True);p.add_argument('--max-model-len',type=int,default=8192);p.add_argument('--max-num-seqs',type=int,default=8);p.add_argument('--adapters');p.add_argument('--memory',type=float,default=.85);a=p.parse_args()
    if not 1<=a.max_num_seqs<=32:raise SystemExit('Invalid sequence concurrency')
    from medagent.gpu import snapshot,allowed_indices
    states={g['index']:g for g in snapshot()}
    indices=[int(i) for i in a.gpu.split(',')]
    if not set(indices)<=allowed_indices():raise SystemExit('GPU outside configured pool')
    if any(states[i]['has_process'] for i in indices):raise SystemExit('GPU occupied; use a reserved deployment window')
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    # This host's CUDA/NCCL cuMem host allocation crashes during TP initialization.
    os.environ.setdefault('NCCL_CUMEM_HOST_ENABLE','0')
    os.environ.setdefault('NCCL_CUMEM_ENABLE','0')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    args=command(a.model,a.name,a.port,read_json(a.adapters) if a.adapters else None,a.memory,a.max_model_len,a.max_num_seqs)
    if len(indices)>1:args+=['--tensor-parallel-size',str(len(indices))]
    os.execv(sys.executable,args)
