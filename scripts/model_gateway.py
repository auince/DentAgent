"""Local OpenAI-compatible endpoint with on-demand model loading on an idle GPU.

One subprocess per request releases all CUDA resources. This mode favors shared-server
safety over latency; production serving can replace it with vLLM without changing agents.
"""
import base64
import io
import json
import os
import subprocess
import sys
import time
import fcntl
import select
import signal
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
from medagent.gpu import exclusive,idle_gpu


class PersistentWorker:
    def __init__(self):self.process=None;self.gpu=None;self.lease=None;self.completed=0;self.buffer=b''
    def start(self):
        policy=read_json(ROOT/'configs/runtime.json')
        deadline=time.monotonic()+policy['gpu_wait_seconds']
        gpu=idle_gpu()
        while gpu is None and time.monotonic()<deadline:
            write_json(ROOT/'runtime/gateway_status.json',{'status':'waiting_for_gpu','allowed_gpus':[0,1]})
            time.sleep(2);gpu=idle_gpu()
        if gpu is None:raise RuntimeError('no_idle_permitted_gpu')
        if gpu not in (0,1):raise RuntimeError('disallowed_gpu')
        lease=(ROOT/'runtime'/f'gpu-{gpu}.lock').open('a')
        fcntl.flock(lease,fcntl.LOCK_EX)
        self.lease=lease;self.gpu=gpu
        log=(ROOT/'runtime/processes/gateway_worker.log').open('ab')
        self.process=subprocess.Popen([sys.executable,__file__,'--worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,
            text=True,bufsize=1,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONUNBUFFERED':'1'})
        log.close()
    def generate(self,request):
        if self.process is None or self.process.poll() is not None:
            self.close();self.start()
        self.process.stdin.write(json.dumps(request,ensure_ascii=False)+'\n');self.process.stdin.flush()
        deadline=time.monotonic()+read_json(ROOT/'configs/runtime.json')['generation_timeout_seconds']
        while time.monotonic()<deadline:
            if b'\n' not in self.buffer:
                ready,_,_=select.select([self.process.stdout],[],[],max(0,deadline-time.monotonic()))
                if not ready:break
                chunk=os.read(self.process.stdout.fileno(),65536)
                if not chunk:raise RuntimeError('inference_worker_exited')
                self.buffer+=chunk
                continue
            line,self.buffer=self.buffer.split(b'\n',1)
            if not line.startswith(b'MEDAGENT_RESULT:'):continue
            reply=json.loads(line[len(b'MEDAGENT_RESULT:'):])
            if not reply['ok']:raise RuntimeError(reply['error'])
            self.completed+=1;result=reply['result'];result['runtime']['physical_gpu']=self.gpu
            write_json(ROOT/'runtime/gateway_status.json',{'status':'ready','allowed_gpus':[0,1],'gpu':self.gpu,
                'worker_pid':self.process.pid,'completed_requests':self.completed,'cached_models':result['runtime']['cached_models']})
            return result
        self.close();raise subprocess.TimeoutExpired('persistent_inference',600)
    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.process=None
        self.buffer=b''
        if self.lease:self.lease.close();self.lease=None


def generate(request):
    import torch
    from transformers import AutoConfig,AutoTokenizer,AutoProcessor,AutoModelForCausalLM,AutoModelForImageTextToText
    cfg=read_json(ROOT/'configs/gateway.json')
    model_path=cfg['models'][request['model']]
    config=AutoConfig.from_pretrained(model_path,trust_remote_code=False)
    vision=config.model_type in ('qwen3_vl','qwen2_5_vl')
    images=[]; messages=request['messages']
    if vision:
        from PIL import Image
        for msg in messages:
            if isinstance(msg['content'],list):
                content=[]
                for part in msg['content']:
                    if part['type']=='image_url':
                        url=part['image_url']['url']
                        if not url.startswith('data:image/'): raise ValueError('remote_image_not_allowed')
                        im=Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1],validate=True))).convert('RGB')
                        im.thumbnail((1536,1536));images.append(im);content.append({'type':'image'})
                    else: content.append(part)
                msg['content']=content
        tokenizer=AutoProcessor.from_pretrained(model_path,trust_remote_code=False)
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=tokenizer(text=[text],images=images or None,return_tensors='pt')
        cls=AutoModelForImageTextToText
    else:
        tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=False)
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=tokenizer(text,return_tensors='pt');cls=AutoModelForCausalLM
    count=inputs['input_ids'].shape[1]
    if count>16000: raise ValueError('input_too_long')
    model=cls.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map='cuda:0',trust_remote_code=False).eval()
    maximum=min(int(request.get('max_tokens',4096)),4096)
    with torch.inference_mode():
        output=model.generate(**inputs.to('cuda:0'),max_new_tokens=maximum,do_sample=False)
    completion=tokenizer.batch_decode(output[:,count:],skip_special_tokens=True)[0]
    return {'choices':[{'message':{'role':'assistant','content':completion},
        'finish_reason':'length' if output.shape[1]-count>=maximum else 'stop'}]}


def serve():
    worker=PersistentWorker()
    def shutdown(*args):
        worker.close();raise SystemExit(0)
    signal.signal(signal.SIGTERM,shutdown)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def response(self,code,value):
            raw=json.dumps(value,ensure_ascii=False).encode()
            self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            self.response(200,{'status':'ready','mode':'persistent_worker','allowed_gpus':[0,1],'worker_gpu':worker.gpu})
        def do_POST(self):
            if self.path!='/v1/chat/completions': return self.response(404,{'error':'not_found'})
            try:
                n=int(self.headers.get('Content-Length','0'))
                if n<=0 or n>24_000_000: return self.response(413,{'error':'payload_too_large'})
                data=json.loads(self.rfile.read(n)); cfg=read_json(ROOT/'configs/gateway.json')
                if data.get('model') not in cfg['models']: return self.response(400,{'error':'unknown_model'})
                state=ROOT/'runtime/evaluation/status.json'
                if state.exists() and read_json(state).get('status') in ('loading_model','running'):
                    return self.response(503,{'error':'evaluation_in_progress'})
                with exclusive():
                    result=worker.generate(data)
                self.response(200,result)
            except subprocess.TimeoutExpired: self.response(504,{'error':'generation_timeout'})
            except Exception as exc:
                write_json(ROOT/'runtime/gateway_error.json',{'time':time.time(),'error':str(exc)[:600],'gpu':worker.gpu})
                self.response(500,{'error':'model_generation_failed'})
    # Serial handling gives the parent a bounded number of CUDA workers.
    from http.server import HTTPServer
    HTTPServer(('127.0.0.1',18101),Handler).serve_forever()


if __name__=='__main__':
    if '--worker' in sys.argv:
        from medagent.gateway_runtime import worker_loop
        worker_loop()
    elif '--generate' in sys.argv: print(json.dumps(generate(json.load(sys.stdin)),ensure_ascii=False))
    else: serve()
