import argparse
import concurrent.futures
import hmac
import json
import os
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .agents import LeaderAgent
from .common import ROOT, read_json, write_json,digest
from .consultation import ConsultationService
from .sessions import SessionBusy


def serve(host='127.0.0.1',port=18100):
    if host not in ('127.0.0.1','localhost','::1') and not os.environ.get('MEDAGENT_TOKEN'):
        raise RuntimeError('非回环监听必须设置 MEDAGENT_TOKEN')
    leader=LeaderAgent(); pool=concurrent.futures.ThreadPoolExecutor(max_workers=2)
    consultation=ConsultationService()
    capacity=threading.BoundedSemaphore(8)
    submission_lock=threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass  # No case text, access tokens or patient IDs in access logs.

        def send(self,status,value,kind='application/json'):
            data=(json.dumps(value,ensure_ascii=False) if kind=='application/json' else value).encode()
            self.send_response(status); self.send_header('Content-Type',kind+'; charset=utf-8')
            self.send_header('Content-Length',str(len(data))); self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff'); self.end_headers(); self.wfile.write(data)

        def authorized(self):
            token=os.environ.get('MEDAGENT_TOKEN','')
            return not token or hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token)

        def do_GET(self):
            path=urlsplit(self.path).path
            if path=='/': return self.send(200,(ROOT/'web/index.html').read_text(),'text/html')
            if path=='/consult': return self.send(200,(ROOT/'web/consult.html').read_text(),'text/html')
            if not self.authorized(): return self.send(401,{'error':'unauthorized'})
            if path=='/health': return self.send(200,{'service':'MedAgent','status':'ready','inference':'local endpoints required','orchestrator':'langgraph','graph_version':leader.graph_version})
            if path=='/api/consultation/graph':
                return self.send(200,{'graph':consultation.agent.graph.get_graph().draw_mermaid(),'version':consultation.agent.config['version']})
            if path=='/api/training':
                p=ROOT/'runtime/small_model/pipeline/status.json'
                return self.send(200,read_json(p) if p.exists() else {'status':'not_started'})
            if path=='/api/training/queue':
                p=ROOT/'runtime/small_model/queue/status.json'
                from .training_queue import queue_view
                return self.send(200,queue_view() if p.exists() else {'status':'not_started'})
            conversation=re.fullmatch('/api/conversations/([a-f0-9]{32})',path)
            if conversation:
                try:return self.send(200,consultation.store.get(conversation[1]))
                except KeyError:return self.send(404,{'error':'conversation_not_found'})
            m=re.fullmatch('/api/runs/([a-f0-9]{32})',path)
            if m:
                p=ROOT/'runtime/runs'/(m[1]+'.json')
                if p.exists(): return self.send(200,read_json(p))
            if path=='/api/evaluation':
                p=ROOT/'runtime/evaluation/status.json'
                return self.send(200,read_json(p) if p.exists() else {'status':'not_started'})
            if path=='/api/scenarios':
                directory=ROOT/'runtime/scenario_eval_v1'
                p=directory/'status.json'
                state=read_json(p) if p.exists() else {'status':'not_started'}
                state['metrics_ready']=(directory/'metrics.json').exists()
                return self.send(200,state)
            if path=='/api/ablations':
                full=ROOT/'runtime/full_test'
                if (full/'status.json').exists():
                    state=read_json(full/'status.json')
                    state['protocol']='full-test-v1'
                    state['jobs']={f.stem:read_json(f) for f in full.glob('*_status.json')}
                    state['report_ready']=(full/'results.json').exists()
                    state['historical_subset_results']='superseded'
                    return self.send(200,state)
                directory=ROOT/'runtime/ablation';p=directory/'status.json'
                state=read_json(p) if p.exists() else {'status':'not_started'}
                state['jobs']={f.stem:read_json(f) for f in directory.glob('*_status.json')}
                state['report_ready']=(directory/'results.json').exists()
                state['followup_report_ready']=(directory/'followup_results.json').exists()
                state['initial_ablation_status']=state['status']
                if any(j.get('status') in ('running','waiting','waiting_for_gpu','loading_model') for j in state['jobs'].values()):state['status']='running'
                return self.send(200,state)
            if path=='/api/graph':
                return self.send(200,{'orchestrator':'langgraph','leader':leader.graph.get_graph().draw_mermaid(),
                    'specialties':{k:v.graph.get_graph().draw_mermaid() for k,v in leader.specialists.items()}})
            self.send(404,{'error':'not_found'})

        def do_POST(self):
            if not self.authorized(): return self.send(401,{'error':'unauthorized'})
            path=urlsplit(self.path).path
            if path=='/api/conversations':return self.send(201,consultation.store.create())
            conversation=re.fullmatch('/api/conversations/([a-f0-9]{32})/messages',path)
            if path!='/api/runs' and not conversation:return self.send(404,{'error':'not_found'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if size<=0 or size>24_000_000: return self.send(413,{'error':'invalid_payload_size'})
                request=json.loads(self.rfile.read(size))
                if not isinstance(request,dict): raise ValueError()
            except (ValueError,UnicodeDecodeError): return self.send(400,{'error':'invalid_json'})
            if conversation:
                if not isinstance(request.get('text',''),str) or not isinstance(request.get('record_text',''),str) or not isinstance(request.get('images',[]),list):return self.send(400,{'error':'invalid_message'})
                if not request.get('text','').strip() and not request.get('record_text','').strip() and not request.get('images'):return self.send(400,{'error':'empty_message'})
                if not capacity.acquire(blocking=False):return self.send(429,{'error':'queue_full'})
                cid=conversation[1];rid=uuid.uuid4().hex
                try:state=consultation.store.reserve(cid,rid)
                except (KeyError,SessionBusy) as exc:
                    capacity.release();return self.send(409 if isinstance(exc,SessionBusy) else 404,{'error':str(exc)})
                write_json(ROOT/'runtime/runs'/(rid+'.json'),{'run_id':rid,'conversation_id':cid,'status':'queued'})
                def consultation_job():
                    try:consultation.run(cid,request,rid,state)
                    except Exception:pass  # Service persists the failure and releases the session lease.
                    finally:capacity.release()
                pool.submit(consultation_job)
                return self.send(202,{'run_id':rid,'conversation_id':cid,'status':'queued'})
            key=self.headers.get('Idempotency-Key','')
            if key and not re.fullmatch(r'[a-f0-9]{64}',key):return self.send(400,{'error':'invalid_idempotency_key'})
            cache=ROOT/'runtime/submissions'/(digest(key+leader.graph_version)+'.json') if key else None
            with submission_lock:
                if cache and cache.exists():return self.send(202,read_json(cache))
                if not capacity.acquire(blocking=False): return self.send(429,{'error':'queue_full'})
                rid=uuid.uuid4().hex
                write_json(ROOT/'runtime/runs'/(rid+'.json'),{'run_id':rid,'status':'queued'})
                if cache:write_json(cache,{'run_id':rid,'status':'queued'})
            def job():
                try: leader.run(request,rid)
                except Exception as exc:
                    write_json(ROOT/'runtime/runs'/(rid+'.json'),{'run_id':rid,'status':'failed','error':'execution_boundary:'+str(exc)[:400]})
                finally: capacity.release()
            pool.submit(job)
            return self.send(202,{'run_id':rid,'status':'queued'})
    print(f'MedAgent listening on http://{host}:{port}',flush=True)
    try: ThreadingHTTPServer((host,port),Handler).serve_forever()
    finally:
        pool.shutdown();leader.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=18100)
    a=p.parse_args();serve(a.host,a.port)
