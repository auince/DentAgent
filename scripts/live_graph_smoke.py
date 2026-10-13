"""Two real cases through the deployed HTTP -> LangGraph -> local model path."""
import json,os,sys,time,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
out=ROOT/'runtime/ablation';os.umask(0o077)
while True:
 state=read_json(ROOT/'runtime/evaluation/status.json')
 if state['status']=='failed':raise RuntimeError('baseline_failed')
 if state['status']=='completed':break
 time.sleep(15)
def call(path,payload=None):
 data=json.dumps(payload,ensure_ascii=False).encode() if payload is not None else None
 req=urllib.request.Request('http://deployment.invalid'+path,data=data,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=60) as r:return json.load(r)
selected=read_json(out/'selection.json');pending={};results=[]
for task,rows in selected.items():
 row=rows[0];run=call('/api/runs',{'text':row['text'],'tasks':[task]});pending[run['run_id']]={'task':task,'case_id':row['id']}
write_json(out/'live_graph_status.json',{'status':'running','pending':pending})
start=time.monotonic()
while pending and time.monotonic()-start<2400:
 for rid in list(pending):
  report=call('/api/runs/'+rid)
  if report['status'] not in ('running','queued'):
   results.append({**pending.pop(rid),'run_id':rid,'status':report['status'],'orchestrator':report.get('orchestrator'),'graph_version':report.get('graph_version'),'elapsed_seconds':report.get('elapsed_seconds'),
    'results':{k:{'status':v['status'],'assessments':v.get('assessments'), 'issue_count':len(v.get('issues',[])),'error':v.get('error')} for k,v in report.get('results',{}).items()},'error':report.get('error')})
 write_json(out/'live_graph_status.json',{'status':'running' if pending else 'completed','pending':pending,'results':results})
 if pending:time.sleep(3)
if pending:write_json(out/'live_graph_status.json',{'status':'timeout','pending':pending,'results':results})
