"""Run all 160 development workflows; no case sampling or clinical-validity claim."""
import argparse
import base64
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json,digest
from medagent.consultation import ConsultationAgent,ConsultationService
from medagent.sessions import SessionStore
from medagent.retrieval import KnowledgeIndex

class FaultIndex:
    def __init__(self,base,faults):
        self.base=base;self.remaining=sum(f['times'] for f in faults if f['tool']=='knowledge_search');self.injected=0
    def search(self,*args,**kwargs):
        if self.remaining:
            self.remaining-=1;self.injected+=1;raise TimeoutError('simulated_retrieval_timeout')
        return self.base.search(*args,**kwargs)

def materialize(request):
    request=dict(request)
    request['images']=['data:image/jpeg;base64,'+base64.b64encode((ROOT/p).read_bytes()).decode() for p in request.pop('image_files',[])]
    return request

def main(label):
    path=ROOT/'runtime/small_model/development_scenarios/manifest.json';manifest=read_json(path)
    if manifest['count']!=160 or len(manifest['scenarios'])!=160:raise ValueError('complete_160_workflows_required')
    out=ROOT/'runtime/small_model/evaluation'/label
    protocol={'manifest_sha256':digest(path.read_bytes()),'config':read_json(ROOT/'configs/consultation.json'),'count':160,'development_only':True}
    if (out/'manifest.json').exists() and read_json(out/'manifest.json')!=protocol:raise ValueError('evaluation_protocol_changed')
    write_json(out/'manifest.json',protocol);results=[];base=KnowledgeIndex(ROOT/'knowledge')
    for fixture in manifest['scenarios']:
        target=out/'cases'/(fixture['id']+'.json')
        if target.exists():results.append(read_json(target));continue
        index=FaultIndex(base,fixture.get('fault_injections',[]))
        service=ConsultationService(agent=ConsultationAgent(index=index),store=SessionStore(out/'sessions.sqlite'),runtime=out/'runs')
        cid=service.store.create()['conversation_id'];turns=[];started=time.monotonic();error=None
        try:
            result=service.run(cid,materialize(fixture['initial_request']));turns.append(result)
            for turn in fixture['followup_messages']:
                if result['status']=='urgent_referral':break
                result=service.run(cid,materialize(turn['request']));turns.append(result)
            for _ in range(3):
                if result['status']!='awaiting_patient':break
                action=next((t['action'] for t in reversed(result['trace']) if t.get('action',{}).get('action')=='ask'),{})
                answer=fixture['simulator_answers'].get(action.get('field'))
                result=service.run(cid,answer if isinstance(answer,dict) else {'text':'不知道。'});turns.append(result)
        except Exception as exc:error=str(exc)[:300]
        reference=fixture['reference'];called=set().union(*(set(t.get('experts',{})) for t in turns))
        expected=set(reference['required_tasks'])
        for turn in fixture['followup_messages']:expected.update(turn.get('expected_tasks',[]))
        stages=set(t.get('step') for r in turns for t in r.get('trace',[]))
        stages|={'parse'} if 'facts' in stages else set();stages|={'policy'} if 'leader' in stages else set()
        urgent=bool(turns and turns[0].get('status')=='urgent_referral')
        record={'id':fixture['id'],'specialty':fixture['specialty'],'variant':fixture['variant'],
            'status':'failed' if error else 'completed','error':error,'turns':turns,
            'expected_tasks':sorted(expected),'called_tasks':sorted(called),'route_exact':called==expected,
            'urgent_match':urgent==reference['urgent_referral'],
            'missing_stages':sorted(set(reference['expected_stages'])-stages),
            'faults_injected':index.injected,'faults_not_exercised':index.remaining,
            'elapsed_seconds':time.monotonic()-started,'clinical_validation':False}
        write_json(target,record);results.append(record)
        write_json(out/'status.json',{'status':'running','completed':len(results),'total':160})
    summary={'samples':160,'development_only':True,'seed_case_groups':10,'clinical_validation':False,
        'failed':[r['id'] for r in results if r['status']=='failed'],
        'route_exact':sum(r['route_exact'] for r in results)/160,
        'urgent_mismatch':[r['id'] for r in results if not r['urgent_match']],
        'incomplete_flow':[r['id'] for r in results if r['missing_stages']],
        'faults_injected':sum(r['faults_injected'] for r in results),
        'faults_not_exercised':sum(r['faults_not_exercised'] for r in results)}
    write_json(out/'results.json',summary);write_json(out/'status.json',{'status':'complete','samples':160})
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args();main(a.label)
