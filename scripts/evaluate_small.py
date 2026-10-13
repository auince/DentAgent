"""Complete fixed test evaluation; failed outputs stay in every denominator."""
import argparse
import base64
import json
import math
import sys
import time
import concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json,parse_json,digest
from medagent.evaluation import evaluate,grades
from medagent.datasets import prepare
from medagent.provider import LocalModelClient
from medagent.training_rewards import fact_atoms

def legacy(endpoint,model,label,policy='explicit'):
    from medagent.extraction_policy import TrainingAlignedExtractSkill,ExplicitFieldExtractSkill
    from ablation_worker import capture
    cfg=read_json(ROOT/'configs/experiments.json');out=ROOT/'runtime/small_model/evaluation'/label
    client=LocalModelClient({'model':{'endpoint':endpoint,'model':model,'max_tokens':4096,'timeout':180,'retries':0}})
    results={}
    for task,spec in cfg['tasks'].items():
        rows,audit=prepare(task,spec)
        if len(rows)!={'root_canal':101,'crack':20}[task]:raise ValueError('full_test_size_changed')
        skill=ExplicitFieldExtractSkill if policy=='explicit' else TrainingAlignedExtractSkill
        jobs=[capture(lambda c:skill(c).run(task,row['text'])) for row in rows]
        protocol={'audit':audit,'requests_sha256':digest(json.dumps(jobs,sort_keys=True,ensure_ascii=False)),'policy':policy,'model':model,'endpoint':endpoint,'decoding':'greedy_non_thinking_4096'}
        manifest=out/(task+'_manifest.json')
        if manifest.exists() and read_json(manifest)!=protocol:raise ValueError('evaluation_protocol_changed')
        write_json(manifest,protocol)
        def predict(pair):
            i,row=pair
            path=out/task/(f'{i:04}.json')
            if path.exists():return read_json(path)
            started=time.monotonic();item=dict(row)
            try:item['prediction']=client.complete('model',jobs[i]['system'],jobs[i]['text']);parse_json(item['prediction'])
            except Exception as exc:item['prediction']=item.get('prediction','');item['error']=str(exc)[:200]
            item['elapsed_seconds']=time.monotonic()-started
            write_json(path,item);return item
        predictions=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for item in pool.map(predict,enumerate(rows)):
                predictions.append(item)
                write_json(out/'status.json',{'status':'running','task':task,'completed':len(predictions),'total':len(rows)})
        results[task]=evaluate(task,predictions,strict_schema=False)
        results[task]['rule_grade']={'correct':0,'total':0}
        for row in predictions:
            gold=grades(task,row['gold'],True)
            try:pred=grades(task,row['prediction'],True)
            except Exception:pred={}
            union=set(gold)|set(pred)
            results[task]['rule_grade']['total']+=len(union)
            results[task]['rule_grade']['correct']+=sum(t in pred and t in gold and pred[t]==gold[t] for t in union)
        r=results[task]['rule_grade'];r['accuracy']=r['correct']/max(1,r['total'])
    write_json(out/'results.json',results);write_json(out/'status.json',{'status':'complete','samples':121})
    return results

def consultation(label):
    from medagent.consultation import ConsultationService,ConsultationAgent
    from medagent.sessions import SessionStore
    from medagent.retrieval import KnowledgeIndex
    cases=read_json(ROOT/'runtime/small_modelLOCAL_RESOURCE')
    if len(cases)!=1000:raise ValueError('complete_1000_case_manifest_required')
    out=ROOT/'runtime/small_model/evaluation'/label
    protocol={'case_sha256':digest((ROOT/'runtime/small_modelLOCAL_RESOURCE').read_bytes()),
        'config':read_json(ROOT/'configs/consultation.json'),'samples':1000}
    if (out/'manifest.json').exists() and read_json(out/'manifest.json')!=protocol:raise ValueError('consultation_evaluation_protocol_changed')
    write_json(out/'manifest.json',protocol)
    class FaultIndex:
        def __init__(self,base,faults):self.base=base;self.remaining=sum(f['times'] for f in faults if f['tool']=='knowledge_search');self.injected=0
        def search(self,*args,**kwargs):
            if self.remaining:
                self.remaining-=1;self.injected+=1;raise TimeoutError('simulated_knowledge_search_timeout')
            return self.base.search(*args,**kwargs)
    base_index=KnowledgeIndex(ROOT/'knowledge')
    tp=fp=fn=route_hits=route_total=exact=structured=schema=0;latencies=[];errors=[];urgent_misses=[]
    for case in cases:
        path=out/'cases'/(case['id']+'.json')
        if path.exists():report=read_json(path)
        else:
            fixture=case.get('scenario',{})
            faults=FaultIndex(base_index,fixture.get('fault_injections',[]))
            service=ConsultationService(agent=ConsultationAgent(index=faults),store=SessionStore(out/'sessions.sqlite'),runtime=out/'runs')
            cid=service.store.create()['conversation_id']
            request=dict(fixture.get('initial_request',{'text':case['patient_text'],'image_files':case.get('images',[])}))
            request['images']=['data:image/jpeg;base64,'+base64.b64encode((ROOT/p).read_bytes()).decode() for p in request.pop('image_files',[])]
            started=time.monotonic();turns=[]
            try:
                report=service.run(cid,request)
                initial_facts=report.get('facts',[]);turns=[report]
                for _ in range(3):
                    if report.get('status')!='awaiting_patient':break
                    action=next((e['action'] for e in reversed(report['trace']) if e.get('action',{}).get('action')=='ask'),{})
                    answer=case.get('patient_answers',{}).get(action.get('field'))
                    text=answer.get('text','不知道') if isinstance(answer,dict) else (answer or '不知道')
                    report=service.run(cid,{'text':text});turns.append(report)
                report={**report,'initial_facts':initial_facts,'turns':turns}
            except Exception as exc:report={'status':'failed','error':str(exc)[:200],'facts':[],'experts':{},'turns':turns}
            report.update(scenario_elapsed_seconds=time.monotonic()-started,faults_injected=faults.injected,
                faults_not_exercised=faults.remaining,modality=fixture.get('modality','unknown'))
            write_json(path,report)
            write_json(out/'status.json',{'status':'running','completed':len(list((out/'cases').glob('*.json'))),'total':1000,'last_case':case['id']})
        p={json.dumps(a,ensure_ascii=False) for a in fact_atoms(report.get('initial_facts',report.get('facts',[])))};g={json.dumps(a,ensure_ascii=False) for a in fact_atoms(case['initial_facts'])}
        tp+=len(p&g);fp+=len(p-g);fn+=len(g-p)
        chosen=set().union(*(set(t.get('experts',{})) for t in report.get('turns',[report])));gold=set(case['required_tasks']);route_hits+=len(chosen&gold);route_total+=len(gold);exact+=chosen==gold
        structured+=report.get('status')!='failed'
        schema+=report.get('status')!='failed' and not any(str(issue).startswith(('leader_action_failed','answer_unavailable','expert_failed','invalid_','unlocated_')) for issue in report.get('issues',[]))
        if case.get('category')=='urgent' and report.get('turns',[report])[0].get('status')!='urgent_referral':urgent_misses.append(case['id'])
        if report.get('status')=='failed':errors.append(case['id'])
        if not case.get('images'):
            latencies += [t['elapsed_seconds'] for t in report.get('turns',[report]) if 'elapsed_seconds' in t]
    latencies.sort()
    result={'samples':1000,'failed':errors,'symptom_f1':2*tp/max(1,2*tp+fp+fn),'route_recall':route_hits/max(1,route_total),
        'route_exact':exact/1000,'execution_success_rate':structured/1000,'workflow_schema_success_rate':schema/1000,'urgent_misses':urgent_misses,'text_latency_sample_count':len(latencies),
        'text_p95_seconds':latencies[math.ceil(.95*len(latencies))-1] if latencies else None,
        'clinical_validation':False,'reference_type':'synthetic_unreviewed'}
    records=[read_json(p) for p in sorted((out/'cases').glob('*.json'))]
    result['modality_counts']={m:sum(r.get('modality')==m for r in records) for m in {r.get('modality') for r in records}}
    result['faults_injected']=sum(r.get('faults_injected',0) for r in records)
    result['faults_not_exercised']=sum(r.get('faults_not_exercised',0) for r in records)
    write_json(out/'results.json',result);write_json(out/'status.json',{'status':'complete','samples':1000});return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('suite',choices=['legacy','consultation']);p.add_argument('--endpoint',default='http://deployment.invalid');p.add_argument('--model',default='oral-baseline');p.add_argument('--label',required=True);p.add_argument('--policy',choices=['explicit','training_aligned'],default='explicit');a=p.parse_args()
    print(json.dumps(legacy(a.endpoint,a.model,a.label,a.policy) if a.suite=='legacy' else consultation(a.label),ensure_ascii=False))
