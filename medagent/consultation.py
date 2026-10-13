"""Patient consultation: bounded learned policy loop implemented with LangGraph."""
import concurrent.futures
import json
import re
import time
import uuid
import contextvars
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from .common import ROOT, read_json, write_json, parse_json, digest
from .consult_contracts import (REGISTRY, FACT_PROMPT, LEADER_PROMPT, EXPERT_PROMPT,
    ANSWER_PROMPT, VISION_PROMPT, validate_facts, merge_facts, validate_action, visible_messages)
from .provider import LocalModelClient,CALL_METADATA
from .retrieval import KnowledgeIndex
from .sessions import SessionStore
from .routing import NEEDS_PROMPT, needs_view, validate_needs
from .evidence import validate_expert, evidence_report
from .triage import urgent_fact_ids

class ConsultationState(TypedDict, total=False):
    sources: dict
    facts: list
    messages: list
    images: list
    text: str
    message_id: str
    question_count: int
    asked_questions: list
    asked_fields: list
    tool_count: int
    action_count: int
    action: dict
    observations: list
    experts: dict
    knowledge: list
    trace: list
    issues: list
    response: dict
    image_reviewed: bool
    urgent: bool
    needs: dict
    retry_policy: bool
    historical_memory: list

def _json(x): return json.dumps(x,ensure_ascii=False)
MODEL_TRACE=contextvars.ContextVar('oral_model_trace',default=None)

class ConsultationAgent:
    def __init__(self, client=None, config=None, index=None):
        self.config=config or read_json(ROOT/'configs/consultation.json')
        self.client=client or LocalModelClient(self.config['models'])
        self.index=index or KnowledgeIndex(ROOT/'knowledge')
        if index is None and self.config.get('retrieval',{}).get('type')=='hybrid':
            from .dense_retrieval import DenseKnowledgeIndex
            r=self.config['retrieval']
            self.index=DenseKnowledgeIndex(ROOT/'knowledge',ROOT/r['index_path'],r['encoder_path'],r.get('device','cpu'))
        if self.config.get('retrieval',{}).get('reranker_path'):
            from .dense_retrieval import RerankedIndex
            r=self.config['retrieval'];self.index=RerankedIndex(self.index,r['reranker_path'],r.get('device','cpu'))
        g=StateGraph(ConsultationState)
        for name,fn in [('parse',self.parse),('triage',self.triage),('assess_needs',self.assess_needs),('policy',self.policy),('tools',self.tools),('answer',self.answer)]:g.add_node(name,fn)
        g.add_edge(START,'parse');g.add_edge('parse','triage')
        g.add_conditional_edges('triage',lambda s:'answer' if s.get('urgent') else 'assess_needs')
        g.add_edge('assess_needs','policy')
        g.add_conditional_edges('policy',lambda s:'policy' if s.get('retry_policy') else 'tools' if s['action']['action'] in {'dispatch','retrieve','review_image'} else 'answer')
        g.add_conditional_edges('tools',lambda s:'triage' if s['action']['action']=='review_image' else 'policy')
        g.add_edge('answer',END)
        self.graph=g.compile(name=self.config['version'])

    def call(self,role,prompt,data,images=()):
        started=time.monotonic();record={'role':role,'model':self.config['models'].get(role,{}).get('model'),'prompt_sha256':digest(prompt)}
        CALL_METADATA.set(None)
        try:
            raw=self.client.complete(role,prompt,_json(data),images);record['raw_output']=raw
            value=parse_json(raw);record['status']='parsed';return value
        except Exception as exc:
            record['status']='failed';record['error']=str(exc)[:160];raise
        finally:
            metadata=CALL_METADATA.get()
            if metadata:record.update(metadata)
            record['elapsed_seconds']=round(time.monotonic()-started,4)
            trace=MODEL_TRACE.get()
            if trace is not None:trace.append(record)

    def parse(self,s):
        sources=dict(s['sources']);issues=list(s['issues']);text=s['text']
        if s.get('images'):
            doc=self.call('vision',VISION_PROMPT,{'instruction':'转写本次上传病历'},s['images'])
            if not isinstance(doc,dict) or not isinstance(doc.get('text'),str):raise ValueError('invalid_transcription')
            sources[s['message_id']+'-image']={'text':doc['text'],'kind':'clinical_record','time':time.time()}
            issues += ['unreadable_image_region'] if doc.get('uncertain_regions') else []
        # Only this turn is extracted; older facts retain their original provenance.
        current={k:v for k,v in sources.items() if k.startswith(s['message_id'])}
        aliases={('patient' if key==s['message_id'] else 'image_record' if key.endswith('-image') else 'record'):key for key in current}
        payload=self.call('frontend',FACT_PROMPT,{'sources':{alias:current[key] for alias,key in aliases.items()},'prior_facts':s['facts'][-32:]})
        if isinstance(payload,dict) and isinstance(payload.get('facts'),list):
            for fact in payload['facts']:
                if isinstance(fact,dict):fact['source_id']=aliases.get(fact.get('source_id'),fact.get('source_id'))
        facts,rejected=validate_facts(payload,current)
        return {'sources':sources,'facts':merge_facts(s['facts'],facts),'issues':issues+rejected,
            'trace':s['trace']+[{'step':'facts','count':len(facts),'rejected':rejected}]}

    def triage(self,s):
        # Explicit positive statements only. Keyword matching alone never overrides negation.
        ids=urgent_fact_ids(s['facts'])
        return {'urgent':bool(ids),'trace':s['trace']+[{'step':'triage','urgent':bool(ids),'fact_ids':ids}]}

    def assess_needs(self,s):
        try:
            raw=self.call('needs',NEEDS_PROMPT,needs_view(s['facts'],observations=s['observations'],recent_messages=visible_messages(s['messages'])))
            needs=validate_needs(raw,s['facts'])
        except Exception as exc:
            return {'needs':{'status':'failed','candidates':[],'error':str(exc)[:160]},
                'issues':s['issues']+['needs_analysis_failed'],
                'trace':s['trace']+[{'step':'assess_needs','status':'failed','error':str(exc)[:160]}]}
        return {'needs':needs,'trace':s['trace']+[{'step':'assess_needs','status':'ok','needs':needs}]}

    def policy(self,s):
        if s['action_count']>=self.config['max_actions']:
            return {'action':{'action':'finish'},'retry_policy':False,'issues':s['issues']+['action_budget_exhausted']}
        from .consult_contracts import policy_view
        view=policy_view(s,self.config)
        try:
            action=self.call('leader',LEADER_PROMPT,view)
            validate_action(action,s,self.config)
        except Exception as exc:
            # Invalid actions are observations, never silently relabeled successful tools.
            return {'action':{'action':'finish'},'action_count':s['action_count']+1,'retry_policy':s['action_count']+1<min(2,self.config['max_actions']),
                'issues':s['issues']+['leader_action_failed'],
                'observations':s['observations']+[{'error':str(exc)[:160]}]}
        return {'action':action,'action_count':s['action_count']+1,'retry_policy':False,
            'trace':s['trace']+[{'step':'leader','action':action}]}

    def expert(self,task,s):
        spec=REGISTRY[task]
        prompt=EXPERT_PROMPT.format(specialty=spec['name'],scope=spec['scope'])
        result=self.call(spec['adapter'],prompt,{k:s[k] for k in ('facts','sources','knowledge')})
        result['reference_validation']=validate_expert(result,s)
        if 'legacy_factors' in result:
            if 'rule_assess' not in spec['tools']:raise ValueError('rule_tool_not_permitted')
            from .rules import assess
            result['rule_assessment']=assess(spec['legacy_task'],result['legacy_factors'])
        result['validation']='schema_and_reference_ids_only'
        return result

    def tools(self,s):
        a=s['action'];kind=a['action'];cost=validate_action(a,s,self.config)
        obs=list(s['observations']);experts=dict(s['experts']);knowledge=list(s['knowledge']);issues=list(s['issues'])
        updates={}
        try:
            if kind=='dispatch':
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    jobs={t:pool.submit(contextvars.copy_context().run,self.expert,t,s) for t in a['tasks']}
                    for task,job in jobs.items():
                        try:
                            experts[task]=job.result();obs.append({'tool':'expert','task':task,'status':'ok'})
                        except Exception as exc:
                            experts[task]={'error':str(exc)[:160],'status':'failed'}
                            obs.append({'tool':'expert','task':task,'status':'failed','error':str(exc)[:160]});issues.append('expert_failed:'+task)
            elif kind=='retrieve':
                hits=self.index.search(a['query'],None if a['task']=='all' else a['task'],limit=3)
                knowledge=list({x['id']:x for x in knowledge+hits}.values())
                obs.append({'tool':'knowledge_search','task':a['task'],'count':len(hits)})
            elif kind=='review_image':
                doc=self.call('vision',VISION_PROMPT,{'previous_text':s['sources'],'instruction':'局部核对，不补造'},s['images'])
                if not isinstance(doc.get('text'),str):raise ValueError('invalid_review')
                sid=s['message_id']+'-review'
                source={sid:{'text':doc['text'],'kind':'clinical_record','time':time.time()}}
                extracted=self.call('frontend',FACT_PROMPT,{'sources':source})
                facts,bad=validate_facts(extracted,source)
                updates={'sources':s['sources']|source,'facts':merge_facts(s['facts'],facts),'image_reviewed':True}
                # Earlier draft experts must be recomputed against changed clinical facts.
                experts={};issues+=bad;obs.append({'tool':'image_review','status':'ok','expert_results_invalidated':True})
        except Exception as exc:
            obs.append({'tool':kind,'status':'failed','error':str(exc)[:160]});issues.append('tool_failed:'+kind)
        return {**updates,'experts':experts,'knowledge':knowledge,'observations':obs,'issues':issues,
            'tool_count':s['tool_count']+cost,'trace':s['trace']+[{'step':kind,'cost':cost}]}

    def answer(self,s):
        issues=list(s['issues']);a=s.get('action',{'action':'refer'})
        if s.get('urgent'):
            response={'answer':'你描述了呼吸或吞咽困难等需要紧急处理的情况，请立即联系当地急救或前往急诊，避免等待线上咨询。',
                'status':'urgent_referral','citations':['https://www.nhs.uk/conditions/dental-abscess/']}
        elif a['action']=='ask':
            response={'answer':a['question'],'status':'awaiting_patient','citations':[]}
        else:
            try:
                response=self.call('answer',ANSWER_PROMPT,{k:s[k] for k in ('facts','experts','knowledge','observations')}|{'decision':a})
                if not isinstance(response,dict) or not isinstance(response.get('answer'),str) or not isinstance(response.get('citations'),list):raise ValueError('invalid_answer')
                valid={k['id'] for k in s['knowledge']}
                if any(c not in valid for c in response['citations']):raise ValueError('invalid_citation')
                response['status']='needs_review' if issues else 'answered'
            except Exception:
                response={'answer':'目前的信息不足以形成可靠的咨询分析，请补充症状持续时间、诱因和已有检查记录，或由口腔医生进一步检查。','status':'needs_review','citations':[]}
                issues.append('answer_unavailable')
        return {'response':response,'issues':issues,'trace':s['trace']+[{'step':'answer','status':response['status']}],
            'question_count':s['question_count']+int(a['action']=='ask'),
            'asked_fields':s.get('asked_fields',[])+([a['field']] if a['action']=='ask' else []),
            'asked_questions':s['asked_questions']+([a['question']] if a['action']=='ask' else [])}

class ConsultationService:
    def __init__(self,agent=None,store=None,runtime=None):
        self.agent=agent or ConsultationAgent();self.store=store or SessionStore()
        self.runtime=runtime or ROOT/'runtime/runs'

    def run(self,cid,request,rid=None,reserved=None):
        rid=rid or uuid.uuid4().hex
        old=reserved if reserved is not None else self.store.reserve(cid,rid)
        start=time.monotonic()
        calls=[];trace_token=MODEL_TRACE.set(calls)
        try:
            text=request.get('text','');images=request.get('images',[]);record=request.get('record_text','')
            if not isinstance(text,str) or not isinstance(record,str) or not isinstance(images,list) or not (text.strip() or record.strip() or images):raise ValueError('empty_or_invalid_message')
            if len(text)+len(record)>16000 or len(images)>8:raise ValueError('input_too_large')
            mid=uuid.uuid4().hex
            sources=old['sources']|{mid:{'text':text,'kind':'patient_report','time':time.time()}}
            if record:sources[mid+'-record']={'text':record,'kind':'clinical_record','time':time.time()}
            initial={**old,'historical_memory':self.store.profile(old['patient_id']),'sources':sources,'text':text,'images':images,'message_id':mid,
                'asked_fields':old.get('asked_fields',[]),
                'messages':old['messages']+[{'role':'user','content':text,'id':mid}],
                'experts':{},'knowledge':[],'observations':[],'trace':[],'issues':[],
                'tool_count':0,'action_count':0,'image_reviewed':False,'urgent':False}
            write_json(self.runtime/(rid+'.json'),{'run_id':rid,'conversation_id':cid,'status':'running'})
            result=self.agent.graph.invoke(initial,{'recursion_limit':30})
            saved={k:result[k] for k in ('sources','facts','messages','question_count','asked_questions','asked_fields')}
            saved['conversation_id']=cid
            saved['patient_id']=old['patient_id']
            saved['historical_memory']=self.store.profile(old['patient_id'])
            saved['messages'].append({'role':'assistant','content':result['response']['answer'],'run_id':rid})
            self.store.finish(cid,rid,saved)
            report={k:result[k] for k in ('facts','experts','knowledge','observations','trace','issues','tool_count','action_count')}
            report['assessment_needs']=result.get('needs',{})
            report['patient_id']=old['patient_id']
            report['structured_symptoms']=[{k:f.get(k) for k in ('id','name','value','status','location','tooth','quote','source_id')} for f in result['facts']]
            report['symptom_text']='\n'.join(f"{f['name']}：{f.get('value','')}（{f['status']}；原话：{f['quote']}）" for f in result['facts'])
            report['pre_diagnosis_document']={'kind':'auxiliary_pre_diagnosis','symptoms':report['structured_symptoms'],'specialty_documents':result['experts'],'assessment_needs':report['assessment_needs'],'unresolved':result['issues'],'confirmed_diagnosis':False}
            report['evidence']=evidence_report(result)
            report.update(result['response']);report.update(run_id=rid,conversation_id=cid,elapsed_seconds=round(time.monotonic()-start,3),graph_version=self.agent.config['version'])
            report['model_calls']=calls
            measured=[c['usage'] for c in calls if isinstance(c.get('usage'),dict)]
            report['inference_metrics']={'model_call_count':len(calls),'usage_complete':len(measured)==len(calls),
                'input_tokens':sum(u.get('prompt_tokens',0) for u in measured),
                'output_tokens':sum(u.get('completion_tokens',0) for u in measured),
                'role_seconds':{role:round(sum(c['elapsed_seconds'] for c in calls if c['role']==role),4) for role in {c['role'] for c in calls}}}
            write_json(self.runtime/(rid+'.json'),report)
            return report
        except Exception as exc:
            self.store.release(cid,rid)
            write_json(self.runtime/(rid+'.json'),{'run_id':rid,'conversation_id':cid,'status':'failed','error':str(exc)[:300],'model_calls':calls})
            raise
        finally:MODEL_TRACE.reset(trace_token)
