"""Patient consultation: bounded learned policy loop implemented with LangGraph."""
from .settings import configured_path
import concurrent.futures
import json
import re
import time
import uuid
import contextvars
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from .common import ROOT, read_json, write_json, parse_json, digest
from .consult_contracts import (REGISTRY, FACT_PROMPT, LEADER_PROMPT,
    ANSWER_PROMPT, VISION_PROMPT, validate_facts, merge_facts, validate_action, visible_messages)
from .provider import LocalModelClient,CALL_METADATA
from .retrieval import create_index
from .settings import get_settings
from .sessions import SessionStore
from .context import source_version, filter_facts
from .skills import build_skill_registry
from .routing import NEEDS_PROMPT, needs_view, validate_needs
from .evidence import evidence_report
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
    retrievals: dict
    leader_analysis: dict
    record_review: dict
    source_changed: bool
    history_summary: dict
    context_version: str
    context_revision: int
    conversation_id: str
    context_audit: dict

def _json(x): return json.dumps(x,ensure_ascii=False)
MODEL_TRACE=contextvars.ContextVar('oral_model_trace',default=None)

class TracedClient:
    """Share the role boundary services while preserving per-request model traces."""
    def __init__(self,agent):self.agent=agent
    def complete(self,role,prompt,text,images=(),response_format=None):
        return self.agent.complete(role,prompt,text,images)


class ConsultationAgent:
    def __init__(self, client=None, config=None, index=None, store=None, registry=None):
        self.config=config or get_settings().consultation()
        self.client=client or LocalModelClient(self.config['models'])
        self.index=index or create_index()
        self.store=store
        from .leader_analysis import LeaderAnalysis
        self.traced_client=TracedClient(self)
        self.registry=registry or build_skill_registry(self.traced_client,self.index,store=self.store)
        self.registry.require('information_review','evidence_select','symptom_extract',
            'evidence_ground','rule_assess','analysis_explain','knowledge_search')
        if self.store:self.registry.require('evidence_lookup')
        self.analysis=LeaderAnalysis(self.traced_client,self.index,registry=self.registry)
        g=StateGraph(ConsultationState)
        for name,fn in [('parse',self.parse),('triage',self.triage),('assess_needs',self.assess_needs),('policy',self.policy),('tools',self.tools),('answer',self.answer)]:g.add_node(name,fn)
        g.add_edge(START,'parse');g.add_edge('parse','triage')
        g.add_conditional_edges('triage',lambda s:'answer' if s.get('urgent') else 'assess_needs')
        g.add_edge('assess_needs','policy')
        g.add_conditional_edges('policy',lambda s:'policy' if s.get('retry_policy') else 'tools' if s['action']['action'] in {'dispatch','retrieve','review_image'} else 'answer')
        g.add_conditional_edges('tools',lambda s:'triage' if s['action']['action']=='review_image' or s.get('source_changed') else 'policy')
        g.add_edge('answer',END)
        self.graph=g.compile(name=self.config['version'])

    def call(self,role,prompt,data,images=()):
        return parse_json(self.complete(role,prompt,_json(data),images))

    def complete(self,role,prompt,text,images=()):
        started=time.monotonic();record={'role':role,'model':self.config['models'].get(role,{}).get('model'),'prompt_sha256':digest(prompt)}
        CALL_METADATA.set(None)
        try:
            raw=self.client.complete(role,prompt,text,images);record['raw_output']=raw
            parse_json(raw);record['status']='parsed';return raw
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
        if s.get('images') and s['message_id']+'-record' not in sources:
            doc=self.call('vision',VISION_PROMPT,{'instruction':'转写本次上传病历'},s['images'])
            if not isinstance(doc,dict) or not isinstance(doc.get('text'),str):raise ValueError('invalid_transcription')
            sources[s['message_id']+'-image']={'text':doc['text'],'kind':'clinical_record','time':time.time()}
            issues += ['unreadable_image_region'] if doc.get('uncertain_regions') else []
        # Only this turn is extracted; older facts retain their original provenance.
        current={k:v for k,v in sources.items() if k.startswith(s['message_id'])}
        aliases={('patient' if key==s['message_id'] else 'image_record' if key.endswith('-image') else 'record'):key for key in current}
        payload=self.call('frontend',FACT_PROMPT,{'sources':{alias:current[key] for alias,key in aliases.items()},'prior_facts':filter_facts(s['facts'],limit=2000)[0],
            'historical_summary':s.get('history_summary',{})})
        if isinstance(payload,dict) and isinstance(payload.get('facts'),list):
            for fact in payload['facts']:
                if isinstance(fact,dict):fact['source_id']=aliases.get(fact.get('source_id'),fact.get('source_id'))
        facts,rejected=validate_facts(payload,current)
        return {'sources':sources,'facts':merge_facts(s['facts'],facts),'issues':issues+rejected,
            'context_version':digest(source_version(s)+json.dumps(current,sort_keys=True,ensure_ascii=False)),
            'trace':s['trace']+[{'step':'facts','count':len(facts),'rejected':rejected}]}

    def triage(self,s):
        # Explicit positive statements only. Keyword matching alone never overrides negation.
        ids=urgent_fact_ids(s['facts'])
        return {'urgent':bool(ids),'trace':s['trace']+[{'step':'triage','urgent':bool(ids),'fact_ids':ids}]}

    def assess_needs(self,s):
        try:
            from .context import policy_context
            projection=policy_context(s,self.config)
            visible=projection['facts'];audit=projection['context_audit']
            raw=self.call('needs',NEEDS_PROMPT,needs_view(visible,
                recent_messages=projection['recent_messages'],historical_summary=projection['history_summary']))
            needs=validate_needs(raw,visible)
        except Exception as exc:
            return {'needs':{'status':'failed','candidates':[],'error':str(exc)[:160]},
                'issues':s['issues']+['needs_analysis_failed'],
                'trace':s['trace']+[{'step':'assess_needs','status':'failed','error':str(exc)[:160]}]}
        return {'needs':needs,'context_audit':audit,'issues':s['issues']+(['routing_context_incomplete'] if audit['incomplete'] else []),
            'trace':s['trace']+[{'step':'assess_needs','status':'ok','needs':needs,'context_audit':audit}]}

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
        from .disease_agents import DiseaseAgent
        from .routing import task_scopes
        scopes=task_scopes(s['needs'],s['facts'],[task])
        archived=self.registry.run('evidence_lookup',task=task,scope=scopes.get(task,[]),
            payload={'conversation_id':s['conversation_id']}) if self.store else {'sources':{},'facts':[]}
        sources={**archived['sources'],**s['sources']}
        facts=list({f['id']:f for f in archived['facts']+s['facts']}.values())
        output=DiseaseAgent(task,registry=self.registry).invoke_sources(sources,scopes.get(task),facts,source_version(s),archived.get('archive_audit'))
        return output['result']

    def review_record(self,s):
        current={k:v for k,v in s['sources'].items() if k.startswith(s['message_id']) and v['kind']=='clinical_record'}
        text='\n'.join(v['text'] for v in current.values())
        review=self.registry.run('information_review',text=text,images=s['images'])
        return current,review

    def apply_review(self,s,current,review):
        changed=review['text']!='\n'.join(v['text'] for v in current.values())
        updates={'record_review':review,'image_reviewed':True,'source_changed':changed}
        if changed:
            sid=s['message_id']+'-review'
            source={sid:{'text':review['text'],'kind':'clinical_record','time':time.time(),
                'replaces':list(current),'sha256':digest(review['text'])}}
            extracted=self.call('frontend',FACT_PROMPT,{'sources':source})
            facts,bad=validate_facts(extracted,source)
            if bad:raise ValueError('reviewed_facts_validation:'+','.join(bad))
            remaining=[f for f in s['facts'] if f['source_id'] not in current]
            # Superseded OCR facts are removed, never merged with their corrections.
            remaining_ids={f['id'] for f in remaining}
            remaining=[{**fact,'conflicts_with':[i for i in fact.get('conflicts_with',[]) if i in remaining_ids]} for fact in remaining]
            updates.update(sources={k:v for k,v in s['sources'].items() if k not in current}|source,
                           facts=merge_facts(remaining,facts),
                           context_version=digest(source_version(s)+digest(review['text'])))
        return updates

    def tools(self,s):
        a=s['action'];kind=a['action'];cost=validate_action(a,s,self.config)
        obs=list(s['observations']);experts=dict(s['experts']);knowledge=list(s['knowledge']);issues=list(s['issues'])
        retrievals=dict(s.get('retrievals',{}));updates={'source_changed':False}
        try:
            if kind=='dispatch':
                review_needed=bool(s.get('images')) and not s.get('image_reviewed')
                review_allowed=review_needed and s['tool_count']+cost+1<=self.config['max_tools']
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    review_job=pool.submit(contextvars.copy_context().run,self.review_record,s) if review_allowed else None
                    if review_job:cost+=1
                    jobs={t:pool.submit(contextvars.copy_context().run,self.expert,t,s) for t in a['tasks']}
                    for task,job in jobs.items():
                        try:
                            experts[task]=job.result();obs.append({'tool':'symptom_extract','task':task,'status':experts[task]['status']})
                            if experts[task]['status']=='failed':issues.append('expert_failed:'+task)
                        except Exception as exc:
                            experts[task]={'error':str(exc)[:160],'status':'failed'}
                            obs.append({'tool':'symptom_extract','task':task,'status':'failed','error':str(exc)[:160]});issues.append('expert_failed:'+task)
                    if review_job:
                        try:
                            current,review=review_job.result();updates.update(self.apply_review(s,current,review))
                            if not review['approved']:issues.append('unresolved_document_review')
                            obs.append({'tool':'image_review','status':'ok','parallel_with_extraction':True,
                                'source_changed':updates['source_changed'],'details':review})
                        except Exception as exc:
                            updates.update(image_reviewed=True,record_review={'approved':False,'error':str(exc)[:160]})
                            if review_job and 'review' in locals() and review.get('text')!='\n'.join(v['text'] for v in current.values()):
                                experts={};knowledge=[];retrievals={}
                            issues.append('image_review_failed');obs.append({'tool':'image_review','status':'failed','error':str(exc)[:160]})
                if review_needed and not review_allowed:issues.append('image_review_budget_exhausted')
                if updates.get('source_changed'):
                    experts={};knowledge=[];retrievals={}
                    fresh={**s,**updates,'experts':{},'knowledge':[],'issues':issues,'observations':obs}
                    need_update=self.assess_needs(fresh);updates['needs']=need_update['needs']
                    issues=need_update.get('issues',issues)
                    from .routing import selected_tasks
                    rerun=[t for t in a['tasks'] if t in selected_tasks(updates['needs'])]
                    fresh.update(needs=updates['needs'])
                    for task in rerun:
                        if s['tool_count']+cost>=self.config['max_tools']:
                            issues.append('review_reflow_budget_exhausted');break
                        cost+=1
                        try:experts[task]=self.expert(task,fresh)
                        except Exception as exc:experts[task]={'status':'failed','error':str(exc)[:160]}
                    obs.append({'tool':'review_reflow','invalidated_previous_experts':True,'rerun_tasks':list(experts)})
            elif kind=='retrieve':
                targets=[t for t,r in experts.items() if r.get('status')!='failed'] if a['task']=='all' else [a['task']]
                for task in targets:
                    value=self.analysis.retrieve(task,experts[task])
                    retrievals[task]={'input_version':experts[task]['input_version'],**value}
                knowledge=list({x['id']:x for v in retrievals.values() for x in v['knowledge']}.values())
                obs.append({'tool':'knowledge_search','owner':'leader','tasks':targets,'query_basis':'structured_symptoms','count':len(knowledge)})
            elif kind=='review_image':
                current,review=self.review_record(s);updates.update(self.apply_review(s,current,review))
                if not review['approved']:issues.append('unresolved_document_review')
                if updates['source_changed']:experts={};knowledge=[];retrievals={}
                obs.append({'tool':'image_review','status':'ok','expert_results_invalidated':updates['source_changed'],'details':review})
        except Exception as exc:
            if kind=='review_image' and 'review' in locals() and review.get('text')!='\n'.join(v['text'] for v in current.values()):
                experts={};knowledge=[];retrievals={}
                updates.update(image_reviewed=True,record_review={**review,'approved':False,'error':str(exc)[:160]})
            obs.append({'tool':kind,'status':'failed','error':str(exc)[:160]});issues.append('tool_failed:'+kind)
        return {**updates,'experts':experts,'knowledge':knowledge,'retrievals':retrievals,'observations':obs,'issues':issues,
            'tool_count':s['tool_count']+cost,'trace':s['trace']+[{'step':kind,'cost':cost}]}

    def answer(self,s):
        issues=list(s['issues']);a=s.get('action',{'action':'refer'})
        analyses={};knowledge=[];tool_count=s['tool_count']
        observations=list(s['observations'])
        if s.get('urgent'):
            response={'answer':'你描述了呼吸或吞咽困难等需要紧急处理的情况，请立即联系当地急救或前往急诊，避免等待线上咨询。',
                'status':'urgent_referral','citations':[get_settings().data['references']['urgent_referral']]}
        elif a['action']=='ask':
            response={'answer':a['question'],'status':'awaiting_patient','citations':[]}
        else:
            try:
                from .consult_contracts import finish_issue
                pending=finish_issue(s,self.config)
                if pending:issues.append(pending)
                from .routing import selected_tasks
                for task,extraction in s['experts'].items():
                    if task not in selected_tasks(s.get('needs',{'candidates':[]})):continue
                    if extraction.get('input_version')!=source_version(s):
                        issues.append('stale_specialty_result:'+task);continue
                    cached=s.get('retrievals',{}).get(task)
                    if cached and cached['input_version']!=extraction.get('input_version'):cached=None
                    allowed=tool_count<self.config['max_tools']
                    if not cached and extraction.get('status')!='failed' and allowed:tool_count+=1
                    analyses[task]=self.analysis.organize(task,extraction,retrieved=cached,retrieve_allowed=allowed)
                    if analyses[task].get('issues'):issues.append('leader_analysis_incomplete:'+task)
                    knowledge.extend(analyses[task].get('knowledge',[]))
                knowledge=list({hit['id']:hit for hit in knowledge}.values())
                observations.append({'tool':'leader_analysis','tasks':list(analyses),'query_basis':'structured_symptoms'})
                from .context import answer_context
                packet=answer_context(s,analyses,knowledge,a)
                if packet['context_audit']['incomplete']:issues.append('answer_context_incomplete')
                response=self.call('answer',ANSWER_PROMPT+'\n若context_audit.incomplete为true，明确说明摘要未覆盖全部证据，不能据此排除疾病或确认低风险。',packet)
                if not isinstance(response,dict) or not isinstance(response.get('answer'),str) or not isinstance(response.get('citations'),list):raise ValueError('invalid_answer')
                valid={k['id'] for k in knowledge}
                if any(c not in valid for c in response['citations']):raise ValueError('invalid_citation')
                response['status']='needs_review' if issues else 'answered'
            except Exception:
                response={'answer':'目前的信息不足以形成可靠的咨询分析，请补充症状持续时间、诱因和已有检查记录，或由口腔医生进一步检查。','status':'needs_review','citations':[]}
                issues.append('answer_unavailable')
        return {'response':response,'issues':issues,'leader_analysis':analyses,'knowledge':knowledge,
            'observations':observations,'tool_count':tool_count,'trace':s['trace']+[{'step':'answer','status':response['status']}],
            'question_count':s['question_count']+int(a['action']=='ask'),
            'asked_fields':s.get('asked_fields',[])+([a['field']] if a['action']=='ask' else []),
            'asked_questions':s['asked_questions']+([a['question']] if a['action']=='ask' else [])}

class ConsultationService:
    def __init__(self,agent=None,store=None,runtime=None):
        self.store=store or SessionStore();self.agent=agent or ConsultationAgent(store=self.store)
        self.runtime=runtime or configured_path('conversation_runs_dir')

    def run(self,cid,request,rid=None,reserved=None):
        rid=rid or uuid.uuid4().hex
        old=reserved if reserved is not None else self.store.reserve(cid,rid)
        start=time.monotonic()
        calls=[];trace_token=MODEL_TRACE.set(calls)
        try:
            text=request.get('text','');images=request.get('images',[]);record=request.get('record_text','')
            if not isinstance(text,str) or not isinstance(record,str) or not isinstance(images,list) or not (text.strip() or record.strip() or images):raise ValueError('empty_or_invalid_message')
            limits=get_settings().data['limits']
            if len(text)+len(record)>limits['chat_chars'] or len(images)>limits['images']:raise ValueError('input_too_large')
            mid=uuid.uuid4().hex
            sources=old['sources']|{mid:{'text':text,'kind':'patient_report','time':time.time()}}
            if record:sources[mid+'-record']={'text':record,'kind':'clinical_record','time':time.time()}
            initial={**old,'context_version':digest(source_version(old)+json.dumps({k:v for k,v in sources.items() if k.startswith(mid)},sort_keys=True,ensure_ascii=False)),
                'historical_memory':self.store.profile(old['patient_id']),'sources':sources,'text':text,'images':images,'message_id':mid,
                'asked_fields':old.get('asked_fields',[]),
                'messages':old['messages']+[{'role':'user','content':text,'id':mid}],
                'experts':{},'knowledge':[],'observations':[],'trace':[],'issues':[],
                'tool_count':0,'action_count':0,'image_reviewed':False,'urgent':False,
                'retrievals':{},'leader_analysis':{},'record_review':{},'source_changed':False}
            # Explicit FDI mentions can recall old evidence before routing, without treating a summary as a source.
            mentioned=sorted({tooth for pair in re.findall(r'(?<!\d)([1-4][1-8])\s*牙|牙位\s*([1-4][1-8])(?!\d)',text+'\n'+record) for tooth in pair if tooth})
            if mentioned and getattr(self.agent,'registry',None) and getattr(self.agent,'store',None):
                recalled=self.agent.registry.run('evidence_lookup',scope=mentioned,payload={'conversation_id':cid})
                initial['sources']={**recalled['sources'],**initial['sources']}
                initial['facts']=list({f['id']:f for f in recalled['facts']+old['facts']}.values())
                audit=recalled.get('archive_audit',{})
                if audit.get('facts_truncated') or audit.get('sources_truncated'):
                    initial['issues'].append('history_recall_incomplete')
            write_json(self.runtime/(rid+'.json'),{'run_id':rid,'conversation_id':cid,'status':'running'})
            result=self.agent.graph.invoke(initial,{'recursion_limit':30})
            saved={k:result[k] for k in ('sources','facts','messages','question_count','asked_questions','asked_fields')}
            for key in ('history_summary','context_version','context_revision'):
                if key in result:saved[key]=result[key]
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
            report['pre_diagnosis_document']={'kind':'auxiliary_pre_diagnosis','symptoms':report['structured_symptoms'],'specialty_extractions':result['experts'],'specialty_documents':{t:r.get('analysis_explanation',{}) for t,r in result['experts'].items()},'leader_analysis':result.get('leader_analysis',{}),
                'closest_disease_documents':{t:r.get('closest_disease_documents',[]) for t,r in result.get('leader_analysis',{}).items()},'assessment_needs':report['assessment_needs'],'unresolved':result['issues'],'confirmed_diagnosis':False}
            report['specialty_results']=result['experts']
            report['leader_analysis']=result.get('leader_analysis',{})
            report['closest_disease_documents']=report['pre_diagnosis_document']['closest_disease_documents']
            report['record_review']=result.get('record_review',{})
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
