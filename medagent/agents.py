"""LangGraph orchestration: a leader graph and two bounded specialty subgraphs."""
import copy
import json
import operator
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass,field
from typing import Annotated,TypedDict

from langgraph.graph import START,END,StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from .common import ROOT,digest,parse_json,write_json,read_json
from .provider import LocalModelClient
from .consult_contracts import FACT_PROMPT, validate_facts, REGISTRY
from .routing import NEEDS_PROMPT, ROUTE_PROMPT, needs_view, validate_needs, selected_tasks, task_scopes
from .retrieval import KnowledgeIndex
from .skills import BOUNDARY,DocumentParseSkill,ReviewSkill,SymptomExtractSkill,KnowledgeSearchSkill


def merge_tasks(left,right):
    """Parallel branches own distinct task keys; copies avoid in-place graph mutation."""
    return {**left,**right}


def event(step,status,**kw):
    return {'time':time.time(),'step':step,'status':status,**kw}


def route_tasks(plan):
    """Accept exact task names and explicit boolean task objects, never infer from prose."""
    if not isinstance(plan,dict) or not isinstance(plan.get('tasks'),list):
        raise ValueError('invalid_route_schema')
    allowed={'root_canal','crack'};tasks=[];adapted=False
    for item in plan['tasks']:
        if isinstance(item,str) and item in allowed:tasks.append(item)
        elif isinstance(item,dict) and (set(item)&allowed):
            if set(item)-allowed-{'reason','evidence'}:raise ValueError('invalid_route_task_object')
            for task in set(item)&allowed:
                if type(item[task]) is not bool:raise ValueError('invalid_route_task_boolean')
                if item[task]:tasks.append(task)
            adapted=True
        else:raise ValueError('invalid_route_task_name')
    if not tasks:raise ValueError('unsupported_or_unclear_case')
    return sorted(set(tasks)),adapted


@dataclass
class SharedContext:
    """Compatibility adapter for callers; graph state owns execution data."""
    run_id:str
    events:list=field(default_factory=list)
    lock:object=field(default_factory=threading.Lock)
    def log(self,step,status,**kw):
        with self.lock:self.events.append(event(step,status,**kw))


class SpecialtyState(TypedDict,total=False):
    task:str
    text:str
    images:list
    scope:list
    attempt:int
    error:str
    fatal:bool
    result:dict
    trace:Annotated[list,operator.add]


class SpecialtyAgent:
    def __init__(self,task,skills):
        self.task,self.skills=task,skills
        graph=StateGraph(SpecialtyState)
        for name,node in [('extract',self._extract),('retrieve',self._retrieve),('ground',self._ground),('explain',self._explain),('finish',self._finish)]:graph.add_node(name,node)
        graph.add_edge(START,'extract')
        graph.add_conditional_edges('extract',self._after_extract,{'retry':'extract','success':'retrieve','failed':'finish'})
        graph.add_edge('retrieve','ground');graph.add_edge('ground','explain');graph.add_edge('explain','finish');graph.add_edge('finish',END)
        self.graph=graph.compile(checkpointer=False,name='specialty_'+task)

    def _extract(self,state):
        attempt=state.get('attempt',0)+1
        try:
            kwargs={'scope':state['scope']} if state.get('scope') else {}
            result=self.skills['symptom_extract'].run(self.task,state['text'],state.get('error') or None,**kwargs)
            return {'attempt':attempt,'error':'','fatal':False,'result':result,'trace':[event(self.task,'extracted',attempt=attempt)]}
        except ValueError as exc:
            return {'attempt':attempt,'error':str(exc),'fatal':False,'trace':[event(self.task,'schema_retry',attempt=attempt,error=str(exc))]}
        except Exception as exc:
            return {'attempt':attempt,'error':str(exc)[:400],'fatal':True,'trace':[event(self.task,'failed',error=str(exc)[:400])]}

    def _after_extract(self,state):
        if not state.get('error'):return 'success'
        return 'retry' if not state.get('fatal') and state['attempt']<2 else 'failed'

    def _retrieve(self,state):
        result=copy.deepcopy(state['result'])
        query='\n'.join(str(x) for x in result['assessments'])+'\n'+state['text'][:2000]
        try:result['knowledge']=self.skills['knowledge_search'].run(self.task,query)
        except Exception as exc:result['knowledge']=[];result['issues'].append({'code':'retrieval_failed','detail':str(exc)[:300]})
        if not result['knowledge']:result['issues'].append({'code':'knowledge_not_found'})
        return {'result':result,'trace':[event(self.task+'.retrieve','done',count=len(result['knowledge']))]}

    def _ground(self,state):
        result=copy.deepcopy(state['result'])
        try:
            result['evidence'],issues=self.skills['symptom_extract'].ground(self.task,state['text'],result['factors'],state['images'])
            result['issues']+=issues
        except Exception as exc:
            result['evidence']=[];result['issues'].append({'code':'evidence_failed','detail':str(exc)[:300]})
        return {'result':result,'trace':[event(self.task+'.ground','done',evidence_count=len(result['evidence']))]}

    def _explain(self,state):
        result=copy.deepcopy(state['result'])
        try:result['explanation']=self.skills['symptom_extract'].explain(self.task,state['text'],result,state['images'])
        except Exception as exc:result['issues'].append({'code':'explanation_validation_failed','detail':str(exc)[:400]})
        return {'result':result,'trace':[event(self.task+'.explain','done')]}

    def _finish(self,state):
        if state.get('error'):
            return {'result':{'status':'failed','error':('schema_repair_exhausted:' if not state.get('fatal') else '')+self.task+':'+state['error']},'trace':[event(self.task,'failed')]}
        result=copy.deepcopy(state['result']);result['status']='needs_review' if result['issues'] else 'assessed'
        return {'result':result,'trace':[event(self.task,'done',issue_count=len(result['issues']))]}

    def invoke(self,text,images=(),scope=None):
        return self.graph.invoke({'task':self.task,'text':text,'images':list(images),'scope':scope or [],'attempt':0,'error':'','trace':[event(self.task,'running',input_sha256=digest(text),scope=scope or [])]},config={'recursion_limit':16})

    def run(self,text,ctx,images=()):
        state=self.invoke(text,images)
        with ctx.lock:ctx.events.extend(state['trace'])
        return state['result']


class LeaderState(TypedDict,total=False):
    run_id:str
    request:dict
    document:dict
    tasks:list
    facts:list
    needs:dict
    scopes:dict
    review:dict
    changed:bool
    drafts:Annotated[dict,merge_tasks]
    finals:Annotated[dict,merge_tasks]
    issues:Annotated[list,operator.add]
    trace:Annotated[list,operator.add]
    report:dict


class LeaderAgent:
    def __init__(self,client=None,runtime=None,checkpointer=None,review_policy=None,extraction_policy=None):
        self.client=client or LocalModelClient();self.runtime=runtime or ROOT/'runtime/runs'
        policy=read_json(ROOT/'configs/agent.json')
        review_policy=review_policy or policy['review_policy']
        extraction_policy=extraction_policy or policy['extraction_policy']
        self.graph_version=policy['graph_version']
        review=ReviewSkill(self.client)
        if review_policy=='conservative':
            from .review_policy import ConservativeReviewSkill
            review=ConservativeReviewSkill(self.client)
        elif review_policy!='original':raise ValueError('invalid_review_policy')
        extraction=SymptomExtractSkill(self.client)
        if extraction_policy=='explicit_fields':
            from .extraction_policy import ExplicitFieldExtractSkill
            extraction=ExplicitFieldExtractSkill(self.client)
        elif extraction_policy=='training_aligned':
            from .extraction_policy import TrainingAlignedExtractSkill
            extraction=TrainingAlignedExtractSkill(self.client)
        elif extraction_policy!='original':raise ValueError('invalid_extraction_policy')
        self.policies={'review':review_policy,'extraction':extraction_policy}
        self.skills={s.name:s for s in [DocumentParseSkill(self.client),review,extraction,KnowledgeSearchSkill(KnowledgeIndex(ROOT/'knowledge'))]}
        self.specialists={t:SpecialtyAgent(t,self.skills) for t in ('root_canal','crack')}
        self._connection=None
        if checkpointer is None:
            self.runtime.mkdir(parents=True,exist_ok=True,mode=0o700)
            path=self.runtime/'graph_checkpoints.sqlite3'
            self._connection=sqlite3.connect(str(path),check_same_thread=False)
            path.chmod(0o600)
            checkpointer=SqliteSaver(self._connection)
        builder=StateGraph(LeaderState)
        builder.add_node('document_parse',self._parse)
        builder.add_node('identify_symptoms',self._identify_symptoms)
        builder.add_node('assess_needs',self._assess_needs)
        builder.add_node('route',self._route)
        builder.add_node('information_review',self._review)
        for task in ('root_canal','crack'):
            builder.add_node('draft_'+task,self._specialty_node(task,False))
            builder.add_node('reflow_'+task,self._specialty_node(task,True))
        builder.add_node('reconcile',self._reconcile)
        builder.add_node('reflow',lambda state:{'trace':[event('leader','reflow',reason='review_changed_source')]})
        builder.add_node('aggregate',self._aggregate)
        builder.add_edge(START,'document_parse');builder.add_edge('document_parse','identify_symptoms')
        builder.add_edge('identify_symptoms','assess_needs');builder.add_edge('assess_needs','route')
        branches=['information_review','draft_root_canal','draft_crack']
        for node in branches:builder.add_edge('route',node)
        builder.add_edge(branches,'reconcile')
        builder.add_conditional_edges('reconcile',lambda state:'reflow' if state['changed'] else 'aggregate',{'reflow':'reflow','aggregate':'aggregate'})
        for task in ('root_canal','crack'):builder.add_edge('reflow','reflow_'+task)
        builder.add_edge(['reflow_root_canal','reflow_crack'],'aggregate');builder.add_edge('aggregate',END)
        self.graph=builder.compile(checkpointer=checkpointer,name='dental_leader')

    def close(self):
        if self._connection is not None:self._connection.close();self._connection=None

    def _routing_facts(self,text):
        # Chunk rather than silently routing from only the first 4K characters.
        facts=[]
        for start in range(0,len(text),3000):
            sid='record:'+str(start)
            source={sid:{'text':text[start:start+3400],'kind':'clinical_record'}}
            raw=parse_json(self.client.complete('review',FACT_PROMPT,json.dumps({'sources':source},ensure_ascii=False)))
            accepted,issues=validate_facts(raw,source)
            if issues:raise ValueError('routing_fact_validation:'+','.join(issues))
            facts.extend(accepted)
        return list({(f['name'],f.get('tooth'),f['quote'],f['status']):f for f in facts}.values())

    def _routing_needs(self,instruction,facts):
        registry={k:REGISTRY[k] for k in self.specialists}
        raw=parse_json(self.client.complete('review',NEEDS_PROMPT,
            json.dumps(needs_view(facts,registry,instruction=instruction),ensure_ascii=False)))
        return validate_needs(raw,facts,registry)

    def _routing_decision(self,instruction,facts,needs,ctx):
        plan=parse_json(self.client.complete('review',ROUTE_PROMPT,
            json.dumps({'instruction':instruction,'facts':facts,'needs':needs},ensure_ascii=False)))
        if isinstance(plan,dict) and plan.get('tasks')==[]:
            tasks=[];adapted=False
        else:tasks,adapted=route_tasks(plan)
        if not set(tasks)<=selected_tasks(needs):raise ValueError('route_outside_supported_needs')
        scopes=task_scopes(needs,facts,tasks)
        plan['task_teeth']={k:[t.removesuffix('牙') for t in v] for k,v in scopes.items()}
        ctx.log('leader.intent','recognized',source='staged_fact_need_route',tasks=tasks,
            reason=str(plan.get('reason',''))[:1000],evidence=plan.get('evidence',[]),
            task_schema_adapted=adapted,decision=plan)
        return tasks

    def plan(self,instruction,text,ctx):
        # Compatibility entry point uses the same three stages as the main graph.
        facts=self._routing_facts(text)
        ctx.log('leader.symptoms','identified',facts=facts)
        needs=self._routing_needs(instruction,facts)
        ctx.log('leader.needs','analysed',needs=needs)
        return self._routing_decision(instruction,facts,needs,ctx)

    def _identify_symptoms(self,state):
        if state['request'].get('tasks') is not None:return {'facts':[]}
        facts=self._routing_facts(state['document']['text'])
        return {'facts':facts,'trace':[event('leader.symptoms','identified',facts=facts)]}

    def _assess_needs(self,state):
        if state['request'].get('tasks') is not None:return {'needs':{'candidates':[]}}
        needs=self._routing_needs(state['request'].get('instruction','综合评估'),state['facts'])
        return {'needs':needs,'trace':[event('leader.needs','analysed',needs=needs)]}

    def _parse(self,state):
        request=state['request'];text=request.get('text','');images=request.get('images',[])
        if not isinstance(text,str) or len(text)>100000 or not isinstance(images,list) or len(images)>8:raise ValueError('invalid_document_size_or_type')
        doc=self.skills['document_parse'].run(text,images)
        return {'document':{'text':doc['text'],'method':doc['method'],'sha256':digest(doc['text'])},'issues':doc['issues'],
            'trace':[event('document_parse','done',sha256=digest(doc['text']))]}

    def _route(self,state):
        tasks=state['request'].get('tasks');ctx=SharedContext(state['run_id'])
        if tasks is None:tasks=self._routing_decision(state['request'].get('instruction','综合评估'),state['facts'],state['needs'],ctx)
        if not isinstance(tasks,list) or not all(t in self.specialists for t in tasks):raise ValueError('invalid_tasks')
        tasks=list(dict.fromkeys(tasks))
        scopes=self._scopes(ctx,state['document']['text'],tasks)
        return {'tasks':tasks,'scopes':scopes,'trace':ctx.events+[event('leader','routed',tasks=tasks,scopes=scopes)]}

    def _scopes(self,ctx,text,tasks):
        decisions=[e.get('decision',{}) for e in ctx.events if e['step']=='leader.intent']
        raw=decisions[-1].get('task_teeth',{}) if decisions else {}
        if not isinstance(raw,dict):return {}
        scopes={}
        for task in tasks:
            values=raw.get(task,[])
            if not isinstance(values,list):continue
            valid=[]
            for value in values:
                if not isinstance(value,(str,int)):continue
                number=str(value).removesuffix('牙').strip()
                if re.fullmatch(r'[1-4][1-8]',number) and re.search(r'(?<!\d)'+number+r'(?!\d)',text):valid.append(number+'牙')
            if valid:scopes[task]=sorted(set(valid))
        return scopes

    def _review(self,state):
        try:review=self.skills['information_review'].run(state['document']['text'],state['request'].get('images',[]))
        except Exception as exc:review={'text':state['document']['text'],'approved':False,'issues':[{'code':'review_failed','detail':str(exc)[:400]}]}
        return {'review':review,'issues':review['issues'],'trace':[event('information_review','done',approved=review['approved'],changed=review['text']!=state['document']['text'])]}

    def _specialty_node(self,task,reflow):
        def run(state):
            if task not in state['tasks']:return {}
            text=state['review']['text'] if reflow else state['document']['text']
            try:
                output=self.specialists[task].invoke(text,state['request'].get('images',[]),state.get('scopes',{}).get(task));result=output['result'];trace=output['trace']
            except Exception as exc:result={'status':'failed','error':str(exc)[:400]};trace=[event(task,'failed',error=str(exc)[:400])]
            return {'finals' if reflow else 'drafts':{task:result},'trace':trace}
        return run

    def _reconcile(self,state):
        changed=state['review']['text']!=state['document']['text']
        if changed and state['request'].get('tasks') is None:
            ctx=SharedContext(state['run_id'])
            try:
                tasks=self.plan(state['request'].get('instruction','综合评估'),state['review']['text'],ctx)
                scopes=self._scopes(ctx,state['review']['text'],tasks)
                facts=next((e['facts'] for e in ctx.events if e['step']=='leader.symptoms'),[])
                needs=next((e['needs'] for e in ctx.events if e['step']=='leader.needs'),{'candidates':[]})
                return {'changed':True,'tasks':tasks,'scopes':scopes,'facts':facts,'needs':needs,'trace':ctx.events+[event('leader','rerouted',tasks=tasks,scopes=scopes)]}
            except ValueError as exc:
                # Keep the draft route visible, but never silently approve uncertain routing.
                return {'changed':True,'issues':[{'code':'reviewed_route_uncertain','detail':str(exc)}],
                        'trace':ctx.events+[event('leader','reroute_failed')]}
        return {'changed':changed}

    def _aggregate(self,state):
        # An altered source invalidates every draft; no stale factors/evidence are reused.
        results=copy.deepcopy(state['finals'] if state['changed'] else state['drafts'])
        for task,result in results.items():
            if result['status']!='failed' and not state['review']['approved']:
                result['status']='needs_review';result['issues'].append({'code':'unresolved_document_review'})
        statuses=[x['status'] for x in results.values()]
        status='needs_review' if not statuses else 'failed' if all(s=='failed' for s in statuses) else 'needs_review' if state.get('issues') or not state['review']['approved'] or any(s!='assessed' for s in statuses) else 'completed'
        return {'report':{'run_id':state['run_id'],'status':status,'schema_version':2,'orchestrator':'langgraph','graph_version':self.graph_version,'policies':self.policies,
            'routing_facts':state.get('facts',[]),'assessment_needs':state.get('needs',{}),
            'document':state['document'],'tasks':state['tasks'],'review':state['review'],'reviewed_sha256':digest(state['review']['text']),
            'results':results,'issues':state.get('issues',[]),
            'summary':[{'task':task,'status':r['status'],'assessments':r.get('assessments',[])} for task,r in results.items()]},
            'trace':[event('leader','aggregated',status_result=status)]}

    def run(self,request,run_id=None):
        start=time.monotonic();run_id=run_id or uuid.uuid4().hex
        if not re.fullmatch(r'[a-f0-9]{32}',run_id):raise ValueError('invalid_run_id')
        config={'configurable':{'thread_id':run_id},'recursion_limit':32,'max_concurrency':3}
        # Reject duplicate execution IDs before any write to an existing report.
        if (self.runtime/(run_id+'.json')).exists() and (self.runtime/(run_id+'.json')).stat().st_size:
            from .common import read_json
            existing=read_json(self.runtime/(run_id+'.json'))
            if existing.get('status')!='queued':raise ValueError('run_id_already_exists')
        if self.graph.checkpointer and self.graph.get_state(config).values:raise ValueError('run_id_already_exists')
        write_json(self.runtime/(run_id+'.json'),{'run_id':run_id,'status':'running','graph_version':self.graph_version,'started_at':time.time()})
        try:
            state=self.graph.invoke({'run_id':run_id,'request':request,'drafts':{},'finals':{},'issues':[],'trace':[]},config)
            report=state['report'];events=state['trace']
        except Exception as exc:
            partial={}
            if self.graph.checkpointer:
                try:partial=self.graph.get_state(config).values
                except Exception:pass
            report={'run_id':run_id,'status':'failed','schema_version':2,'orchestrator':'langgraph','graph_version':self.graph_version,
                    'document':partial.get('document',{}),'tasks':partial.get('tasks',[]),'review':partial.get('review',{}),
                    'results':partial.get('finals') or partial.get('drafts') or {},'issues':partial.get('issues',[]),'error':str(exc)[:400]}
            events=partial.get('trace',[])+[event('leader','failed',error=str(exc)[:400])]
        report['elapsed_seconds']=round(time.monotonic()-start,3)
        report['trace']=[{**e,'seq':i} for i,e in enumerate(sorted(events,key=lambda x:x['time']))]
        write_json(self.runtime/(run_id+'.json'),report)
        return report
