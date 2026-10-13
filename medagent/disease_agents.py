"""Two disease-specific subgraphs using source facts and bounded explanation."""
import json
import operator
import time
from typing import TypedDict,Annotated
from langgraph.graph import START,END,StateGraph
from .common import ROOT,parse_json,digest
from .consult_contracts import REGISTRY,FACT_PROMPT,EXPERT_PROMPT,validate_facts
from .evidence import validate_expert
from .retrieval import KnowledgeIndex


def trace_event(step,status='done',**details):
    return {'time':time.time(),'step':step,'status':status,**details}


class DiseaseState(TypedDict,total=False):
    text:str
    scope:list
    attempt:int
    raw:dict
    facts:list
    sources:dict
    error:str
    failed:bool
    knowledge:list
    evidence:list
    result:dict
    trace:Annotated[list,operator.add]


class DiseaseAgent:
    def __init__(self,task,client):
        if task not in {'pulp_periapical','periodontal'}:raise ValueError('unsupported_disease_agent')
        self.task,self.client=task,client
        self.index=KnowledgeIndex(ROOT/'knowledge')
        graph=StateGraph(DiseaseState)
        for name,fn in [('extract',self.extract),('validate',self.validate),('retrieve',self.retrieve),('ground',self.ground),('explain',self.explain),('finish',self.finish)]:graph.add_node(name,fn)
        graph.add_edge(START,'extract');graph.add_edge('extract','validate')
        graph.add_conditional_edges('validate',lambda s:'retrieve' if not s.get('error') else 'extract' if s['attempt']<2 else 'finish')
        graph.add_edge('retrieve','ground');graph.add_edge('ground','explain');graph.add_edge('explain','finish');graph.add_edge('finish',END)
        self.graph=graph.compile(checkpointer=False,name='specialty_'+task)

    def extract(self,s):
        attempt=s.get('attempt',0)+1
        sources={'record':{'text':s['text'],'kind':'clinical_record'}}
        try:
            raw=parse_json(self.client.complete(self.task,FACT_PROMPT+
                '\n仅提取'+REGISTRY[self.task]['name']+'相关因素；患者级背景与患牙事实均保留来源。只使用原文牙位，无检查不能补写。',
                json.dumps({'sources':sources,'scope':s.get('scope',[]),'validation_feedback':s.get('error')},ensure_ascii=False)))
            return {'attempt':attempt,'raw':raw,'sources':sources,'error':'','failed':False,
                    'trace':[trace_event(self.task+'.extract',attempt=attempt)]}
        except Exception as exc:return {'attempt':attempt,'raw':{},'sources':sources,'error':str(exc)[:200],
            'trace':[trace_event(self.task+'.extract','failed',attempt=attempt,error=str(exc)[:200])]}

    def validate(self,s):
        try:
            facts,issues=validate_facts(s['raw'],s['sources'])
            if issues or not facts:raise ValueError('invalid_or_empty_source_facts:'+','.join(issues))
            if s.get('scope'):
                allowed={str(t).removesuffix('牙') for t in s['scope']}
                if any(f.get('tooth') and f['tooth'] not in allowed for f in facts):raise ValueError('out_of_scope_tooth')
            return {'facts':facts,'error':'','failed':False,'trace':[trace_event(self.task+'.validate','ok')]}
        except Exception as exc:return {'facts':[],'error':str(exc)[:240],'failed':True,'trace':[trace_event(self.task+'.validate','failed')]}

    def retrieve(self,s):
        return {'knowledge':self.index.search(s['text'],task=self.task,limit=4),'trace':[trace_event(self.task+'.retrieve')]}

    def ground(self,s):
        evidence=[{'factor_id':f['id'],'quote':f['quote'],'start':f['start'],'end':f['end'],
                   'source_span_verified':s['text'][f['start']:f['end']]==f['quote'],
                   'clinical_entailment_verified':False} for f in s['facts']]
        return {'evidence':evidence,'trace':[trace_event(self.task+'.ground')]}

    def explain(self,s):
        try:
            spec=REGISTRY[self.task]
            raw=parse_json(self.client.complete(self.task,EXPERT_PROMPT.format(specialty=spec['name'],scope=spec['scope']),
                json.dumps({k:s[k] for k in ('facts','sources','knowledge')},ensure_ascii=False)))
            validation=validate_expert(raw,s)
            result={'task':self.task,'status':'needs_review','factors':s['facts'],
                'patient_factors':[f for f in s['facts'] if f.get('tooth') is None],
                'tooth_factors':[f for f in s['facts'] if f.get('tooth') is not None],
                'evidence':s['evidence'],'knowledge':s['knowledge'],'assessments':[],
                'specialty_document':raw,'reference_validation':validation,
                'issues':[],'source_sha256':digest(s['text']),'confirmed_diagnosis':False}
            return {'result':result,'error':'','failed':False,'trace':[trace_event(self.task+'.explain','ok')]}
        except Exception as exc:return {'failed':True,'error':str(exc)[:240],'trace':[trace_event(self.task+'.explain','failed')]}

    def finish(self,s):
        if s.get('failed') or s.get('error'):
            return {'result':{'task':self.task,'status':'failed','error':s.get('error','unknown_error'),
                              'source_sha256':digest(s['text']),'issues':[]}}
        return {}

    def invoke(self,text,images=(),scope=None):
        return self.graph.invoke({'text':text,'scope':scope or [],'attempt':0,'trace':[]},{'recursion_limit':20})
