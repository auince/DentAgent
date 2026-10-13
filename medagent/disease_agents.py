"""Specialties extract symptoms and explain provisional risk; the Leader owns RAG and synthesis."""
import json
import operator
import time
from typing import TypedDict,Annotated
from langgraph.graph import START,END,StateGraph
from .common import parse_json,digest
from .consult_contracts import REGISTRY,FACT_PROMPT,validate_facts


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
    evidence:list
    result:dict
    trace:Annotated[list,operator.add]


class DiseaseAgent:
    """Clinical disease extraction and all specialties' patient-dialogue extraction."""
    def __init__(self,task,client):
        if task not in REGISTRY:raise ValueError('unsupported_disease_agent')
        self.task,self.client=task,client
        graph=StateGraph(DiseaseState)
        for name,fn in [('extract',self.extract),('validate',self.validate),('ground',self.ground),('explain',self.explain),('finish',self.finish)]:graph.add_node(name,fn)
        graph.add_edge(START,'extract');graph.add_edge('extract','validate')
        graph.add_conditional_edges('validate',lambda s:'ground' if not s.get('error') else 'extract' if s['attempt']<2 else 'finish')
        graph.add_edge('ground','explain');graph.add_edge('explain','finish');graph.add_edge('finish',END)
        self.graph=graph.compile(checkpointer=False,name='specialty_'+task)

    def extract(self,s):
        attempt=s.get('attempt',0)+1
        sources=s.get('sources') or {'record':{'text':s['text'],'kind':'clinical_record'}}
        try:
            raw=parse_json(self.client.complete(self.task,FACT_PROMPT+
                '\n你是'+REGISTRY[self.task]['name']+'结构化症状提取Agent。仅提取本专业相关的症状、患者级背景、患牙因素。'
                '仅使用给定来源和明确牙位。保留否定、未知、冲突，不生成诊断、疾病方向、评分、知识检索或治疗建议。',
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

    def ground(self,s):
        evidence=[{'factor_id':f['id'],'source_id':f['source_id'],'quote':f['quote'],'start':f['start'],'end':f['end'],
                   'source_span_verified':s['sources'][f['source_id']]['text'][f['start']:f['end']]==f['quote'],
                   'clinical_entailment_verified':False} for f in s['facts']]
        return {'evidence':evidence,'trace':[trace_event(self.task+'.ground')]}

    def explain(self,s):
        from .specialty_explanation import SpecialtyExplanation
        result={'factors':s['facts'],'facts':s['facts'],'sources':s['sources'],'evidence':s['evidence'],'issues':[]}
        try:
            explanation=SpecialtyExplanation(self.task,self.client).run(result,s['text'])
            result.update(analysis_explanation=explanation,risk_assessments=explanation['risk_assessments'])
            result['issues'].append({'code':'provisional_risk_requires_clinical_review'})
        except Exception as exc:
            result.update(risk_assessments=[])
            result['issues'].append({'code':'specialty_explanation_failed','detail':str(exc)[:240]})
        return {'result':result,'trace':[trace_event(self.task+'.explain','done' if result['risk_assessments'] else 'failed',risk_count=len(result['risk_assessments']))]}

    def finish(self,s):
        common={'task':self.task,'role':'structured_symptom_and_risk_analysis','source_sha256':digest(s['text']),
                'input_version':digest(json.dumps(s['sources'],sort_keys=True,ensure_ascii=False))}
        if s.get('failed') or s.get('error'):
            return {'result':{**common,'status':'failed','error':s.get('error','unknown_error'),'issues':[]}}
        return {'result':{**s['result'],**common,'status':'needs_review' if s['result']['issues'] else 'extracted','factors':s['facts'],'facts':s['facts'],
            'patient_factors':[f for f in s['facts'] if f.get('tooth') is None],
            'tooth_factors':[f for f in s['facts'] if f.get('tooth') is not None],
            'sources':s['sources'],'evidence':s['evidence'],'issues':s['result']['issues']}}

    def invoke(self,text,images=(),scope=None):
        return self.graph.invoke({'text':text,'scope':scope or [],'attempt':0,'trace':[]},{'recursion_limit':12})

    def invoke_sources(self,sources,scope=None):
        text='\n'.join(source['text'] for source in sources.values())
        return self.graph.invoke({'text':text,'sources':sources,'scope':scope or [],'attempt':0,'trace':[]},{'recursion_limit':12})
