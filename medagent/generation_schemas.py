"""Constrain generated references to visible IDs; retain semantic validation."""
from .consult_contracts import REGISTRY
from .routing import FIELDS
import re

def response_schema(name, schema):
    return {'type':'json_schema','json_schema':{'name':name,'strict':True,'schema':schema}}

def object_schema(properties, required=None):
    return {'type':'object','properties':properties,'required':required or list(properties),'additionalProperties':False}

def references(ids, minimum=1):
    schema={'type':'array','items':{'type':'string','enum':list(ids)},'minItems':minimum}
    if not ids:
        schema.update(items={'type':'string'},maxItems=0,minItems=0)
    return schema

def needs_schema(facts):
    question=object_schema({'field':{'type':'string','enum':sorted(FIELDS)},'question':{'type':'string'}})
    candidate=object_schema({'task':{'type':'string','enum':list(REGISTRY)},
        'decision':{'type':'string','enum':['select','defer','exclude']},
        'fact_ids':references([f['id'] for f in facts],0),'reason':{'type':'string'},
        'missing':{'type':'array','items':question}})
    return response_schema('oral_needs',object_schema({'candidates':{'type':'array','items':candidate,'maxItems':len(REGISTRY)}}))

def expert_schema(state):
    finding=object_schema({'fact_ids':references([f['id'] for f in state['facts']]),'interpretation':{'type':'string','minLength':1}})
    direction=object_schema({'text':{'type':'string'},'evidence_ids':references([k['id'] for k in state['knowledge']])})
    # Facts are reconstructed from referenced source objects, never rewritten by the teacher.
    directions={'type':'array','items':direction}
    if not state['knowledge']:directions['maxItems']=0
    return response_schema('oral_expert',object_schema({'findings':{'type':'array','items':finding,'minItems':1},
        'directions':directions,'missing_information':{'type':'array','items':{'type':'string'}}}))

def action_schema(state, config):
    from .consult_contracts import finish_issue
    variants=[object_schema({'action':{'const':'refer'},'reason':{'type':'string'}})]
    if not state.get('urgent'):
        if not finish_issue(state,config):variants.append(object_schema({'action':{'const':'finish'}}))
        if state['question_count']<config['max_questions']:
            fields=sorted(FIELDS-set(state.get('asked_fields',[])))
            if fields:variants.append(object_schema({'action':{'const':'ask'},'field':{'type':'string','enum':fields},'question':{'type':'string'}}))
        remaining=config['max_tools']-state['tool_count']
        if remaining>0:
            variants.append(object_schema({'action':{'const':'retrieve'},'task':{'type':'string','enum':[*REGISTRY,'all']},'query':{'type':'string'}}))
            from .routing import selected_tasks
            tasks=sorted(selected_tasks(state['needs'])-{k for k,v in state['experts'].items() if v.get('status')!='failed'})
            if tasks:
                completed=sum(v.get('status')!='failed' for v in state['experts'].values())
                for independent,limit in ((False,config['max_experts']),(True,config['max_experts_independent'])):
                    capacity=min(remaining,limit-completed,len(tasks))
                    if capacity>0:
                        variants.append(object_schema({'action':{'const':'dispatch'},
                            'independent_problems':{'const':independent},'tasks':{
                                'type':'array','items':{'type':'string','enum':tasks},
                                'minItems':1,'maxItems':capacity}}))
    return response_schema('oral_action',{'anyOf':variants})

def complaint_schema(card):
    original=card['patient_text']
    spans={original,*[s.strip() for s in re.split(r'(?<=[。！？；])',original) if s.strip()]}
    spans.update(f['quote'] for f in card['initial_facts'] if f.get('source_kind')=='patient_report' and f['quote'] in original)
    explicit=re.findall(r'(?:牙位|牙齿)[：:\s]*([1-4][1-8])|(?<!\d)([1-4][1-8])(?:号)?牙',original)
    teeth=sorted({a or b for a,b in explicit})
    segment=object_schema({'text':{'type':'string'},'original_quote':{'type':'string','enum':sorted(spans)},
        'name':{'type':'string'},'value':{'type':'string'},
        'status':{'type':'string','enum':['positive','negative','unknown','conflict']},
        'location':{'type':['string','null']},'tooth':{'enum':[None,*teeth]}})
    return response_schema('oral_complaint_segments',object_schema({'segments':{'type':'array','items':segment,'minItems':1,'maxItems':20},'intent':{'type':'string'}}))
