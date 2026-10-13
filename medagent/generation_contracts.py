"""Constrained teacher output shapes; source and semantic checks remain separate."""
import re
from .consult_contracts import REGISTRY

FIELDS=['duration','trigger','location','swelling','history','report']

def obj(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}

def arr(items,minimum=0):return {'type':'array','items':items,'minItems':minimum}

def fact_schema(sources):
    return obj({'name':{'type':'string','minLength':1},'value':{'type':'string'},
        'status':{'type':'string','enum':['positive','negative','unknown','conflict']},
        'location':{'type':['string','null']},'tooth':{'type':['string','null']},
        'quote':{'type':'string','minLength':1},'source_id':{'type':'string','enum':sources}})

def card_format(chunks,record_visible,category=None,tasks=None):
    ids=[c['id'] for c in chunks]
    support=[]
    for chunk in chunks:
        quotes=[x.strip() for x in re.split(r'(?<=[.!?。])\s+|\n',chunk['text']) if 15<=len(x.strip())<=350]
        if not quotes:quotes=[chunk['text'][:300]]
        support.append(obj({'id':{'type':'string','enum':[chunk['id']]},
            'quote':{'type':'string','minLength':1},'supports':{'type':'string'}}))
    answer=obj({'text':{'type':'string'},'fact':fact_schema(['followup'])})
    schema=obj({'patient_text':{'type':'string','minLength':1},'clinical_record':{'type':'string'},
        'initial_facts':arr(fact_schema(['patient','record'] if record_visible else ['patient']),1),
        'required_tasks':arr({'type':'string','enum':list(REGISTRY)}),
        'useful_questions':arr({'type':'string','enum':FIELDS}),
        'patient_answers':obj({field:{'anyOf':[{'type':'null'},answer]} for field in FIELDS}),
        'knowledge_ids':arr({'type':'string','enum':ids},1),
        'knowledge_support':arr({'anyOf':support},1)})
    if category=='urgent':schema['properties']['required_tasks']['maxItems']=0
    elif category=='single' and tasks:
        schema['properties']['required_tasks']={'type':'array','items':{'type':'string','enum':[tasks[0]]},'minItems':1,'maxItems':1}
    return {'type':'json_schema','json_schema':{'name':'oral_case','strict':True,'schema':schema}}
