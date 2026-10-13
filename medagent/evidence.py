"""Source/reference validation, deliberately distinct from clinical entailment."""
import re

def validate_expert(result,state):
    if not isinstance(result,dict) or any(not isinstance(result.get(k),list) for k in ('findings','directions','missing_information')):
        raise ValueError('invalid_expert_schema')
    facts={f['id']:f for f in state['facts']};knowledge={k['id']:k for k in state['knowledge']}
    if not result['findings']:raise ValueError('expert_no_supported_findings')
    for finding in result['findings']:
        if not isinstance(finding,dict) or not isinstance(finding.get('interpretation'),str) or not finding['interpretation'].strip():raise ValueError('invalid_finding')
        ids=finding.get('fact_ids')
        if not isinstance(ids,list) or not ids or any(i not in facts for i in ids):raise ValueError('expert_unknown_fact')
        for fid in ids:
            f=facts[fid];source=state['sources'].get(f.get('source_id'))
            if not source or not f.get('quote') or f['quote'] not in source['text']:raise ValueError('expert_unlocated_fact')
    for direction in result['directions']:
        if not isinstance(direction,dict) or not isinstance(direction.get('text'),str):raise ValueError('invalid_direction')
        ids=direction.get('evidence_ids')
        if not isinstance(ids,list) or not ids or any(i not in knowledge for i in ids):raise ValueError('expert_unsupported_reference')
    return {'reference_links_valid':True,'clinical_entailment_verified':False}

def evidence_report(state):
    facts=[]
    for f in state.get('facts',[]):
        source=state['sources'].get(f['source_id'],{}).get('text','')
        start=f.get('start');end=f.get('end')
        located=isinstance(start,int) and isinstance(end,int) and source[start:end]==f['quote']
        facts.append({'id':f['id'],'quote':f['quote'],'offset':{'start':start,'end':end,'unit':'unicode_codepoint'},
            'source_id':f['source_id'],'source_kind':f.get('source_kind'),'status':f['status'],'tooth':f.get('tooth'),
            'conflicts_with':f.get('conflicts_with',[]),'source_span_verified':located,'supported':None})
    links=[]
    for task,result in state.get('leader_analysis',{}).items():
        for finding in result.get('specialty_document',{}).get('findings',[]):links.append({'specialty':task,**finding,'support_type':'source_reference_only'})
    risk_links=[{'specialty':task,'risk_level':risk['risk_level'],'basis':risk['basis'],
        'evidence_ids':risk['evidence_ids'],'clinical_entailment_verified':False}
        for task,result in state.get('experts',{}).items() for risk in result.get('risk_assessments',[])]
    return {'facts':facts,'leader_links':links,'specialty_risk_links':risk_links,'expert_links':[],'knowledge':[{k:v for k,v in item.items() if k!='tf'} for item in state.get('knowledge',[])],
        'supported_semantics':'null means clinical entailment has not been independently verified'}
