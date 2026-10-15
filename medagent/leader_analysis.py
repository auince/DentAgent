"""Leader-owned retrieval of real documents and synthesis of specialty risk explanations."""
import copy
import json
from .common import digest,parse_json
from .consult_contracts import REGISTRY,EXPERT_PROMPT
from .evidence import validate_expert
from .settings import get_settings
from .skills import BOUNDARY


class LeaderAnalysis:
    def __init__(self,client,index,registry=None):
        from .skills import build_skill_registry
        self.client,self.index=client,index
        self.registry=registry or build_skill_registry(client,index)

    def query(self,task,extraction):
        # Retain negation, uncertainty and provenance; do not insert model diagnoses.
        text=json.dumps({'specialty':REGISTRY[task]['name'],
            'structured_symptoms':extraction.get('factors',extraction.get('facts',[])),
            'specialty_risk':extraction.get('risk_assessments',[])},ensure_ascii=False)
        limit=get_settings().data['limits']['search_query_chars']
        return text[:limit],len(text)>limit

    def retrieve(self,task,extraction):
        query,truncated=self.query(task,extraction)
        envelope=self.registry.run('knowledge_search',task=task,payload={'query':query})
        hits=envelope['results']
        documents=[];by_source={}
        for rank,hit in enumerate(hits,1):
            key=(hit['source'],hit['sha256'])
            if key not in by_source:
                doc={'rank':len(documents)+1,'source':hit['source'],'sha256':hit['sha256'],
                    'source_url':hit.get('source_url'),'kind':'knowledge_document_match',
                    'match_basis':'retrieval_rank','confirmed_diagnosis':False,'excerpts':[]}
                by_source[key]=doc;documents.append(doc)
            by_source[key]['excerpts'].append({**hit,'retrieval_rank':rank})
        return {'knowledge':hits,'closest_disease_documents':documents,
                'retrieval':{**envelope['retrieval'],'owner':'leader','query':query,'query_truncated':truncated,
                    'extraction_sha256':digest(json.dumps(extraction.get('factors',[]),sort_keys=True,ensure_ascii=False))}}

    def organize(self,task,extraction,text='',images=(),retrieved=None,retrieve_allowed=True):
        result=copy.deepcopy(extraction)
        if result.get('status')=='failed':return result
        result['extraction_status']=result['status'];result['risk_analysis_owner']='specialty_agent';result['analysis_owner']='leader';result['synthesis_owner']='leader'
        result.setdefault('assessments',[]);result.setdefault('issues',[])
        # Dialogue fact lists do not satisfy the legacy scoring contracts.
        legacy=task=='root_canal' and isinstance(result.get('factors'),dict) or task=='crack' and result.get('factors') and all(isinstance(f,dict) and '患牙编号' in f for f in result['factors'])
        try:
            if retrieved is None and not retrieve_allowed:raise ValueError('leader_retrieval_budget_exhausted')
            result.update({k:v for k,v in (retrieved or self.retrieve(task,result)).items() if k!='input_version'})
            if not result['knowledge']:result['issues'].append({'code':'knowledge_not_found'})
            if result['retrieval'].get('degraded'):result['issues'].append({'code':'retrieval_degraded'})
            if result['retrieval']['query_truncated']:result['issues'].append({'code':'retrieval_query_truncated'})
        except Exception as exc:
            result.update(knowledge=[],closest_disease_documents=[],retrieval={'owner':'leader','status':'failed'})
            result['issues'].append({'code':'retrieval_failed','detail':str(exc)[:240]})
        try:
            if legacy:
                explanation=parse_json(self.client.complete('leader',BOUNDARY+
                    '整理结构化因素、独立规则结果和实际检索文档。不得改变规则分级或补写诊断、检查及处方。'
                    '说明未知因素与信息局限；知识文档仅表示相似匹配，不代表确诊。仅输出 '
                    '{"summary":"简短中文说明","citations":["实际知识片段ID"]}。',
                    json.dumps({'task':task,'factors':result['factors'],'assessments':result['assessments'],
                        'evidence':result.get('evidence',[]),'specialty_analysis':result.get('analysis_explanation',{}),'references':result['knowledge'],'issues':result['issues']},ensure_ascii=False)))
                if not isinstance(explanation,dict) or not isinstance(explanation.get('summary'),str) or not isinstance(explanation.get('citations'),list):raise ValueError('invalid_leader_summary')
                legal={r['id'] for r in result['knowledge']}
                if any(not isinstance(c,str) or c not in legal for c in explanation['citations']):raise ValueError('unverified_citation')
                result['explanation']=explanation
            else:
                raw=parse_json(self.client.complete('leader',EXPERT_PROMPT.format(specialty=REGISTRY[task]['name'],scope=REGISTRY[task]['scope']),
                    json.dumps({k:result[k] for k in ('facts','sources','knowledge')}|{'specialty_analysis':result.get('analysis_explanation',{})},ensure_ascii=False)))
                result['reference_validation']=validate_expert(raw,result)
                result['specialty_document']={**raw,'author':'leader','confirmed_diagnosis':False}
        except Exception as exc:result['issues'].append({'code':'leader_summary_failed','detail':str(exc)[:240]})
        result['status']='needs_review' if result['issues'] else 'assessed'
        result['confirmed_diagnosis']=False
        return result
