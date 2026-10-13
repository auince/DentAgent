"""Specialists explain risk using source evidence and independently computed rule levels."""
import json
from .common import parse_json,tooth_key
from .consult_contracts import REGISTRY
from .skills import BOUNDARY


class SpecialtyExplanation:
    def __init__(self,task,client):self.task,self.client=task,client

    def evidence(self,result,text):
        references=[]
        for i,item in enumerate(result.get('evidence',[])):
            quote=item.get('quote','');start=item.get('start');end=item.get('end')
            source_id=item.get('source_id','record')
            source=result.get('sources',{}).get(source_id,{}).get('text',text)
            if not isinstance(quote,str) or not quote or type(start) is not int or type(end) is not int or source[start:end]!=quote:continue
            if item.get('supported') is False:continue
            references.append({'id':item.get('factor_id') or 'factor-evidence-'+str(i),
                'factor':item.get('factor'),'quote':quote,'source_id':source_id,'start':start,'end':end,
                'source_span_verified':True,'clinical_entailment_verified':False})
        return references

    def run(self,result,text=''):
        feedback=None
        for attempt in range(2):
            try:return self._attempt(result,text,feedback)
            except ValueError as exc:
                feedback=str(exc)[:240]
                if attempt==1:raise

    def _attempt(self,result,text='',feedback=None):
        references=self.evidence(result,text);legal={e['id']:e for e in references}
        scores=result.get('assessments',[])
        fixed=[{'tooth':r['tooth'],'risk_level':str(r['grade']) if r.get('grade') is not None else 'unknown',
            'basis':'deterministic_rule_engine','provisional_grade':r.get('provisional_grade'),
            'score':r.get('score'),'rule_version':r.get('rule_version')} for r in scores]
        qualitative=self.task in {'pulp_periapical','periodontal'}
        prompt=(BOUNDARY+'你是'+REGISTRY[self.task]['name']+'专病Agent的分析说明模块。'+(
            '风险等级严格使用给定 rule_levels，不得重算或改变独立规则引擎等级。每个患牙一项；未知保持unknown，临时等级仅可作为信息不足下的规则结果说明。'
            if fixed else '当前没有可用的确定性分级。'+(
                '允许给出low/moderate/high/unknown的暂定风险判断，必须说明依据与缺少的检查；不能当作确诊或已验证临床分级。证据不足用unknown，不因未记载而判低风险。'
                if qualitative else '风险等级只能为unknown，缺少规则因素无法判断根管难度或牙隐裂规则等级；不要自行发明评分。'))+
            '仅根据结构化因素和原文证据解释，不调用RAG、不输出知识文档、不补造检查、诊断或处方。')
        prompt+='仅输出 JSON：{"summary":"中文分析说明","risk_assessments":[{"tooth":null,"risk_level":"等级","explanation":"为什么是此等级及限制","evidence_ids":["原文证据ID"],"missing_information":["待补充因素"]}]}。'
        raw=parse_json(self.client.complete(self.task,prompt,json.dumps({'factors':result.get('factors',[]),
            'rule_levels':fixed,'evidence':references,'issues':result.get('issues',[]),'validation_feedback':feedback},ensure_ascii=False)))
        if not isinstance(raw,dict) or not isinstance(raw.get('summary'),str) or not raw['summary'].strip() or not isinstance(raw.get('risk_assessments'),list) or not raw['risk_assessments']:raise ValueError('invalid_specialty_explanation')
        expected={r['tooth']:r for r in fixed};seen=set();items=[]
        supported_teeth={f.get('tooth') for f in result.get('facts',[]) if f.get('tooth')}
        for item in raw['risk_assessments']:
            if not isinstance(item,dict):raise ValueError('invalid_risk_assessment')
            tooth=tooth_key(str(item['tooth'])) if item.get('tooth') is not None else None
            level=item.get('risk_level');ids=item.get('evidence_ids');missing=item.get('missing_information')
            if not isinstance(item.get('explanation'),str) or not item['explanation'].strip() or not isinstance(ids,list) or any(not isinstance(x,str) or x not in legal for x in ids) or not isinstance(missing,list) or any(not isinstance(x,str) for x in missing):raise ValueError('invalid_risk_evidence')
            if tooth in seen:raise ValueError('duplicate_risk_tooth')
            seen.add(tooth)
            if fixed:
                if tooth not in expected or level!=expected[tooth]['risk_level']:raise ValueError('specialty_rule_level_disagreement')
                for eid in ids:
                    factor=legal[eid].get('factor')
                    if factor and factor[0] not in {'patient',tooth}:raise ValueError('risk_evidence_wrong_tooth')
                basis=expected[tooth]
            else:
                if tooth is not None and tooth.removesuffix('牙') not in supported_teeth:raise ValueError('risk_unsupported_tooth')
                if level not in ({'low','moderate','high','unknown'} if qualitative else {'unknown'}):raise ValueError('invalid_provisional_risk_level')
                if tooth is not None:
                    for eid in ids:
                        fact=next((f for f in result.get('facts',[]) if f['id']==eid),None)
                        if fact and fact.get('tooth') and fact['tooth']!=tooth.removesuffix('牙'):raise ValueError('risk_evidence_wrong_tooth')
                basis={'basis':'model_provisional' if qualitative else 'insufficient_rule_factors','provisional':True}
            if level=='unknown' and not missing:raise ValueError('unknown_risk_requires_missing_information')
            if level!='unknown' and not ids:raise ValueError('risk_level_without_source_evidence')
            items.append({**item,'tooth':tooth,**basis,'risk_level':level,'evidence':[legal[i] for i in ids],
                'clinical_entailment_verified':False,'confirmed_diagnosis':False,
                'level_system':{'root_canal':'root_treatment_difficulty','crack':'project_crack_confidence','pulp_periapical':'provisional_clinical_risk','periodontal':'provisional_clinical_risk'}[self.task]})
        if fixed and seen!=set(expected):raise ValueError('risk_rule_tooth_coverage')
        if not fixed and supported_teeth and {t.removesuffix('牙') for t in seen if t is not None}!=supported_teeth:raise ValueError('risk_symptom_tooth_coverage')
        return {'summary':raw['summary'],'risk_assessments':items,'author':'specialty_agent',
                'task':self.task,'confirmed_diagnosis':False,'clinical_entailment_verified':False}
