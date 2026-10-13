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
        clinical=result.get('clinical_rule_assessment')
        if clinical:fixed=clinical['risk_assessments']
        prompt=(BOUNDARY+'你是'+REGISTRY[self.task]['name']+'专病Agent的分析说明模块。'+(
            '严格解释给定 rule_levels，risk_level、牙位、病例门槛、分期、分级、候选诊断与就诊紧急程度都不得改变。'
            '牙周风险是进展Grade，Stage是严重程度；minimum_*只代表下限，不能当已确定分期分级。'
            '牙髓risk_level是就诊紧急程度，不是疾病评分；diagnostic_candidates只是假设候选。'
            '必须引用全部规则触发evidence_ids，并保留原始missing_information。不得从无记录推断阴性或低风险。'
            if clinical else '风险等级严格使用给定 rule_levels，不得重算或改变独立规则引擎等级；未知保持unknown。'
            if fixed else '风险等级只能为unknown，缺少规则因素无法判断；不要自行发明评分。')+
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
                if clinical and tooth is not None:
                    for eid in ids:
                        fact=next((f for f in result.get('facts',[]) if f['id']==eid),None)
                        if fact and fact.get('tooth') not in {None,tooth.removesuffix('牙')}:raise ValueError('risk_evidence_wrong_tooth')
                basis=expected[tooth]
                if clinical and not set(basis['evidence_ids'])<=set(ids):raise ValueError('missing_rule_trigger_evidence')
                if clinical:missing=list(dict.fromkeys(basis['missing_information']+missing))
            else:
                if tooth is not None and tooth.removesuffix('牙') not in supported_teeth:raise ValueError('risk_unsupported_tooth')
                if level!='unknown':raise ValueError('invalid_unsupported_risk_level')
                if tooth is not None:
                    for eid in ids:
                        fact=next((f for f in result.get('facts',[]) if f['id']==eid),None)
                        if fact and fact.get('tooth') and fact['tooth']!=tooth.removesuffix('牙'):raise ValueError('risk_evidence_wrong_tooth')
                basis={'basis':'insufficient_rule_factors','provisional':True}
            if level=='unknown' and not missing:raise ValueError('unknown_risk_requires_missing_information')
            if level!='unknown' and not ids:raise ValueError('risk_level_without_source_evidence')
            items.append({**item,'tooth':tooth,**basis,'risk_level':level,'evidence':[legal[i] for i in ids],
                'missing_information':missing,'clinical_entailment_verified':False,'confirmed_diagnosis':False,
                'level_system':{'root_canal':'root_treatment_difficulty','crack':'project_crack_confidence','pulp_periapical':'oral_care_urgency','periodontal':'periodontitis_progression_grade'}[self.task]})
        if fixed and seen!=set(expected):raise ValueError('risk_rule_tooth_coverage')
        if not fixed and supported_teeth and {t.removesuffix('牙') for t in seen if t is not None}!=supported_teeth:raise ValueError('risk_symptom_tooth_coverage')
        return {'summary':raw['summary'],'risk_assessments':items,'author':'specialty_agent',
                'task':self.task,'confirmed_diagnosis':False,'clinical_entailment_verified':False}
