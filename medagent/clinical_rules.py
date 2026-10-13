"""Source-versioned candidate rules over grounded, explicitly typed clinical facts.

Extraction and quote-to-field entailment remain model judgments. This module
checks provenance, types and rule arithmetic; it does not confirm a diagnosis.
"""
import json
import math
import re
from .settings import configured_path

TASKS={'pulp_periapical','periodontal'}
RED_FLAGS=('breathing_difficulty','speaking_difficulty','swallowing_difficulty',
           'eye_swelling_or_pain','sudden_vision_change','extensive_oral_swelling',
           'marked_mouth_opening_difficulty')


def definition(task):
    if task not in TASKS:raise ValueError('unsupported_clinical_rule_task')
    return json.loads((configured_path('schemas_dir')/(task+'_rules.json')).read_text(encoding='utf-8'))


def extraction_contract(task):
    if task not in TASKS:return ''
    fields=definition(task)['fields']
    return ('\n对明确可提取的规则因素，在对应fact内增加clinical_key和clinical_value；数值另加clinical_unit。'
            '字段定义如下：'+json.dumps(fields,ensure_ascii=False)+
            '\n只选定义内的键；无明确证据不输出该键。数值保留原数和单位，不换算、不从症状估计。'
            '布尔否定用status=negative及clinical_value=false，阳性用positive及true；冲突/未知保留状态。'
            'objective=true只可来自clinical_record，不能把患者口述视为临床测试。'
            'scope=patient的字段tooth=null；scope=tooth必须有明确原文FDI牙位，否则只保留普通事实。'
            'max_*必须是明确全口最重值，不可从一颗牙推测。全口检查、排除原因、复杂度完整评估只有原文明确才填true。'
            '五年进展不能从其他时段外推；吸烟未提及不等于0；单位%使用百分数，如30，不是0.3。')


UNIT_MARKERS={'mm':('mm','毫米'),'%':('%','％','百分'),
              'teeth':('颗','牙','teeth'),'stage':('期','stage'),
              'years':('岁','年','year'),'cigarettes/day':('支','根','cigarette')}


def validate_clinical_facts(task,facts):
    if task not in TASKS:return facts
    fields=definition(task)['fields']
    for f in facts:
        key=f.get('clinical_key')
        if key is None:continue
        if key not in fields:raise ValueError('unknown_clinical_key')
        spec=fields[key];value=f.get('clinical_value')
        if f['status'] in {'unknown','conflict'}:
            f['clinical_value']=None;continue
        if spec['objective'] and f['source_kind']!='clinical_record':raise ValueError('objective_fact_requires_record:'+key)
        if spec['scope']=='patient' and f.get('tooth') is not None:raise ValueError('patient_field_has_tooth:'+key)
        if spec['scope']=='tooth' and not f.get('tooth'):raise ValueError('tooth_field_requires_location:'+key)
        if spec['type']=='boolean':
            if type(value) is not bool or value!=(f['status']=='positive'):raise ValueError('clinical_boolean_status_disagreement:'+key)
        else:
            if f['status']!='positive' or type(value) not in (int,float) or not math.isfinite(value) or not spec['min']<=value<=spec['max']:raise ValueError('invalid_clinical_measurement:'+key)
            if f.get('clinical_unit')!=spec['unit']:raise ValueError('invalid_clinical_unit:'+key)
            numbers=[float(n) for n in re.findall(r'(?<![\d.])\d+(?:\.\d+)?(?![\d.])',f['quote'])]
            if value not in numbers or not any(m in f['quote'].lower() for m in UNIT_MARKERS[spec['unit']]):raise ValueError('clinical_measurement_not_in_quote:'+key)
            if key in {'periodontitis_tooth_loss','prior_stage','cigarettes_per_day'} and value!=int(value):raise ValueError('clinical_count_not_integer:'+key)
            if key in {'prior_stage','age_years'} and value<=0:raise ValueError('clinical_measurement_requires_positive:'+key)
    return facts


class Inputs:
    def __init__(self,facts):
        self.facts=facts;self.used=set();self.conflicts=set()
    def value(self,key):
        rows=[f for f in self.facts if f.get('clinical_key')==key]
        if not rows:return None
        values={f.get('clinical_value') for f in rows}
        if any(f['status'] in {'unknown','conflict'} or f.get('conflicts_with') for f in rows) or len(values)!=1:
            self.conflicts.add(key);return None
        return next(iter(values))
    def use(self,*keys):
        for key in keys:
            if self.value(key) is not None:self.used.update(f['id'] for f in self.facts if f.get('clinical_key')==key)
    def match(self,key,value):return self.value(key) is value


def urgency(i):
    found=[k for k in RED_FLAGS if i.match(k,True)]
    if found:i.use(*found);return 'emergency_referral',found
    # Neither unspecified red flags nor their absence establish a low-risk state.
    local=[k for k in ('pus','swelling') if i.match(k,True)]
    if local:i.use(*local);return 'urgent_dental_review',local
    return 'unknown',[]


def envelope(task,d,items):
    return {'task':task,'rule_version':d['version'],'sources':d['sources'],
            'status':'requires_clinical_review','confirmed_diagnosis':False,
            'clinical_entailment_verified':False,'risk_assessments':items,
            'limitations':d['limitations']}


def pulp_rules(facts,d):
    global_inputs=Inputs([f for f in facts if f.get('tooth') is None])
    global_urgency,red=urgency(global_inputs)
    teeth=sorted({f['tooth'] for f in facts if f.get('tooth')}) or [None]
    items=[]
    for tooth in teeth:
        i=Inputs([f for f in facts if f.get('tooth')==tooth])
        candidates=[]
        if tooth is not None:
            for p in d['candidate_patterns']:
                matched_all=all(i.match(k,v) for k,v in p['all'])
                matched_any=[(k,v) for k,v in p['any'] if i.match(k,v)]
                if matched_all and (not p['any'] or matched_any):
                    keys=[k for k,v in p['all']]+[k for k,v in matched_any];i.use(*keys)
                    ids=sorted({f['id'] for f in i.facts if f.get('clinical_key') in keys})
                    candidates.append({'id':p['id'],'label':p['label'],'axis':p['axis'],
                        'evidence_ids':ids,'source_key':'diagnostic_terms','confirmed_diagnosis':False})
        local_urgency,flags=urgency(i)
        level='emergency_referral' if global_urgency=='emergency_referral' else local_urgency
        if level=='emergency_referral':i.used.update(global_inputs.used)
        missing=['临床综合检查确认；候选组合不等于充分必要诊断标准']
        if tooth is None:missing.append('明确患牙牙位；不合并未定位症状推导逐牙诊断')
        for axis,label in [('pulp','牙髓测试与病史'),('apical','根尖叩触诊、影像与病因')]:
            if not any(c['axis']==axis for c in candidates):missing.append(label)
        if i.value('cold_nonresponsive') is True and not any(c['id']=='pulp_necrosis' for c in candidates):missing.append('对照电测/冷测及假阴性排除，单次无反应不能定坏死')
        if i.match('deep_caries_or_exposure',True) and i.match('asymptomatic',True):missing.append('无症状性不可复性牙髓炎需临床判断，本实现不自动判定')
        if level=='unknown':missing.append('就诊紧急程度需补充症状与红旗；未记载不等于无风险')
        items.append({'tooth':tooth+'牙' if tooth else None,'risk_level':level,
            'level_system':'oral_care_urgency','basis':'reference_rule_engine',
            'diagnostic_candidates':candidates,'urgency_flags':sorted(set(red+flags)),
            'rule_version':d['version'],'rule_sources':d['sources'],'evidence_ids':sorted(i.used),
            'missing_information':missing,'conflicting_fields':sorted(i.conflicts),
            'confirmed_diagnosis':False,'clinical_entailment_verified':False})
    return envelope('pulp_periapical',d,items)


def periodontal_rules(facts,d):
    i=Inputs([f for f in facts if f.get('tooth') is None]);t=d['thresholds']
    u,flags=urgency(i);stage=grade=minimum_stage=minimum_grade=None;triggers=[];missing=[]
    case_keys=('interdental_cal_two_nonadjacent','buccal_oral_cal_case')
    selected=[k for k in case_keys if i.match(k,True)]
    case_supported=bool(selected) and i.match('nonperiodontal_causes_excluded',True) and i.match('special_periodontitis_excluded',True)
    if case_supported:
        i.use(*selected,'nonperiodontal_causes_excluded','special_periodontitis_excluded')
        cal=i.value('max_interdental_cal_mm');rbl=i.value('max_rbl_percent');severity=None
        if cal is not None:
            if 1<=cal<=2:severity=1
            elif 3<=cal<=4:severity=2
            elif cal>=t['cal_stage_iii']:severity=3
            else:missing.append('CAL落在表格整数区间之外，需要临床解释，不自动四舍五入')
            if severity:i.use('max_interdental_cal_mm');triggers.append('cal_severity')
        elif rbl is not None and rbl>0:
            severity=1 if rbl<t['rbl_stage_ii'] else 2 if rbl<=t['rbl_stage_iii'] else 3
            i.use('max_rbl_percent');triggers.append('rbl_fallback_severity')
        else:missing.append('全口最重邻面CAL；无法取得时使用最重RBL百分数')
        s=severity or 0;loss=i.value('periodontitis_tooth_loss');ppd=i.value('max_ppd_mm')
        if loss is not None and loss>=t['tooth_loss_stage_iii']:
            s=max(s,4 if loss>=t['tooth_loss_stage_iv'] else 3);i.use('periodontitis_tooth_loss');triggers.append('periodontitis_attributable_tooth_loss')
        if ppd is not None and ppd>t['ppd_stage_ii'] and i.match('true_periodontal_pockets',True):
            s=max(s,3 if ppd>=t['ppd_stage_iii'] else 2);i.use('max_ppd_mm','true_periodontal_pockets');triggers.append('true_pocket_complexity')
        for key in ('vertical_defect_ge3','furcation_ii_iii','moderate_ridge_defect'):
            if i.match(key,True):s=max(s,3);i.use(key);triggers.append(key)
        if i.match('stage_iv_rehabilitation_complexity',True):s=4;i.use('stage_iv_rehabilitation_complexity');triggers.append('stage_iv_rehabilitation_complexity')
        prior=i.value('prior_stage')
        if prior is not None:s=max(s,int(prior));i.use('prior_stage');triggers.append('historical_stage_floor')
        complete_keys=('periodontitis_tooth_loss','max_ppd_mm','vertical_defect_ge3',
                       'furcation_ii_iii','moderate_ridge_defect','stage_iv_rehabilitation_complexity')
        complete=(severity is not None and i.match('full_mouth_assessment',True) and
            i.match('complexity_review_complete',True) and all(i.value(k) is not None for k in complete_keys) and
            (ppd<=5 or i.match('true_periodontal_pockets',True)))
        if s:
            minimum_stage=['I','II','III','IV'][s-1]
            if complete or s==4:stage=minimum_stage
            if complete:i.use('full_mouth_assessment','complexity_review_complete',*complete_keys)
        if stage is None:missing.append('完整全口评估、牙周归因缺失牙及复杂度排查；仅给分期下限')
        direct=i.value('progression_5y_mm');age=i.value('age_years');g=None;ratio=None
        if direct is not None:
            g=0 if direct==0 else 1 if direct<t['grade_c_5y_mm'] else 2
            i.use('progression_5y_mm');triggers.append('direct_five_year_progression')
        elif age is not None and age>0 and rbl is not None and i.match('full_mouth_assessment',True):
            ratio=rbl/age;g=0 if ratio<t['grade_b_ratio'] else 1 if ratio<=t['grade_c_ratio'] else 2
            i.use('age_years','max_rbl_percent','full_mouth_assessment');triggers.append('indirect_rbl_percent_divided_by_age')
        else:missing.append('五年纵向进展，或全口最重RBL百分数与年龄；不把默认B当已证实')
        smoke=i.value('cigarettes_per_day');diabetes=i.value('diabetes');hba=i.value('hba1c_percent')
        if smoke is not None and smoke>0:
            g=max(g if g is not None else -1,2 if smoke>=t['smoking_grade_c'] else 1)
            i.use('cigarettes_per_day');triggers.append('smoking_grade_floor')
        if diabetes is True and hba is not None:
            g=max(g if g is not None else -1,2 if hba>=t['diabetes_grade_c_hba1c'] else 1)
            i.use('diabetes','hba1c_percent');triggers.append('diabetes_grade_floor')
        progression_present=direct is not None or ratio is not None
        modifiers_complete=smoke is not None and diabetes is not None and (diabetes is False or hba is not None)
        if g is not None:
            minimum_grade='ABC'[g]
            if g==2 or progression_present and modifiers_complete:grade=minimum_grade
            if modifiers_complete:i.use('cigarettes_per_day','diabetes',*(['hba1c_percent'] if diabetes else []))
        if not modifiers_complete:missing.append('吸烟状态及糖尿病/HbA1c资料；已知风险修饰只能升级')
    else:
        missing.append('牙周炎病例依据、非牙周原因排除及特殊牙周疾病鉴别；不能仅凭牙龈出血分期分级')
    extent=None;affected=i.value('affected_teeth_percent')
    if case_supported and affected is not None and i.match('full_mouth_assessment',True):
        extent='localized' if affected<t['generalized_percent'] else 'generalized';i.use('affected_teeth_percent','full_mouth_assessment')
    level='Grade '+grade if grade else 'unknown'
    if level=='unknown' and not missing:missing.append('分级证据不足')
    item={'tooth':None,'risk_level':level,'level_system':'periodontitis_progression_grade',
        'basis':'reference_rule_engine','case_supported':case_supported,'stage':stage,
        'minimum_stage':minimum_stage,'grade':grade,'minimum_grade':minimum_grade,'extent':extent,
        'care_urgency':u,'urgency_flags':flags,'triggered_rules':triggers,
        'rule_version':d['version'],'rule_sources':d['sources'],'evidence_ids':sorted(i.used),
        'missing_information':missing,'conflicting_fields':sorted(i.conflicts),
        'confirmed_diagnosis':False,'clinical_entailment_verified':False}
    return envelope('periodontal',d,[item])


def assess_clinical(task,facts):
    d=definition(task)
    result=periodontal_rules(facts,d) if task=='periodontal' else pulp_rules(facts,d)
    by_id={f['id']:f for f in facts}
    for item in result['risk_assessments']:
        item['evidence']=[{k:v for k,v in by_id[eid].items() if k in {'id','source_id','quote','start','end','source_kind','clinical_key','clinical_value','clinical_unit','validation'}} for eid in item['evidence_ids']]
    return result
