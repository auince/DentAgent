import re
from .common import ROOT, labels, norm, option_key, parse_json, read_json, tooth_key

PERSON = ['全身病史及麻醉', '最大张口度', '咽反射', '牙科恐惧症', '口腔卫生护理']
COMMON = ['诊断', '牙周病损', '根管感染程度', '牙齿位置', '根管钙化']
FIRST = COMMON + ['牙冠形态和修复体形态']
REPEAT = COMMON + ['牙冠形态', '冠修复', '根充材料和质量']
SYMPTOMS = {'root_canal':['诊断','牙周病损','根管感染程度'],
            'crack':['咬合受力特征','临床激发体征','裂纹根部波及征象']}


def normalize_output(task, value):
    value = parse_json(value)
    if task == 'crack':
        value = value.get('牙隐裂分析', value) if isinstance(value, dict) else value
        if not isinstance(value, list) or not value:
            raise ValueError('牙隐裂输出必须为非空患牙数组')
        if not all(isinstance(t, dict) for t in value):
            raise ValueError('invalid_tooth_object')
        return value
    if not isinstance(value, dict) or 'toothLabel' not in value or 'personLabel' not in value:
        raise ValueError('根管输出缺少 personLabel/toothLabel')
    person, teeth = parse_json(value['personLabel']), parse_json(value['toothLabel'])
    if not isinstance(person, dict) or not isinstance(teeth, list) or not teeth or not all(isinstance(t,dict) for t in teeth):
        raise ValueError('根管标签类型不合法')
    return {'personLabel':person, 'toothLabel':teeth}


def assess(task, value):
    value = normalize_output(task, value)
    problems = []
    results = []
    seen = set()
    if task == 'crack':
        rules = read_json(ROOT/'schemas/crack_rules.json')
        for tooth in value:
            pos = tooth_key(tooth.get('患牙编号',''))
            if not pos or pos in seen:
                raise ValueError('missing_or_duplicate_tooth')
            seen.add(pos)
            scores = {}
            for field, choices in rules['points'].items():
                vals = tooth.get(field)
                canonical={option_key(k):v for k,v in choices.items()}
                if not isinstance(vals, list) or any(not isinstance(x,str) or option_key(x) not in canonical for x in vals):
                    raise ValueError('invalid_taxonomy:' + field)
                if len(vals) != len({option_key(x) for x in vals}):
                    raise ValueError('duplicate_option:' + field)
                sub = sum(canonical[option_key(x)] for x in vals)
                cap = rules['caps'][field]
                scores[field] = min(sub,cap) if cap is not None else sub
            total = sum(scores.values())
            grade = 'Strong' if total >= 45 else 'Moderate' if total >= 15 else 'Low_Confidence'
            results.append({'tooth':pos,'score':total,'grade':grade,'module_scores':scores,
                            'rule_version':rules['version']})
    elif task == 'root_canal':
        taxonomy = read_json(ROOT/'schemas/root_taxonomy.json')
        person = value['personLabel']
        for tooth in value['toothLabel']:
            pos = tooth_key(tooth.get('position',''))
            if not pos or pos in seen or not isinstance(tooth.get('isFirst'),bool):
                raise ValueError('invalid_tooth_or_treatment_type')
            seen.add(pos)
            fields = FIRST if tooth['isFirst'] else REPEAT
            levels = []
            for obj,keys in [(person,PERSON),(tooth,fields)]:
                for field in keys:
                    vals = labels(obj.get(field,[]))
                    if not vals:
                        problems.append({'tooth':pos,'code':'missing_factor','field':field})
                    legal = {norm(x) for x in taxonomy[field]}
                    for label in set(vals):
                        if norm(label) not in legal:
                            raise ValueError('invalid_taxonomy:' + field + ':' + label)
                        m = re.search(r'([一二三])级因素',label)
                        if not m:
                            raise ValueError('missing_factor_level')
                        levels.append('一二三'.index(m[1])+1)
            n2,n3 = levels.count(2),levels.count(3)
            grade = 4 if n3>=2 else 3 if n3 or n2>=2 else 2 if n2 else 1 if levels else None
            model_grade = tooth.get('difficulty')
            if model_grade is not None and model_grade != grade:
                problems.append({'tooth':pos,'code':'model_rule_disagreement','model':model_grade,'rule':grade})
            results.append({'tooth':pos,'grade':grade,'model_grade':model_grade,
                'level2_count':n2,'level3_count':n3,'rule_version':'root-prompt-20260401'})
    else:
        raise ValueError('unsupported_scoring_task')
    missing_teeth={p['tooth'] for p in problems if p['code']=='missing_factor'}
    for result in results:
        result['assessment_origin']='deterministic_rule_engine'
        if result['tooth'] in missing_teeth:
            result['provisional_grade']=result['grade'];result['grade']=None;result['status']='incomplete'
        else:result['status']='assessed'
    return {'factors':value,'assessments':results,'issues':problems}


def factor_items(task, value):
    """Stable identity for evidence and multi-label evaluation."""
    value = normalize_output(task,value)
    out = []
    if task == 'crack':
        fields = read_json(ROOT/'schemas/crack_rules.json')['points']
        for tooth in value:
            out += [(tooth_key(tooth['患牙编号']),f,x) for f in fields for x in labels(tooth.get(f,[]))]
    else:
        for f in PERSON:
            out += [('patient',f,x) for x in labels(value['personLabel'].get(f,[]))]
        for tooth in value['toothLabel']:
            for f in set(FIRST+REPEAT):
                out += [(tooth_key(tooth['position']),f,x) for x in labels(tooth.get(f,[]))]
    return sorted({(tooth,field,option_key(label)) for tooth,field,label in out})
