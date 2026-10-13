"""Short, evidence-linked routing stages shared by clinical and consultation graphs."""
from .consult_contracts import REGISTRY

NEEDS_PROMPT = '''你负责口腔咨询的评估需求分析，不执行工具、不确诊。输入是已经核对出处的facts和registry。
逐项检查症状的部位、阳性/阴性/未知/冲突状态，再检查专业评估所需信息。只返回短JSON：
{"candidates":[{"task":"专业ID","decision":"select|defer|exclude","fact_ids":["事实ID"],"reason":"一句依据","missing":[{"field":"duration|trigger|location|swelling|history|report","question":"一个关键问题"}]}]}。
select表示当前事实足以派发该专业咨询，不代表可以确诊；defer表示需先追问或提供检查，exclude表示只有否定证据或不适用。
牙髓/根尖周的咨询可基于自发痛、夜间痛等阳性主诉开展信息整理；未拍片或牙位含糊应列为缺失信息，不能单独成为拒绝派发该专业咨询的理由。
定点咬合锐痛足以支持疑似牙隐裂咨询，不能把看见裂纹或已确诊设为咨询的必要前提；只有非特异不适时仍需先追问。
不必枚举无关专业，尤其不要为了排除而列出所有专业。允许不同部位多个专业；阴性、未知不能单独构成select依据。不能以未记录作为阴性。
select/defer必须引用输入事实；exclude若仅表示当前没有该专业的阳性依据，可使用空fact_ids，但这仅表示不派发，绝不表示患者没有该病。
单纯牙痛属于牙髓/根尖周或龋病敏感咨询，不能据此选择根管难度。根管难度仅在临床记录提供当前治疗计划/操作或相关诊断的依据时select；仅既往根管史不足。
普通叩痛不能直接选牙隐裂；定点咬合/松口痛、裂损证据才能支持疑似裂损咨询。所有事实ID来自输入，禁止补造牙位或检查。
若患者明确询问牙隐裂风险且提供咬硬物等已发生的风险经历，可以选择牙隐裂Agent整理风险因素与检查缺口；明确标注这不是疑似裂纹的确认或确诊。仅风险经历而无相关咨询意图或特异表现时不要自动选择。
先识别患者真正需要的评估，不为填满专家名额派发所有专家。缺少信息时写关键缺口，不泛泛列检查清单。'''

ROUTE_PROMPT = '''你是口腔评估Leader。根据已定位的事实和needs选择需要执行的评估，只输出JSON：
{"tasks":["专业ID"],"reason":"简短决策依据","evidence":["原文短句"]}。
只能选择needs中decision=select的专业。明确排除的任务不选；不能默认所有专业全做。不同牙有独立需求时取并集。
若只有defer或exclude，返回空tasks，保留信息不足。不要输出诊断或等级。'''

FIELDS = {'duration','trigger','location','swelling','history','report'}

def needs_view(facts, registry=None, **context):
    return {'facts':facts,'registry':list((registry or REGISTRY).values()),**context}

def validate_needs(payload, facts, registry=None):
    allowed=registry or REGISTRY
    if not isinstance(payload,dict) or not isinstance(payload.get('candidates'),list):
        raise ValueError('invalid_needs_schema')
    known={f['id']:f for f in facts};seen=set();accepted=[]
    for raw in payload['candidates']:
        if not isinstance(raw,dict):raise ValueError('invalid_need')
        task=raw.get('task');decision=raw.get('decision');ids=raw.get('fact_ids')
        if task not in allowed or task in seen:raise ValueError('invalid_or_duplicate_need_task')
        if decision not in {'select','defer','exclude'}:raise ValueError('invalid_need_decision')
        if not isinstance(ids,list) or (not ids and decision!='exclude') or any(i not in known for i in ids):raise ValueError('need_unknown_fact')
        missing=raw.get('missing',[])
        if not isinstance(missing,list):raise ValueError('invalid_need_missing')
        for item in missing:
            if not isinstance(item,dict) or item.get('field') not in FIELDS or not isinstance(item.get('question'),str) or not item['question'].strip():
                raise ValueError('invalid_need_question')
        if not isinstance(raw.get('reason'),str) or not raw['reason'].strip():raise ValueError('missing_need_reason')
        item=dict(raw);item['missing']=missing
        if not ids:item['guard_reason']='no_positive_evidence_not_negative_diagnosis'
        positive=[known[i] for i in ids if known[i]['status']=='positive' and not known[i].get('conflicts_with')]
        if decision=='select' and not positive:
            item.update(decision='defer',guard_reason='no_uncontested_positive_fact')
        if task=='root_canal' and item['decision']=='select' and not any(f.get('source_kind')=='clinical_record' for f in positive):
            item.update(decision='defer',guard_reason='clinical_record_required')
            if not any(m['field']=='report' for m in missing):
                item['missing']=missing+[{'field':'report','question':'能提供当前治疗计划或相关检查记录吗？'}]
        accepted.append(item);seen.add(task)
    return {'candidates':accepted,'validation':'schema_source_ids_and_prerequisites_only'}

def selected_tasks(needs):
    return {c['task'] for c in needs['candidates'] if c['decision']=='select'}

def task_scopes(needs, facts, tasks):
    known={f['id']:f for f in facts};scopes={}
    for c in needs['candidates']:
        if c['task'] not in tasks:continue
        teeth={str(known[i]['tooth'])+'牙' for i in c['fact_ids'] if known[i].get('tooth')}
        if teeth:scopes[c['task']]=sorted(teeth)
    return scopes
