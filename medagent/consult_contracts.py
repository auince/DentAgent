"""Versioned contracts shared by serving, augmentation, rewards and evaluation."""
import json
import re
from .common import ROOT, read_json, digest

REGISTRY = {a['id']: a for a in read_json(ROOT/'configs/specialties.json')['agents'] if a.get('enabled',True)}
FACT_PROMPT = '''将患者原话整理成JSON：{"facts":[{"name":"症状或因素","value":"原文信息","status":"positive|negative|unknown|conflict","location":null,"tooth":null,"quote":"原文连续引文","source_id":"消息ID"}],"intent":"咨询意图"}。
只提取可观察事实；未说的内容不补全；部位含糊时保留原话，不能猜FDI牙位；否定、未知、冲突分别保留。所有quote必须来自给定消息。临床检查只来自明确提交的检查记录。无长推理。'''
LEADER_PROMPT = '''你是口腔咨询Leader。根据事实、已有结果、剩余预算决定下一步，仅输出一个JSON动作。
动作：{"action":"ask","field":"duration|trigger|location|swelling|history|report","question":"一个关键问题"}；{"action":"retrieve","task":"专业ID或all","query":"检索词"}；{"action":"dispatch","tasks":["专业ID"],"independent_problems":false}；{"action":"review_image"}；{"action":"finish"}；{"action":"refer","reason":"就诊原因"}。
允许多专业。先阅读needs中的评估依据与缺口；dispatch只能选择decision=select的专业，defer时先问关键缺口。派发前完成影响评估的关键追问，再检索适用知识并派发；不要先派发再用普通追问清空已得结果。检索只在确有知识缺口时调用，避免重复。
ask先选field再写与该field对应的问题；不得将已经问过的诱因问题换成其它field重复问。部位用location、诱因/咬合/冷热用trigger、肿胀或呼吸吞咽用swelling、既往治疗/外伤用history、检查报告用report、时间用duration。不要用report去询问牙位，不要用trigger去询问上下颌。
缺少片子或确切牙位仍可进行牙髓相关咨询，保留缺口；不以获得确诊作为派发咨询专家的前提。根据专家结果列出的关键缺口进行必要追问，已有信息或已回答未知的字段不反复询问。
如果输入urgent为true，立即选择refer，不继续普通追问或工具调用。
结束前逐项核对：needs中select的专业是否都有当前事实版本下的成功结果，影响咨询的关键missing是否已取得回答或已明确未知。尚未覆盖的专业继续派发，不因已有一个专家结果就遗漏其它需求；必要时拆成两次派发，并遵守剩余工具预算。先完成关键追问，再获取专家结果，避免追问更新事实后直接沿用旧结果。每次追问仅针对一个field，不能把部位与诱因合并成同一问题。
历史记忆仅作追问背景，未经本次患者确认不能转成当前症状、检查或诊断。禁止补造临床检查。单纯牙痛优先牙髓咨询，不能直接给根管难度。已有结果充分就结束，不重复调用。工具失败可换合理路径。registry定义职责。'''
EXPERT_PROMPT = '''你是{specialty}专病Agent。职责：{scope}。
只根据facts、sources和knowledge生成JSON：{{"facts":["保留本专业相关的原始事实对象"],"findings":[{{"fact_ids":["事实ID"],"interpretation":"短说明"}}],"directions":[{{"text":"可能方向而非确诊","evidence_ids":["知识片段ID"]}}],"missing_information":["待补充内容"]}}。
患者未知因素保持未知。至少整理一条原文支持的因素及短说明；没有足够信息时明确缺口。knowledge为空时directions必须为空，不提出缺少知识引用的疾病方向。任何解释都不能补写检查结果；多牙事实必须分开。禁止处方。只输出短证据关系，无长CoT。'''
ANSWER_PROMPT = '''基于给定患者事实、专家结果和知识，输出JSON：{"answer":"简短中文咨询回答，包含已知依据、可能方向、缺失信息和下一步","citations":["知识片段ID"]}。仅引用输入中存在且支持结论的知识。不得补写检查、牙位、确诊或处方。专家没有可靠结果时说明信息不足。'''
VISION_PROMPT = '''忠实转写病历图片，仅输出JSON：{"text":"原文","uncertain_regions":["无法辨认的区域"]}。保留牙位、正负号、否定词、数值单位；无法辨认写[无法辨认]，禁止根据医学常识补字。'''

def visible_messages(messages, recent=4):
    """Retain the opening complaint/intent alongside the bounded recent exchange."""
    tail=list(messages[-recent:])
    opening=next((m for m in messages if m.get('role')=='user'),None)
    return ([opening]+tail) if opening is not None and opening not in tail else tail

def policy_view(state, config, observations=None):
    """Bound policy inputs without truncating patient facts or expert evidence links."""
    view={k:state[k] for k in ('facts','needs','question_count','tool_count','asked_questions')}
    view['asked_fields']=state.get('asked_fields',[])
    view['urgent']=state.get('urgent',False)
    view['historical_memory']=state.get('historical_memory',[])
    view['memory_policy']='历史记忆不是本次症状。需患者重新确认后才能生成本次facts；不能当作当前检查或确诊。'
    view['recent_messages']=visible_messages(state.get('messages',[]))
    view['experts']={task:{k:v for k,v in result.items() if k!='facts'} for task,result in state['experts'].items()}
    view['knowledge']=[{k:v for k,v in chunk.items() if k in {'id','source','text','url'}} for chunk in state['knowledge']]
    obs=observations if observations is not None else state.get('observations',[])
    view['observations']=[{k:v for k,v in o.items() if k not in {'results'}} for o in obs]
    view['registry']=[{k:v for k,v in a.items() if k in {'id','name','scope','tools'}} for a in REGISTRY.values()]
    view['limits']={k:config[k] for k in ('max_actions','max_tools','max_questions','max_experts','max_experts_independent')}
    return view

def validate_facts(payload, sources):
    if not isinstance(payload, dict) or not isinstance(payload.get('facts'), list):
        raise ValueError('invalid_fact_schema')
    accepted, issues = [], []
    for raw in payload['facts']:
        if not isinstance(raw, dict):
            issues.append('invalid_fact'); continue
        f = dict(raw)
        source = sources.get(f.get('source_id'))
        quote = f.get('quote')
        if not source or not isinstance(quote, str) or not quote or quote not in source['text']:
            issues.append('unlocated_fact'); continue
        if f.get('status') not in {'positive','negative','unknown','conflict'} or not isinstance(f.get('name'), str) or not f['name']:
            issues.append('invalid_fact_state'); continue
        if f.get('tooth') is not None:
            tooth = str(f['tooth'])
            if not re.fullmatch('[1-4][1-8]', tooth) or not re.search(r'(?<!\d)'+tooth+r'(?!\d)', quote):
                issues.append('unsupported_tooth'); f['tooth'] = None
        f['start'] = source['text'].index(quote)
        f['end'] = f['start'] + len(quote)
        f['source_kind'] = source['kind']
        f['id'] = digest(json.dumps(f, ensure_ascii=False, sort_keys=True))[:20]
        # A located quote is syntactic support, not an independent medical judgment.
        f['validation'] = 'source_span_only'
        accepted.append(f)
    return accepted, issues

def merge_facts(previous, incoming):
    result = {f['id']: dict(f) for f in previous}
    for f in incoming:
        f = dict(f)
        for old in result.values():
            if (old['name'], old.get('tooth'), old.get('location')) == (f['name'], f.get('tooth'), f.get('location')) and old['status'] != f['status'] and old['status'] in {'positive','negative'} and f['status'] in {'positive','negative'}:
                old['conflicts_with'] = list(set(old.get('conflicts_with', []) + [f['id']]))
                f['conflicts_with'] = list(set(f.get('conflicts_with', []) + [old['id']]))
        result[f['id']] = f
    return list(result.values())

def pending_experts(state):
    from .routing import selected_tasks
    selected=selected_tasks(state.get('needs',{'candidates':[]}))
    completed={task for task,result in state.get('experts',{}).items() if result.get('status')!='failed'}
    return selected-completed

def finish_issue(state,cfg):
    """Visible policy consistency only; never reads hidden evaluation labels."""
    if state.get('needs',{}).get('status')=='failed':
        return 'finish_without_valid_needs_analysis'
    pending=pending_experts(state)
    if pending and state['tool_count']<cfg['max_tools']:
        return 'finish_with_pending_selected_experts:'+','.join(sorted(pending))
    return None

def validate_action(action, state, cfg):
    if not isinstance(action, dict): raise ValueError('action_not_object')
    kind = action.get('action')
    if kind not in {'ask','retrieve','dispatch','review_image','finish','refer'}: raise ValueError('unknown_action')
    if kind=='finish':
        issue=finish_issue(state,cfg)
        if issue:raise ValueError(issue)
    if kind == 'ask':
        if action.get('field') not in {'duration','trigger','location','swelling','history','report'}:raise ValueError('invalid_question_field')
        q = action.get('question')
        if not isinstance(q,str) or not q.strip() or len(q)>300: raise ValueError('invalid_question')
        mismatch=question_field_issue(action)
        if mismatch:raise ValueError(mismatch)
        if state['question_count'] >= cfg['max_questions']: raise ValueError('question_budget_exhausted')
        if action['field'] in state.get('asked_fields',[]):raise ValueError('repeated_question_field')
        if q in state.get('asked_questions',[]): raise ValueError('repeated_question')
    if kind == 'dispatch':
        tasks = action.get('tasks')
        if not isinstance(tasks,list) or not tasks or any(not isinstance(t,str) or t not in REGISTRY for t in tasks) or len(tasks)!=len(set(tasks)): raise ValueError('invalid_expert_set')
        if 'needs' in state:
            from .routing import selected_tasks
            if not set(tasks)<=selected_tasks(state['needs']):raise ValueError('dispatch_without_supported_need')
        maximum = cfg['max_experts_independent'] if action.get('independent_problems') is True else cfg['max_experts']
        if len(set(state.get('experts',{})) | set(tasks))>maximum: raise ValueError('expert_budget_exhausted')
        if any(t in state.get('experts',{}) and state['experts'][t].get('status')!='failed' for t in tasks): raise ValueError('repeated_expert')
    cost = len(action['tasks']) if kind=='dispatch' else int(kind in {'retrieve','review_image'})
    if state['tool_count']+cost > cfg['max_tools']: raise ValueError('tool_budget_exhausted')
    if kind=='retrieve':
        if action.get('task') not in {*REGISTRY,'all'} or not isinstance(action.get('query'),str) or not action['query'].strip(): raise ValueError('invalid_retrieval')
    if kind=='review_image' and (not state.get('images') or state.get('image_reviewed')): raise ValueError('image_not_available_or_already_reviewed')
    if kind=='refer' and not isinstance(action.get('reason'),str): raise ValueError('invalid_referral')
    if state.get('urgent') and kind!='refer':raise ValueError('urgent_referral_required')
    return cost

QUESTION_HINTS={
    'duration':('多久','多长','何时','什么时候','几天','几周','几个月','持续','时长','开始','疼一会','痛一会','马上就好','立即消失','多久缓解'),
    'trigger':('诱因','什么情况','咬','冷','热','刺激','松口','松开口','轻触','触碰','按压','自发','夜间','半夜'),
    'location':('哪颗','哪一颗','哪里','哪边','哪侧','哪个位置','左边','右边','左侧','右侧','上牙','下牙','牙位','部位','位置','门牙','前牙','后牙','中切牙','侧切牙','上面','下面','上颌','下颌','同一颗','同一部位','特定咬合点','整个牙冠','疼痛集中'),
    'swelling':('肿','鼓包','呼吸','吞咽','喘'),
    'history':('以前','既往','曾','之前','外伤','补过','根管','史','自发','夜间'),
    'report':('报告','检查','影像','拍片','牙医','医生','X光','X线','CT','裂纹')}

def question_field_issue(action):
    if action.get('action')!='ask':return None
    question=action.get('question','');field=action.get('field')
    if not isinstance(question,str) or field not in QUESTION_HINTS:return None
    matches={k for k,words in QUESTION_HINTS.items() if any(w in question for w in words)}
    if matches and field not in matches:return 'question_field_mismatch:'+field+':'+','.join(sorted(matches))
    return None
