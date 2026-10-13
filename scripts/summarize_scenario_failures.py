"""Post-run accounting for every scenario; no subset selection or new inference."""
import collections,json,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
from medagent.evaluation import evaluate

out=ROOT/'runtime/scenario_eval_v1';metrics=read_json(out/'metrics.json')
assert metrics['completed']==60, 'Final analysis requires all 60 cases'
reasons=collections.Counter();examples=collections.defaultdict(list);issues=collections.Counter();groups={}
fixtures={c['id']:c for c in read_json(ROOT/'scenarios/implicit_v1/cases.json')}
task_diagnostics={t:{'expected':0,'omitted':[],'failed':[],'symptoms_wrong_when_returned':[],
                      'system_grade_wrong_when_returned':[],'returned_rows':[]} for t in ('root_canal','crack')}
status_counts=collections.Counter();route_rejected=[];trace_gaps=[];evidence_counts=collections.Counter();model_disagreements=[];evidence_failures=[]
def mark(kind,sid):
    reasons[kind]+=1;examples[kind].append(sid)
for c in metrics['cases']:
    r=read_json(out/'reports'/(c['id']+'.json'));sid=c['id']
    status_counts[r['status']]+=1
    if (r.get('error') or '').startswith('intent_routing_failed:'):route_rejected.append(sid)
    if not c['llm_route_used']:trace_gaps.append(sid)
    if not c['route_exact']:mark('route_miss_or_extra',sid)
    if not c['stages_reached']:mark('incomplete_expected_workflow',sid)
    if not c['specialties_returned']:mark('required_specialty_failed_or_missing',sid)
    if not c['grade_exact']:mark('grade_mismatch',sid)
    if not c['symptom_fields_exact']:mark('symptom_field_mismatch',sid)
    if not c['explanations_valid']:mark('explanation_failed_or_missing',sid)
    for task,v in r.get('results',{}).items():
        issues.update(i['code'] for i in v.get('issues',[]))
        for issue in v.get('issues',[]):
            if issue['code']=='model_rule_disagreement':model_disagreements.append({'case':sid,'task':task,**issue})
            if issue['code']=='evidence_failed':evidence_failures.append({'case':sid,'task':task,**issue})
        for evidence in v.get('evidence',[]):
            evidence_counts['total']+=1
            evidence_counts['anchored_supported' if evidence.get('supported') else 'not_anchored_or_unsupported']+=1
        error=v.get('error')
        if error:
            kind='model_endpoint_error' if any(x in error for x in ('HTTP ','unavailable','truncated')) else 'output_schema_exhausted'
            mark(kind,sid)
    for task in c['expected_tasks']:
        d=task_diagnostics[task];d['expected']+=1
        if task not in r.get('tasks',[]):d['omitted'].append(sid);continue
        v=r.get('results',{}).get(task,{})
        if v.get('status') not in ('assessed','needs_review'):d['failed'].append(sid);continue
        row={'id':sid,'gold':fixtures[sid]['gold'][task],'prediction':v['factors']}
        raw=evaluate(task,[row],False)
        if raw['symptom_field_exact']['accuracy']!=1:d['symptoms_wrong_when_returned'].append(sid)
        expected={x['tooth']:x['grade'] for x in fixtures[sid]['expected_assessments'][task]}
        actual={x['tooth']:x['grade'] for x in v.get('assessments',[])}
        if expected!=actual:d['system_grade_wrong_when_returned'].append(sid)
        d['returned_rows'].append(row)
for modality in ('text','image_only'):
    rows=[c for c in metrics['cases'] if c['modality']==modality]
    groups[modality]={'n':len(rows),**{k:sum(c[k] for c in rows) for k in metrics['counts']}}
for tasks in (('root_canal',),('crack',),('crack','root_canal')):
    rows=[c for c in metrics['cases'] if tuple(c['expected_tasks'])==tasks]
    groups['+'.join(tasks)]={'n':len(rows),**{k:sum(c[k] for c in rows) for k in metrics['counts']}}
times=sorted(c['elapsed_seconds'] for c in metrics['cases'] if isinstance(c.get('elapsed_seconds'),(int,float)))
latency={'n':len(times),'median_seconds':statistics.median(times),'mean_seconds':statistics.mean(times),
         'p95_seconds':times[max(0,int(.95*len(times)+.999)-1)],'max_seconds':max(times)}
for task,d in task_diagnostics.items():
    rows=d.pop('returned_rows');d['returned_count']=len(rows)
    d['returned_metrics_diagnostic_only']=evaluate(task,rows,False)
result={'total':60,'groups':groups,'failure_reasons':dict(reasons),'failure_cases':dict(examples),
        'issue_codes':dict(issues),'latency':latency,'statuses':dict(status_counts),
        'route_rejected_cases':route_rejected,'recognized_route_trace_absent':trace_gaps,
        'evidence_anchor_counts':dict(evidence_counts),'task_diagnostics':task_diagnostics,
        'model_rule_disagreements':model_disagreements,'evidence_failures':evidence_failures,
        'grade_metric_note':'Main root grade is system rule recomputation, although shared evaluation helper labels its mode model_grade_vs_stored_gold. Returned-only root diagnostic retains raw model difficulty; it is conditional and must not replace full 40-case denominators.'}
write_json(out/'analysis.json',result)
lines=['# 60 场景全流程测试分析','','所有场景已结束。分组仅为同一完整 60 例的分层报告，不另抽样；路由漏项、输出失败均保留。',
       '', 'root_canal 为根管治疗，crack 为牙隐裂，image_only 为纯图片输入。','',
       '| 输入/任务类型 | 病例数 | 路由完全正确 | 到达完整所需流程 | 严格整例正确 | 自动通过 |','|---|---:|---:|---:|---:|---:|']
for name,c in groups.items():lines.append(f"| {name} | {c['n']} | {c['route_exact']} | {c['stages_reached']} | {c['end_to_end_correct']} | {c['automatic_pass']} |")
lines+=['','## 错误分布','','同一病例可有多类错误，不能将计数相加理解为独立病例数。','', '| 问题 | 次数 | 病例编号 |','|---|---:|---|']
labels={'route_miss_or_extra':'任务路由遗漏或多报','incomplete_expected_workflow':'未完成全部所需流程',
        'required_specialty_failed_or_missing':'所需专病分支失败或遗漏','grade_mismatch':'整例等级未全部匹配（含漏路由）',
        'symptom_field_mismatch':'整例症状字段未全部匹配（含漏路由）','explanation_failed_or_missing':'说明失败或遗漏',
        'model_endpoint_error':'模型接口错误','output_schema_exhausted':'标签/结构重试耗尽'}
for name,n in reasons.items():lines.append(f"| {labels[name]} | {n} | {', '.join(examples[name])} |")
details={
    'S026':('图片','47 牙单点窄深袋 7 mm + J 型透射影','OCR 无字符错误；Leader 拒绝路由，未执行牙隐裂评估。拒绝的原始模型决策没有完整保留。'),
    'S027':('文本','27 牙窄深袋 6 mm + 双侧骨吸收阴影','Leader 拒绝路由；无明确牙髓症状不等于无需进行裂损风险评估。此处预期仍按项目场景规则定义。'),
    'S035':('文本','37 牙 CBCT 根折裂线 + 牙根断片移位','正确路由到牙隐裂；两次提取均返回结构破坏模块的非法枚举，分支失败。'),
    'S043':('文本/双任务','16 牙根管处置；26 牙单点深袋 + J 型影','Leader 决策只引用 16 牙，遗漏另一牙的牙隐裂分支。'),
    'S044':('图片/双任务','21 牙根管处置；16 牙深袋 + 双侧骨吸收','Leader 只路由根管；根管一次非法标签重试后修复，但仍缺少牙隐裂分支。'),
    'S047':('文本/双任务','31 牙根管处置；27 牙咬合不适 + 电活力异常','Leader 只引用明确诊断和处置的 31 牙；另一牙尚无确定诊断的风险评估需求遗漏。'),
    'S052':('图片/双任务','46 牙根管处置；36 牙 CBCT 根折裂线 + 断片移位','Leader 正确派发两任务；牙隐裂两次非法枚举失败，根管继续完成并由 Leader 汇总待复核。'),
    'S057':('文本/双任务','25 牙根管处置；27 牙咬合无力 + 夜磨牙，未见裂纹','Leader 只路由根管；未识别另一牙应进行低置信度裂损风险评估。没有可见裂纹也可能需要排查，但不等于已经存在隐裂。')}
lines+=['','## 八个未联合达标病例逐例复盘','','| 病例 | 输入 | 原文关键线索 | 实际失败位置 |','|---|---|---|---|']
for sid,(modality,source,cause) in details.items():
    lines.append(f"| [{sid}](../runtime/scenario_eval_v1/reports/{sid}.json) | {modality} | {source} | {cause} |")
lines+=['','## 路由漏项与专病预测错误分开统计','','主结果始终保留全部应执行任务，根管、牙隐裂各 40 例；下面的已返回统计仅用于定位错误发生在哪一步，不替换主结果分母。',
        '', '| 专病 | 应执行 | 路由遗漏 | 分支失败 | 已返回中的症状错误 | 已返回中的系统分级错误 |','|---|---:|---:|---:|---:|---:|']
for task,d in task_diagnostics.items():
    lines.append(f"| {task} | {d['expected']} | {len(d['omitted'])} | {len(d['failed'])} | {len(d['symptoms_wrong_when_returned'])} | {len(d['system_grade_wrong_when_returned'])} |")
lines+=['',f"运行状态：{dict(status_counts)}。因素证据定位统计：{dict(evidence_counts)}。",
        '', '根管主报表使用系统根据因素复算的难度，不能称为模型直接分级准确率。共享评测函数中的 mode 标签未体现调用前已替换 difficulty，以上说明以实际计算代码为准。',
        '', 'unsupported_factor 只说明模型未返回可逐字定位且标为支持的原文证据，不能直接等价为因素内容错误。输出中的空 quote 已丢弃原始无效片段，无法仅据现有报告继续区分漏引、改写和否定判断。',
        '', f"路由拒绝病例：{', '.join(route_rejected) or '无'}。没有 recognized 路由事件的病例：{', '.join(trace_gaps) or '无'}。recognized 事件缺失不能证明模型未调用；拒绝分支的原始决策没有完整落入轨迹，是当前日志的限制。"]
raw_root=task_diagnostics['root_canal']['returned_metrics_diagnostic_only']['grade']
lines+=['','## 根管直接分级与规则复算','','| 分级来源 | 正确 / 总数 | 准确率 |','|---|---:|---:|',
        f"| 模型原始 difficulty | {raw_root['correct']}/{raw_root['total']} | {100*raw_root['accuracy']:.2f}% |",
        f"| 系统根据提取因素复算 | {metrics['task_metrics']['root_canal']['grade']['correct']}/40 | {100*metrics['task_metrics']['root_canal']['grade']['accuracy']:.2f}% |",'',
        '本轮 40 个根管应执行任务均返回结果，因此此表没有删掉遗漏病例。根管症状字段为诊断、牙周病损、根管感染程度；不能将症状字段正确称为全部因素均正确。',
        '', '| 病例 | 牙位 | 模型等级 | 规则等级 |','|---|---|---:|---:|']
for v in model_disagreements:lines.append(f"| {v['case']} | {v['tooth']} | {v['model']} | {v['rule']} |")
lines+=['','## 实际接口耗时','',f"中位数 {latency['median_seconds']:.1f} 秒；平均 {latency['mean_seconds']:.1f} 秒；P95 {latency['p95_seconds']:.1f} 秒。含内部工具等待、模型切换和必要的反馈重试。",'',
        '## 解释边界','','症状和等级正确、流程完整与自动通过分别统计。证据缺失等问题会要求复核，并不自动证明模型等级错误。数据来自自制病历，缺少独立临床专家盲评，不能直接替代真实病历的全量测试指标。',
        '', '严格整例正确包含结构及引用 ID 校验，不包含医学语义盲评。S041 的证据解释将 11 牙称为第一前磨牙，最终说明还声称未支持因素未计入评分，而规则计算实际使用了全部提取因素。这说明即使等级与标签一致、引用合法，解释仍可能不可信；该指标不能包装为临床回答正确率。',
        '', f"end_to_end_correct 也不要求所有因素证据通过审核。S032 的证据模型返回非法或重复 index，产生 evidence_failed，但症状及等级正确、解释与汇总可返回，因此计入联合指标并标为需复核。证据阶段失败记录：{evidence_failures}。自动通过指标单独要求系统没有这些复核问题。",
        '', '本轮审校是否真正改文及触发回流应以 metrics 中 changed_source/rerouted 的实际计数为准；若均为零，只能报告工程测试覆盖了该路径，不能宣称本轮真实病例已验证更正后的重新推理。',
        '', '## 后续优先事项','','1. Leader 对每颗患牙分别识别支持的评估需求，特别是没有明确裂纹诊断、但有深牙周袋与特征性影像的记录。当前规则检索在专病分支中执行，发生漏路由时相应规则检索也不会运行；可把有来源的专病适用规则提供给路由阶段。后续应采用独立数据验证，不能以本轮标签直接修补结果。',
        '2. 保留被拒绝路由的原始模型决策、证据核验的原始引文与拒绝原因，让复核能够区分定位格式错误和临床证据不足。',
        '3. 对评分是否使用未支持因素、解释是否改写患牙身份或声称错误计算过程增加一致性约束。',
        '4. 引入独立医生复核、真实手机拍摄与手写样本，再评估医学语义和部署性能。']
(ROOT/'docs/scenario_analysis.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(result,ensure_ascii=False))
