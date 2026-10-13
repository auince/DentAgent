"""Select task-specific validated configurations, preserving all losing candidates."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
from medagent.evaluation import evaluate
from summarize_ablations import rule_rows
out=ROOT/'runtime/ablation';aligned=read_json(out/'pipeline_aligned_results.json');candidate=read_json(out/'pipeline_v2_results.json');report={'graph_version':'dental-v3','tasks':{},'selection':'root training-aligned prompt; crack structured-contract prompt; conservative text audit; fixed rules; evidence and RAG','selection_set':'same 20 root / 18 crack development cases; independent revalidation required'}
for task,bundle in [('root_canal',aligned),('crack',candidate)]:
 rows=bundle['rows'][task];scored=rule_rows(task,rows);accepted=[];counts={'total':len(rows),'accepted':0,'without_evidence_gate':0,'without_review_gate':0,'supported_factors':0,'factors':0,'failed':0};expl={m:{'expected':0,'valid':0,'with_citations':0,'invalid':0} for m in ('full','no_rag','no_evidence')}
 for row,score in zip(rows,scored):
  result=bundle['results'][task+':'+row['id']]
  if result.get('status')=='failed':counts['failed']+=1;continue
  codes=[i['code'] for i in result['issues']]
  if not codes:counts['accepted']+=1;accepted.append(score)
  counts['without_evidence_gate']+=not any(c not in ('unsupported_factor','evidence_failed') for c in codes)
  counts['without_review_gate']+=not any(c!='unresolved_document_review' for c in codes)
  counts['factors']+=len(result.get('evidence',[]));counts['supported_factors']+=sum(e['supported'] for e in result.get('evidence',[]))
  for mode,answer in result['explanation_variants'].items():
   expl[mode]['expected']+=1;expl[mode]['invalid']+='error'in answer;expl[mode]['valid']+='error'not in answer;expl[mode]['with_citations']+=bool(answer.get('citations'))
 report['tasks'][task]={'model_direct':evaluate(task,rows,False),'rules':evaluate(task,scored,False),'gate_and_evidence':counts,'accepted_metrics':evaluate(task,accepted,False),'explanations':expl}
write_json(out/'selected_pipeline_results.json',report)
def p(x):return '—' if x is None else f'{100*x:.2f}%'
lines=['# 当前发布配置的开发消融结果','','当前配置：LangGraph dental-v3。根管保留训练时提示词与输入协议；牙隐裂采用结构契约。两个专病共用保守文本审校、规则计算、证据定位和 RAG。候选方案失败与退步均保留在前序报告。','', '| 专病 | 模型直接症状字段准确率 | 症状 F1 | 模型直接分级 | 规则后分级 |','|---|---:|---:|---:|---:|']
for task,d in report['tasks'].items():
 m=d['model_direct'];lines.append(f"| {task} | {p(m['symptom_field_exact']['accuracy'])} | {p(m['symptom_extraction']['micro_f1'])} | {p(m['grade']['accuracy'])} | {p(d['rules']['grade']['accuracy'])} |")
lines+=['','根管 20 例是固定分层开发消融样本；牙隐裂为去重后 18 例。牙隐裂两列等级本来都由规则计算，不构成无规则消融。','']
for task,d in report['tasks'].items():
 c=d['gate_and_evidence'];lines += ['## '+task,'',f"原文逐字证据 {c['supported_factors']}/{c['factors']}。全部门控自动通过 {c['accepted']}/{c['total']}；去除证据门控通过 {c['without_evidence_gate']}/{c['total']}；去除文本审校门控通过 {c['without_review_gate']}/{c['total']}。",'','| 说明配置 | 有效结构 | 合法引用的说明 | 无效输出 |','|---|---:|---:|---:|']
 for mode,e in d['explanations'].items():lines.append(f"| {mode} | {e['valid']}/{e['expected']} | {e['with_citations']} | {e['invalid']} |")
 lines+=['']
lines+=['## 结论与边界','','- 选择根管训练协议优先考虑症状字段和输出协议兼容性：字段准确率 86.67% 高于候选契约的 81.67%，但规则分级 75% 低于候选的 80%（相差 1 例）。这不是所有指标最优的结论，需独立集合确认取舍。','- 严格原文引用仍是明显瓶颈：VLM 会改写标点、缩写、插入省略号，或无法证明复合因素；不把近似引文冒充逐字证据。','- 所有病例保留在完整准确率分母中；自动通过覆盖率与人工复核负担另列。','- 这些是开发集合上的选型结果，需要新病例独立验证，不能声称临床可用性已经获得专家认可。','- RAG 引用有效是工程约束，不是医学回答正确率。','- 本表用保存的真实模型输出组成模块对照；根管复用原协议单次预测，失败例没有追加在线重试，不能等同于完整默认 Agent 的逐例验收。在线真实 HTTP 链路另有两例验证。']
(ROOT/'docs/selected_pipeline_results.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(report,ensure_ascii=False))
