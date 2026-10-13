"""Report follow-up fixes without overwriting the original negative ablations."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
from medagent.evaluation import evaluate
from medagent.format_repair import close_containers
from summarize_ablations import rule_rows
out=ROOT/'runtime/ablation';bundle=read_json(out/'pipeline_v2_results.json');review=read_json(out/'conservative_review_results.json');report={'tasks':{},'development_only':True}
for task,rows in bundle['rows'].items():
 variants={}
 if task=='root_canal':
  raw=read_json(out/'root_canal_contract_v2_predictions.json')
  variants['explicit_contract_raw']=evaluate(task,[{**r,'prediction':r['initial_prediction']} for r in raw],False)
  variants['explicit_contract_format_only']=evaluate(task,[{**r,'prediction':close_containers(task,r['initial_prediction'])['text']} for r in raw],False)
 variants['explicit_contract_repair']=evaluate(task,rows,False)
 scored=rule_rows(task,rows);variants['rules_and_conservative_review']=evaluate(task,scored,False)
 counts={'accepted':0,'without_evidence_gate':0,'without_review_gate':0,'total':len(rows),'factors':0,'supported_factors':0,'failed':0}
 accepted=[];expl={m:{'expected':0,'valid':0,'with_citations':0,'invalid':0} for m in ('full','no_rag','no_evidence')}
 for row,sc in zip(rows,scored):
  result=bundle['results'][task+':'+row['id']]
  if result.get('status')=='failed':counts['failed']+=1;continue
  codes=[x['code'] for x in result['issues']]
  if not codes:counts['accepted']+=1;accepted.append(sc)
  counts['without_evidence_gate']+=not any(c not in ('unsupported_factor','evidence_failed') for c in codes)
  counts['without_review_gate']+=not any(c!='unresolved_document_review' for c in codes)
  counts['factors']+=len(result['evidence']);counts['supported_factors']+=sum(e['supported'] for e in result['evidence'])
  for mode,answer in result['explanation_variants'].items():
   expl[mode]['expected']+=1;expl[mode]['invalid']+='error' in answer;expl[mode]['valid']+='error' not in answer;expl[mode]['with_citations']+=bool(answer.get('citations'))
 report['tasks'][task]={'variants':variants,'gate_and_evidence':counts,'accepted_metrics':evaluate(task,accepted,False),'explanation_variants':expl,
  'review_cases':len(review[task]),'review_approved':sum(r.get('approved',False) for r in review[task]),'format_repaired_cases':sum(r.get('format_repaired',False) for r in rows)}
report['retrieval']=read_json(out/'retrieval_results.json');report['routing_original']=read_json(out/'routing_results.json');report['routing_revised']=read_json(out/'routing_v2_results.json');report['images']=read_json(out/'image_results.json')
report['limitations']=['Follow-up fixes were selected after inspecting the same development cases, so improvements require a new independent validation set.',
 'Final pipeline is scored from cached real model responses; model repair responses were generated under the earlier feedback protocol, so do not infer exact live-call savings.',
 'Evidence substring and legal citation checks are not clinical validation.','Conservative text review does not change source facts; remaining conflict suggestions need professional adjudication.',
 'Routing benchmark includes one ambiguous request whose authored gold chooses both tasks, while current policy requests clarification; 9/10 is not a representative user accuracy.']
write_json(out/'followup_results.json',report)
def pct(x):return '—' if x is None else f'{100*x:.2f}%'
lines=['# 负向消融后的修正结果','', '以下为同一开发消融集上的修正实验，不能当作新的独立测试集结果。原始负向结果保留在 ablation_results.md。','']
for task,data in report['tasks'].items():
 lines+=['## '+('根管治疗' if task=='root_canal' else '牙隐裂'),'','| 配置 | 症状字段准确率 | 症状 F1 | 分级准确率 |','|---|---:|---:|---:|']
 for name,m in data['variants'].items():lines.append(f"| {name} | {pct(m['symptom_field_exact']['accuracy'])} | {pct(m['symptom_extraction']['micro_f1'])} | {pct(m['grade']['accuracy'])} |")
 c=data['gate_and_evidence'];lines+=['',f"根管括号闭合修复命中 {data['format_repaired_cases']} 例。保守文本审校通过 {data['review_approved']}/{data['review_cases']}，不改写原文。",f"证据逐字定位 {c['supported_factors']}/{c['factors']}；全链路自动通过 {c['accepted']}/{c['total']}；去掉证据门控可通过 {c['without_evidence_gate']}/{c['total']}；去掉文本审校门控可通过 {c['without_review_gate']}/{c['total']}。",'', '| 说明配置 | 有效输出 | 带合法引用 | 无效输出 |','|---|---:|---:|---:|']
 for mode,e in data['explanation_variants'].items():lines.append(f"| {mode} | {e['valid']}/{e['expected']} | {e['with_citations']} | {e['invalid']} |")
 lines+=['']
lines+=['## 检索、路由与图像','',json.dumps(report['retrieval']['summary'],ensure_ascii=False), '',f"人工构造的 10 条路由指令：原版本 {report['routing_original']['correct']}/10，修正后 {report['routing_revised']['correct']}/10。修复的是明确排除的专病仍被关键词选中；剩下的模糊请求会要求明确任务。",'','4 例图像全部完成转写，3 例审校输出结构有效；其中 2 例重复训练候选，且图像保留诊断栏。因此不汇报图像与纯文本的风险准确率差。','','## 限制','']+['- '+x for x in report['limitations']]
(ROOT/'docs/followup_results.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({t:{'gate':d['gate_and_evidence'],'variants':{k:{'symptom_accuracy':v['symptom_field_exact']['accuracy'],'symptom_f1':v['symptom_extraction']['micro_f1'],'grade':v['grade']['accuracy']} for k,v in d['variants'].items()}} for t,d in report['tasks'].items()},ensure_ascii=False,indent=2))
