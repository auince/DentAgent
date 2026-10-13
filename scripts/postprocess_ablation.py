"""Full-test deterministic postprocessing ablation on frozen model predictions."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,write_json
from medagent.evaluation import evaluate
from medagent.format_repair import close_containers
from summarize_ablations import rule_rows
report={}
for task in ('root_canal','crack'):
 report[task]={}
 for variant in ('base','finetuned'):
  rows=[json.loads(s) for s in (ROOT/'runtime/evaluation'/(task+'_'+variant+'.jsonl')).read_text().splitlines()]
  rows=[r for r in rows if not r['train_text_overlap'] and not r['train_id_overlap'] and not r['duplicate_test']]
  repaired=[];hits=0
  for row in rows:
   # Budget-truncated completions stay failures, even if a valid prefix can be found.
   result=close_containers(task,row['prediction']);hits+=result['repaired'];repaired.append({**row,'prediction':result['text']})
  metrics={'original':evaluate(task,rows,False),'format_only':evaluate(task,repaired,False)}
  if task=='root_canal':metrics['format_plus_rules']=evaluate(task,rule_rows(task,repaired),False)
  report[task][variant]={'repaired_cases':hits,'metrics':metrics}
write_json(ROOT/'runtime/ablation/full_postprocess_results.json',report)
lines=['# 完整测试集上的确定性模块消融','','保持模型权重、原提示词及已生成预测不变，仅对格式闭合和根管规则重算作对照。截断输出始终计失败，不补造缺失标签。根管 101 例，牙隐裂去重后 18 例。','']
for task,variants in report.items():
 lines+=['## '+task,'','| 模型 | 处理 | 症状字段准确率 | 症状 F1 | 分级准确率 |','|---|---|---:|---:|---:|']
 for variant,data in variants.items():
  for mode,m in data['metrics'].items():
   def p(v):return '—' if v is None else f'{v*100:.2f}%'
   lines.append(f"| {variant} | {mode} | {p(m['symptom_field_exact']['accuracy'])} | {p(m['symptom_extraction']['micro_f1'])} | {p(m['grade']['accuracy'])} |")
 lines+=['']
lines+=['本表仍使用原项目提示词，不能当作当前完整 Agent 系统在 101 例上的结果。规则正确执行也不等于抽取因素医学上正确。']
(ROOT/'docs/full_postprocess_results.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(report,ensure_ascii=False))
