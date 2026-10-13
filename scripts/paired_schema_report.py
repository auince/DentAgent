"""Paired model comparison under an identical explicit schema on frozen development cases."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
from medagent.evaluation import evaluate
from medagent.format_repair import close_containers
from ablation_worker import OUT,records,select
from summarize_ablations import rule_rows
from schema_adapter_audit import adapt
base=records('root_canal_contract_v2_base');fine=records('root_canal_contract_v2');cases={r['id']:r for r in select()['root_canal']}
assert len(base)==len(fine)==len(cases)==20
requests={r['key']:r for r in base}
for r in fine:
 assert all(r[k]==requests[r['key']][k] for k in ('system','text','images')),'prompt_mismatch'
report={'cases':20,'prompt_identical':True,'subset':'frozen stratified development ablation cases','models':{}}
for variant,items in [('base',base),('finetuned',fine)]:
 raw=[{**cases[r['key']],'prediction':r['answer']} for r in items]
 formatted=[{**r,'prediction':close_containers('root_canal',r['prediction'])['text']} for r in raw]
 canonical=[]
 for r in formatted:
  try:pred,_=adapt(r['prediction'])
  except (ValueError,KeyError,TypeError):pred=''
  canonical.append({**r,'prediction':pred})
 report['models'][variant]={'field_aliases':evaluate('root_canal',canonical,False),'raw':evaluate('root_canal',raw,False),'format_only':evaluate('root_canal',formatted,False),'format_plus_rules':evaluate('root_canal',rule_rows('root_canal',formatted),False)}
report['limitations']=['20 stratified development cases; do not combine with the 101-case benchmark.','No model retry is used in this paired table; only identical deterministic postprocessing.','The schema was revised after the original prompt ablation exposed format problems; this is development evidence, not independent validation.']
write_json(OUT/'paired_schema_results.json',report)
lines=['# 相同显式 Schema 下的基座对照','','根管固定 20 例开发消融集，逐例核验两模型 system/user 输入完全一致。两边都不开模型纠错重试，分别比较原输出、相同语法闭合、相同规则重算。','', '| 模型 | 处理 | 症状字段准确率 | 症状 F1 | 分级准确率 |','|---|---|---:|---:|---:|']
for variant,results in report['models'].items():
 for mode,m in results.items():
  def p(x):return '—' if x is None else f'{100*x:.2f}%'
  lines.append(f"| {variant} | {mode} | {p(m['symptom_field_exact']['accuracy'])} | {p(m['symptom_extraction']['micro_f1'])} | {p(m['grade']['accuracy'])} |")
lines+=['']+['- '+x for x in report['limitations']]
(ROOT/'docs/paired_schema_results.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(report,ensure_ascii=False))
