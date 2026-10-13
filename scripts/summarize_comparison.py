"""Recompute task metrics and separately report strict full-output schema gating."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json,digest
from medagent.evaluation import evaluate

out=ROOT/'runtime/evaluation'
report={'dataset_audit':read_json(out/'dataset_audit.json'),'results':{},
    'metric_note':'Task-wise scoring: unrelated schema errors do not erase symptom or root-grade results. Strict whole-output metrics are separately reported. Invalid crack rule options yield unknown grades.',
    'evaluator_sha256':digest((ROOT/'medagent/evaluation.py').read_bytes())}
for task,cfg in read_json(ROOT/'configs/experiments.json')['tasks'].items():
    results={}
    for variant in ('base','finetuned'):
        rows=[json.loads(x) for x in (out/(task+'_'+variant+'.jsonl')).read_text().splitlines()]
        assert len(rows)==len(read_json(cfg['test'])),(task,variant,'incomplete')
        assert len({r['id'] for r in rows})==len(rows),'duplicate_ids'
        clean=[r for r in rows if not r['train_text_overlap'] and not r['train_id_overlap'] and not r['duplicate_test']]
        results[variant]={}
        for name,subset in [('all',rows),('decontaminated',clean)]:
            results[variant][name]=evaluate(task,subset,strict_schema=False)
            results[variant]['strict_'+name]=evaluate(task,subset,strict_schema=True)
    for subset in ('all','decontaminated'):
        a,b=results['base'][subset],results['finetuned'][subset]
        def delta(x,y):return 100*(y-x) if x is not None and y is not None else None
        results[subset+'_delta_pp']={'symptom_micro_f1':delta(a['symptom_extraction']['micro_f1'],b['symptom_extraction']['micro_f1']),
            'symptom_field_accuracy':delta(a['symptom_field_exact']['accuracy'],b['symptom_field_exact']['accuracy']),
            'grade_accuracy':delta(a['grade']['accuracy'],b['grade']['accuracy'])}
    report['results'][task]=results
write_json(out/'comparison.json',report)
print(json.dumps(report,ensure_ascii=False,indent=2))
