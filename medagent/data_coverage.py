"""Count independent cases separately from views and gate specialty training."""
from collections import Counter, defaultdict
from .common import ROOT, read_json, write_json, digest
from .consult_contracts import REGISTRY

def specialty_coverage(rows):
    report={t:{'rows':0,'groups':set(),'legacy_rows':0,'synthetic_rows':0} for t in REGISTRY}
    for row in rows:
        task=row.get('specialty') or row.get('legacy_task') or row.get('task')
        if task not in report:raise ValueError('row_missing_specialty:'+str(row.get('id')))
        item=report[task];item['rows']+=1;item['groups'].add(row['group_id'])
        item['legacy_rows' if row.get('legacy_task') else 'synthetic_rows']+=1
    return {t:{**v,'groups':len(v['groups'])} for t,v in report.items()}

def audit_specialist_dataset(data, minimums=None):
    minimums=minimums or read_json(ROOT/'configs/small_training.json')['specialty_minimums']
    report={'clinical_validation':False,'splits':{},'failures':[]}
    seen={}
    for split in ('train','validation'):
        path=data/(split+'.json')
        if not path.exists():raise ValueError('missing_specialist_split:'+split)
        rows=read_json(path);coverage=specialty_coverage(rows)
        report['splits'][split]={'sha256':digest(path.read_bytes()),'count':len(rows),'specialties':coverage}
        seen[split]={r['group_id'] for r in rows}
        for task,counts in coverage.items():
            if counts['groups']<minimums[split]:report['failures'].append(f'{split}:{task}:independent_groups_below_{minimums[split]}')
    if seen['train']&seen['validation']:report['failures'].append('group_leakage')
    testdir=data.parent/'cards/test'
    testgroups={read_json(p)['group_id'] for p in testdir.glob('*.json')}
    if (seen['train']|seen['validation'])&testgroups:report['failures'].append('test_group_leakage')
    report['status']='passed' if not report['failures'] else 'blocked'
    write_json(data/'coverage.json',report)
    if report['failures']:raise ValueError('specialty_coverage_gate:'+','.join(report['failures']))
    return report
