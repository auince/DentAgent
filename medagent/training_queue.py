"""Dependency queue; disk artifacts, not scheduled names, establish completion."""
import time
import os
from pathlib import Path
from .common import ROOT,read_json,write_json,digest
from .gpu import allowed_indices

QUEUE=ROOT/'configs/training_queue.json'
STATUS=ROOT/'runtime/small_model/queue/status.json'

def external_jobs():
    live={}
    for path in (STATUS.parent/'external').glob('*.json'):
        item=read_json(path)
        if item.get('status') not in {'running','loading_teacher','loading_model','waiting_for_gpu','retrying_training'}:continue
        proc=Path('/proc',str(item.get('pid')),'cmdline')
        if proc.exists() and any(command in proc.read_bytes() for command in
            (b'scripts/data_worker.py',b'scripts/legacy_comparison_worker.py',b'scripts/agentic_rl_supervisor.py')):live[item['job']]=item
    return live

def queue_view(report=None):
    report=read_json(STATUS) if report is None else report
    active=external_jobs()
    for row in report.get('jobs',[]):
        if row['id'] in active and row.get('status')!='complete':
            worker=active[row['id']]
            state=worker['status']
            row.update(status=state,external_worker=worker,
                blocked_by=['gpu_reservation'] if state=='waiting_for_gpu' else [])
    report['external_workers']=active
    report['counts']={s:sum(j['status']==s for j in report.get('jobs',[])) for s in {j['status'] for j in report.get('jobs',[])}}
    return report

def gate_reason(name,job,cfg):
    if name=='dedicated_rl_supervisor':
        return None if os.environ.get('MEDAGENT_EARLY_RL_EXECUTOR')=='dual-v1' else 'requires_dedicated_agentic_rl_supervisor'
    if name=='dual_gpu_executor':
        return None if os.environ.get('MEDAGENT_EXECUTOR_VERSION')=='dual-v1' else 'requires_dual_gpu_executor'
    path=ROOT/cfg['gates'][name]['artifact']
    if not path.exists():return name+':missing_evidence'
    if name=='data_preflight':
        from .data_preflight import valid_proof
        return None if valid_proof() else name+':invalid_or_stale_evidence'
    proof=read_json(path)
    if name=='teacher_validation':
        if proof.get('split')!='validation' or not isinstance(proof.get('teacher_f1'),(int,float)) or not isinstance(proof.get('student_f1'),(int,float)) or proof['teacher_f1']<=proof['student_f1']:
            return name+':teacher_not_verified_better'
    if name=='frozen_experts':
        if not proof.get('validated') or not proof.get('specialist_checkpoint'):return name+':not_validated'
        for role in ('specialist','needs'):
            binding=proof.get('models',{}).get(role,{})
            if not all(binding.get(k) for k in ('endpoint','model','checkpoint','checkpoint_sha256')):return name+':missing_fixed_binding:'+role
        checkpoint=Path(proof['specialist_checkpoint'])
        checkpoint=checkpoint if checkpoint.is_absolute() else ROOT/checkpoint
        if not checkpoint.exists():return name+':checkpoint_missing'
    if name=='finalist_selection':
        if proof.get('split')!='validation' or job['stage'] not in proof.get('stages',[]):return name+':not_selected_on_validation'
    return None

def complete(job):
    try:return _completion_proof(job)
    except (OSError,ValueError,KeyError,TypeError):return False

def _completion_proof(job):
    base=ROOT/'runtime/small_model';data=base/'data'
    if job['kind']=='training':
        path=base/'checkpoints'/job['stage']/str(job['seed'])/'complete.json'
        return path.exists() and read_json(path).get('status')=='complete'
    if job['kind']=='generation':
        if job['phase']=='preflight':
            from .data_preflight import valid_proof
            return valid_proof()
        if job['phase']=='reference_review':
            from .reference_review import VERSION,reviewed_reference
            from .triage import RULE_VERSION
            proof=data/'reference_review'/'complete.json'
            if not proof.exists():return False
            report=read_json(proof)
            if report.get('status')!='complete' or report.get('version')!=VERSION or report.get('count')!=540:return False
            if report.get('triage_rule_version')!=RULE_VERSION:return False
            return report.get('source_manifest_sha256')==digest((data/'compact_manifest.json').read_bytes())
        if job['phase']=='cards':
            cfg=read_json(ROOT/'configs/small_training.json')['card_targets']
            n=sum(a.get('enabled',True) for a in read_json(ROOT/'configs/specialties.json')['agents'])
            return all(len(list((data/'cards'/split).glob('*.json')))==count for split,count in
                [('test',cfg['test']),('train',cfg['train_per_specialty']*n),('validation',cfg['validation_per_specialty']*n)])
        if not job.get('views'):return False
        targets=read_json(ROOT/'configs/small_training.json')['data_targets']
        for view in job['views']:
            if not all((data/view/(s+'.json')).exists() for s in ('train','validation')):return False
            p=data/view/'coverage.json'
            if not p.exists():return False
            coverage=read_json(p)
            if coverage.get('status')!='passed':return False
            if any(coverage['splits'][s]['sha256']!=digest((data/view/(s+'.json')).read_bytes()) for s in ('train','validation')):return False
            if view in targets:
                new_counts=[coverage['splits'][s].get('new_rows',sum(v.get('synthetic_rows',0) for v in coverage['splits'][s].get('specialties',{}).values())) for s in ('train','validation')]
                if sum(new_counts)!=targets[view]:return False
                if len(list((data/'generated'/view).glob('*.json')))!=targets[view]:return False
            if job['phase']=='ablation':
                p=data/view/'complete.json'
                if not p.exists() or read_json(p).get('status')!='complete':return False
                proof=read_json(p)
                if any(proof['matched_source_sha256'][s]!=digest((data/'specialist'/(s+'.json')).read_bytes()) for s in ('train','validation')):return False
                source_coverage=read_json(data/'specialist/coverage.json')
                for split in ('train','validation'):
                    source=source_coverage['splits'][split];target=coverage['splits'][split]
                    count=source.get('rows',source.get('count'))
                    if target.get('rows',target.get('count'))!=count or len(list((data/view/'rows'/split).glob('*.json')))!=count:return False
                    if view=='compressed_teacher_sft':
                        new_count=source.get('new_rows',sum(v.get('synthetic_rows',0) for v in source.get('specialties',{}).values()))
                        if proof.get('compression_audit',{}).get(split,{}).get('synthetic_rows_checked')!=new_count:return False
        return True
    return False

def snapshot(current=None,failures=None,waiting_for=None):
    cfg=read_json(QUEUE);done={j['id'] for j in cfg['jobs'] if complete(j)};rows=[]
    for job in cfg['jobs']:
        row={**job,'status':'queued','blocked_by':[]}
        if job['id'] in done:row['status']='complete'
        elif job['id'] in (current if isinstance(current,(list,set,tuple)) else [current]):row['status']='running'
        elif job['id'] in (failures or {}):row.update(status='failed',error=failures[job['id']])
        elif job.get('implementation_required'):row.update(status='needs_implementation',blocked_by=['training_view_and_entry_point'])
        else:
            missing=[d for d in job['dependencies'] if d not in done]
            missing += [r for gate in job.get('gates',[]) if (r:=gate_reason(gate,job,cfg))]
            row['blocked_by']=missing
            row['status']='waiting_dependency' if missing else 'ready'
        rows.append(row)
    report={'version':cfg['version'],'updated_at':time.time(),'allowed_physical_gpus':sorted(allowed_indices()),
        'waiting_for_supervisor':waiting_for,'jobs':rows,
        'counts':{s:sum(j['status']==s for j in rows) for s in {j['status'] for j in rows}}}
    report=queue_view(report);write_json(STATUS,report);return report
