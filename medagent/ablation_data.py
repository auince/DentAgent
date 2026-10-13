"""Matched-source target variants; no new split and no clinical-label rewriting."""
import copy
import concurrent.futures
import json
from .common import ROOT,read_json,write_json,parse_json,digest
from .generation_schemas import object_schema,references,response_schema

VIEWS={'direct_sft':'直接输出事实和结论；findings保留fact_ids，interpretation用空字符串，不展开证据说明。',
    'short_evidence_sft':'每条findings.interpretation只给简短、可检查的证据关系。',
    'compressed_teacher_sft':'每条findings.interpretation给压缩后的短证据关系，不补造事实。'}

def audit_compression(out,split,rows,source_rows):
    """Audit resumed rows too; saved files alone do not prove a teacher trace."""
    checked=0
    for i,(row,source) in enumerate(zip(rows,source_rows)):
        if row.get('legacy_task'):continue
        gold=parse_json(source['gold']);target=parse_json(row['gold'])
        if {k:v for k,v in gold.items() if k!='findings'}!={k:v for k,v in target.items() if k!='findings'}:
            raise ValueError('compression_changed_fixed_output:'+str(i))
        if [f['fact_ids'] for f in gold['findings']]!=[f['fact_ids'] for f in target['findings']]:
            raise ValueError('compression_changed_fact_ids:'+str(i))
        if any(not isinstance(f.get('interpretation'),str) or not f['interpretation'].strip() or len(f['interpretation'])>80 for f in target['findings']):
            raise ValueError('invalid_compression_explanation:'+str(i))
        context=parse_json(source['prompt'][-1]['content'])
        known={f['id'] for f in context['facts']};kids={k['id'] for k in context['knowledge']}
        trace=read_json(out/'teacher_traces'/split/f'{i:06}.json')
        if not isinstance(trace.get('steps'),list) or not trace['steps']:
            raise ValueError('empty_or_invalid_teacher_trace:'+str(i))
        for step in trace['steps']:
            if not isinstance(step,dict) or not step.get('fact_ids') or not set(step['fact_ids'])<=known or not set(step.get('evidence_ids',[]))<=kids or not isinstance(step.get('explanation'),str) or not step['explanation'].strip():
                raise ValueError('invalid_cached_teacher_step:'+str(i))
        if row['compression_provenance']['teacher_trace_sha256']!=digest(json.dumps(trace,sort_keys=True,ensure_ascii=False)):
            raise ValueError('teacher_trace_binding_mismatch:'+str(i))
        checked+=1
    return {'synthetic_rows_checked':checked,'scope':'fixed_output_reference_ids_nonempty_trace_and_explanation','clinical_semantic_review':False}

def build(view,teacher):
    if view not in VIEWS:raise ValueError('unknown_ablation_view')
    data=ROOT/'runtime/small_model/data';source=data/'specialist';out=data/view
    manifest={'view':view,'contract':'unchanged facts, final directions, missing information and reference IDs',
        'source_sha256':{s:digest((source/(s+'.json')).read_bytes()) for s in ('train','validation')},
        'clinical_validation':False}
    if (out/'manifest.json').exists() and read_json(out/'manifest.json')!=manifest:raise ValueError('ablation_source_changed')
    write_json(out/'manifest.json',manifest)
    audits={}
    for split in ('train','validation'):
        source_rows=read_json(source/(split+'.json'));rows=[]
        def produce(pair):
            i,source_row=pair
            path=out/'rows'/split/f'{i:06}.json'
            if path.exists():return read_json(path)
            row=copy.deepcopy(source_row)
            if not row.get('legacy_task'):
                gold=parse_json(row['gold']);target=copy.deepcopy(gold)
                if view=='direct_sft':
                    for f in target['findings']:f['interpretation']=''
                elif view=='compressed_teacher_sft':
                    context=parse_json(row['prompt'][-1]['content'])
                    trace_path=out/'teacher_traces'/split/f'{i:06}.json'
                    known={f['id'] for f in context['facts']};kids={k['id'] for k in context['knowledge']}
                    step_schema=object_schema({'fact_ids':references(known),'evidence_ids':references(kids,0),'explanation':{'type':'string'}})
                    trace_schema=response_schema('oral_evidence_trace',object_schema({'steps':{'type':'array','items':step_schema,'minItems':1,'maxItems':20}}))
                    trace=read_json(trace_path) if trace_path.exists() else teacher(
                        '根据给定事实、资料和已固定的输出，写可核查的证据分析。只返回JSON {"steps":[{"fact_ids":["原有事实ID"],"evidence_ids":["原有知识ID"],"explanation":"说明"}]}。不得增改患者事实、诊断方向或缺失信息。',
                        {'context':context,'fixed_output':gold},trace_schema)
                    if not isinstance(trace,dict) or not isinstance(trace.get('steps'),list) or not trace['steps']:raise ValueError('invalid_teacher_trace')
                    for step in trace['steps']:
                        if not isinstance(step,dict) or not step.get('fact_ids') or not set(step['fact_ids'])<=known or not set(step.get('evidence_ids',[]))<=kids or not isinstance(step.get('explanation'),str) or not step['explanation'].strip():raise ValueError('unreferenced_teacher_trace')
                    write_json(trace_path,trace)
                    finding_schema=object_schema({'fact_ids':references(known),'interpretation':{'type':'string','maxLength':80}})
                    n=len(gold['findings'])
                    compression_schema=response_schema('oral_evidence_compression',object_schema({'findings':{'type':'array','items':finding_schema,'minItems':n,'maxItems':n}}))
                    result=teacher('把证据分析压缩成原findings的短说明。只返回JSON {"findings":[{"fact_ids":["保持原ID及顺序"],"interpretation":"不超过80个字符的一句证据说明"}]}。保持原条目数与事实ID，不能新加检查或疾病结论。',
                        {'context':context,'trace':trace,'original_findings':gold['findings']},compression_schema)
                    findings=result.get('findings') if isinstance(result,dict) else None
                    if not isinstance(findings,list) or len(findings)!=len(gold['findings']):raise ValueError('compression_changed_findings')
                    for old,new in zip(gold['findings'],findings):
                        if new.get('fact_ids')!=old['fact_ids'] or not isinstance(new.get('interpretation'),str) or not new['interpretation'].strip() or len(new['interpretation'])>80:raise ValueError('compression_changed_references_or_budget')
                    target['findings']=findings
                    row['compression_provenance']={'teacher_trace_sha256':digest(json.dumps(trace,sort_keys=True,ensure_ascii=False)),
                        'validation':'reference_ids_and_length_only','semantic_reviewed':False}
                row['prompt'][0]['content']+='\n'+VIEWS[view]
                row['completion']=[{'role':'assistant','content':json.dumps(target,ensure_ascii=False)}]
                row['gold']=json.dumps(target,ensure_ascii=False)
            row['ablation_view']=view
            write_json(path,row);return row
        concurrency=read_json(ROOT/'configs/generation_teacher.json')['concurrency']
        from .generation_jobs import run_rows
        run_rows(view+'_'+split,len(source_rows),lambda i:produce((i,source_rows[i])),out/'rows'/split,
            data/'generation_status.json',concurrency)
        rows=[read_json(out/'rows'/split/f'{i:06}.json') for i in range(len(source_rows))]
        if [r['group_id'] for r in rows]!=[r['group_id'] for r in source_rows]:raise ValueError('ablation_group_order_changed')
        if view=='compressed_teacher_sft':audits[split]=audit_compression(out,split,rows,source_rows)
        write_json(out/(split+'.json'),rows)
    from .data_coverage import audit_specialist_dataset
    audit_specialist_dataset(out)
    write_json(out/'complete.json',{'status':'complete','view':view,'matched_source_sha256':manifest['source_sha256'],'compression_audit':audits})
