"""Full data preparation proof; excludes model performance and clinical validity."""
from pathlib import Path
from .common import ROOT,read_json,write_json,digest,parse_json

DATA=ROOT/'runtime/small_model/data'
_HASH_CACHE={}

def file_hash(path):
    path=Path(path);s=path.stat();signature=(s.st_ino,s.st_size,s.st_mtime_ns)
    old=_HASH_CACHE.get(str(path))
    if old and old[0]==signature:return old[1]
    value=digest(path.read_bytes());_HASH_CACHE[str(path)]=(signature,value)
    return value

def valid_proof():
    try:
        proof=read_json(DATA/'preflight_complete.json')
        if proof.get('status')!='passed' or proof.get('clinical_validation') is not False:return False
        if not proof.get('bindings') or proof.get('research_test_count')!=1000:return False
        if proof.get('test_counts')!={'root_canal':101,'crack':20}:return False
        if proof['data_targets']!=read_json(ROOT/'configs/small_training.json')['data_targets']:return False
        from .reference_review import VERSION
        if proof['reference_version']!=VERSION:return False
        return all(file_hash(ROOT/p)==sha for p,sha in proof['bindings'].items())
    except (OSError,ValueError,KeyError,TypeError):return False

def build():
    from .view_audit import audit_view
    from .reference_review import VERSION,reviewed_reference
    from .ablation_data import audit_compression
    from transformers import AutoTokenizer
    cfg=read_json(ROOT/'configs/small_training.json');bindings={};reports={}
    def bind(path):bindings[str(path.relative_to(ROOT))]=file_hash(path)
    bind(ROOT/'configs/small_training.json')
    bind(ROOT/'medagent/triage.py')
    bind(ROOT/'medagent/reference_review.py')
    for name in ('consult_contracts','generation_schemas','view_audit'):
        bind(ROOT/'medagent'/(name+'.py'))
    bind(DATA/'compact_manifest.json')
    # Check every frozen source; these files were fingerprinted before label repair.
    histories=sorted((DATA/'audit_history').glob('routing-v2-*'))
    if not histories:raise ValueError('missing_frozen_source_binding_audit')
    original=read_json(histories[-1]/'frozen_source_bindings.json')
    for name,sha in original.items():
        if file_hash(ROOT/name)!=sha:raise ValueError('frozen_source_changed:'+name)
        bindings[name]=sha
    cards=[read_json(p) for s in ('train','validation') for p in (DATA/'cards'/s).glob('*.json')]
    if len(cards)!=540:raise ValueError('frozen_training_card_count')
    for card in cards:
        reviewed_reference(card)
        bind(DATA/'reference_review'/(card['id']+'.json'))
    bind(DATA/'reference_review/complete.json')
    for name in cfg['data_targets']:
        if len(list((DATA/'generated'/name).glob('*.json')))!=cfg['data_targets'][name]:raise ValueError('generated_row_count:'+name)
        reports[name]=audit_view(DATA,name)
        for split in ('train','validation'):bind(DATA/name/(split+'.json'))
    for view in ('direct_sft','short_evidence_sft','compressed_teacher_sft'):
        proof=read_json(DATA/view/'complete.json')
        for split in ('train','validation'):
            original_rows=read_json(DATA/'specialist'/(split+'.json'))
            rows=read_json(DATA/view/(split+'.json'))
            if len(rows)!=len(original_rows) or [r['group_id'] for r in rows]!=[r['group_id'] for r in original_rows]:raise ValueError('unmatched_ablation:'+view)
            if len(list((DATA/view/'rows'/split).glob('*.json')))!=len(rows):raise ValueError('missing_ablation_rows:'+view)
            if proof['matched_source_sha256'][split]!=file_hash(DATA/'specialist'/(split+'.json')):raise ValueError('ablation_source_changed:'+view)
            for row,source in zip(rows,original_rows):
                if source.get('legacy_task'):
                    if row['gold']!=source['gold']:raise ValueError('ablation_changed_legacy_label:'+view)
                    continue
                target=parse_json(row['gold']);gold=parse_json(source['gold'])
                if {k:v for k,v in target.items() if k!='findings'}!={k:v for k,v in gold.items() if k!='findings'}:raise ValueError('ablation_changed_fixed_output:'+view)
                if [f['fact_ids'] for f in target['findings']]!=[f['fact_ids'] for f in gold['findings']]:raise ValueError('ablation_changed_fact_ids:'+view)
                if view=='direct_sft' and any(f.get('interpretation')!='' for f in target['findings']):raise ValueError('direct_ablation_contains_reasoning')
                if view=='short_evidence_sft' and target['findings']!=gold['findings']:raise ValueError('short_ablation_changed_evidence')
            if view=='compressed_teacher_sft':audit_compression(DATA/view,split,rows,original_rows)
            bind(DATA/view/(split+'.json'))
        bind(DATA/view/'complete.json')
    # Use each role's actual tokenizer and every train/validation row, not a subset.
    tokenizers={};lengths={}
    stages=[s for s in cfg['stages'] if s['method']=='sft' and s['id']!='legacy_only_sft']
    for stage in stages:
        role=stage['role'];base=ROOT/cfg['models'][role]
        if role not in tokenizers:tokenizers[role]=AutoTokenizer.from_pretrained(base)
        tokenizer=tokenizers[role];maximum=0;count=0
        for split in ('train','validation'):
            rows=read_json(DATA/stage['data']/(split+'.json'));bind(DATA/stage['data']/(split+'.json'))
            for row in rows:
                for turn in row.get('trajectory',[row]):
                    ids=tokenizer.apply_chat_template(turn['prompt']+turn['completion'],tokenize=True,return_dict=False,enable_thinking=False)
                    if not isinstance(ids,list) or not ids or not isinstance(ids[0],int):raise ValueError('unexpected_tokenizer_encoding')
                    count+=1;maximum=max(maximum,len(ids))
                    if len(ids)>stage.get('max_length',cfg['defaults']['max_length']):raise ValueError('overlength_training_row:'+stage['id']+':'+row['id'])
        lengths[stage['id']]={'sequences':count,'max_tokens':maximum,'limit':stage.get('max_length',cfg['defaults']['max_length'])}
        print(stage['id'],lengths[stage['id']],flush=True)
    for split in ('train','validation'):
        for row in read_json(DATA/'vision'/(split+'.json')):
            for image in row['images']:
                path=ROOT/image
                if not path.exists() or not path.stat().st_size:raise ValueError('missing_vision_image')
                bind(path)
    test_counts={}
    for name,spec in read_json(ROOT/'configs/experiments.json')['tasks'].items():
        path=ROOT/spec['test'];test_counts[name]=len(read_json(path));bind(path)
    if test_counts!={'root_canal':101,'crack':20}:raise ValueError('original_full_test_counts_changed')
    if len(list((DATA/'cards/test').glob('*.json')))!=1000:raise ValueError('full_research_test_count')
    result={'status':'passed','clinical_validation':False,'scope':'full_counts_sources_group_isolation_contracts_and_sequence_lengths',
        'model_performance_evaluated':False,'data_targets':cfg['data_targets'],'reference_version':VERSION,
        'frozen_files_checked':len(original),'test_counts':test_counts,'research_test_count':1000,
        'view_reports':reports,'sequence_lengths':lengths,'bindings':bindings}
    write_json(DATA/'preflight_complete.json',result)
    return result
