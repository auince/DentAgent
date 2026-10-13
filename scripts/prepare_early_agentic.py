"""Freeze all currently valid Leader rows for user-authorized early RL."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json,digest
from medagent.view_audit import audit_view

def prepare():
    from transformers import AutoTokenizer
    data=ROOT/'runtime/small_model/data'
    target=data/'leader_agentic_early'
    if (target/'manifest.json').exists():
        raise ValueError('early_rl_snapshot_already_frozen')
    rows=[read_json(p) for p in sorted((data/'generated/leader').glob('*.json'))]
    if not rows:raise ValueError('no_valid_leader_rows')
    for split in ('train','validation'):
        write_json(data/'leader'/(split+'.json'),[r for r in rows if r['split']==split])
    audit=audit_view(data,'leader',expected_new_rows=len(rows))
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'runtime/small_model/models/Qwen3.5-4B')
    maximum=0;sequences=0
    for row in rows:
        for turn in row['trajectory']:
            ids=tokenizer.apply_chat_template(turn['prompt']+turn['completion'],tokenize=True,return_dict=False,enable_thinking=False)
            if len(ids)>16384:raise ValueError('early_rl_sequence_overlength:'+row['id'])
            maximum=max(maximum,len(ids));sequences+=1
    history=sorted((data/'audit_history').glob('routing-v2-*/frozen_source_bindings.json'))[-1]
    originals=read_json(history)
    for p,sha in originals.items():
        if digest((ROOT/p).read_bytes())!=sha:raise ValueError('frozen_source_changed:'+p)
    bindings={}
    for split in ('train','validation'):
        path=target/(split+'.json')
        write_json(path,[r for r in rows if r['split']==split])
        bindings[str(path.relative_to(ROOT))]=digest(path.read_bytes())
    complete_ids={r['id'] for r in rows}
    manifest={'status':'ready_for_early_rl','planned_rows':400,'actual_rows':len(rows),
        'split_counts':{s:sum(r['split']==s for r in rows) for s in ('train','validation')},
        'missing_ids':['leader'+str(i) for i in range(400) if 'leader'+str(i) not in complete_ids],
        'bindings':bindings,'audit':audit,'all_sequences_checked':sequences,'max_tokens':maximum,
        'frozen_original_files_checked':len(originals),'count_relaxation_authorized_by_user':True,
        'full_research_test_count':1000,'original_test_counts':{'root_canal':101,'crack':20},
        'clinical_validation':False,'formal_model_evaluation':False,'initialization':'base_model_no_sft'}
    write_json(target/'manifest.json',manifest)
    print({k:manifest[k] for k in ('status','actual_rows','split_counts','all_sequences_checked','max_tokens')})

if __name__=='__main__':prepare()
