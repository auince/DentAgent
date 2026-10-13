"""Case-text deduplication + stable splits; test cases excluded before new training."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json,digest,parse_json
from medagent.datasets import user_text,case_body,mask_grade
from medagent.rules import assess


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['root_canal','crack'],required=True);a=p.parse_args()
    cfg=read_json(ROOT/'configs/experiments.json')['tasks'][a.task]
    prompt=(ROOT/cfg['prompt']).read_text()
    source=read_json(cfg['train']);test=read_json(cfg['test'])
    excluded={digest(case_body(user_text(x))) for x in test};seen=set();splits={'train':[],'validation':[]};rejected=[]
    for i,row in enumerate(source):
        text=user_text(row);h=digest(case_body(text))
        if h in excluded or h in seen:continue
        seen.add(h)
        try:
            target=[m['content'] for m in row['messages'] if m['role']=='assistant'][-1]
            gold=parse_json(target);assess(a.task,gold)
        except (ValueError,KeyError,TypeError) as exc:
            rejected.append({'id':str(row.get('id',i)),'error':str(exc)});continue
        split='validation' if int(h[:8],16)%10==0 else 'train'
        if a.task=='root_canal':text=mask_grade(text)
        splits[split].append({'id':str(row.get('id',i)),'split':split,'task':a.task,'case_sha256':h,
            'prompt':[{'role':'system','content':prompt},{'role':'user','content':text}],
            'completion':[{'role':'assistant','content':json.dumps(gold,ensure_ascii=False)}],
            'gold':json.dumps(gold,ensure_ascii=False)})
    folder=ROOT/'runtime/training_data'/a.task
    for name,rows in splits.items():write_json(folder/(name+'.json'),rows)
    write_json(folder/'manifest.json',{'source':cfg['train'],'source_sha256':digest(Path(cfg['train']).read_bytes()),
        'test_excluded_before_split':True,'split_policy':'case-text SHA256 modulo 10; no verified patient identity',
        'counts':{k:len(v) for k,v in splits.items()},'rejected':rejected})
    print(json.dumps({k:len(v) for k,v in splits.items()}))


if __name__=='__main__': main()
