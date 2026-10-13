"""Select already generated cards without changing the complete test suite."""
import json
import shutil
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json,digest
from medagent.consult_contracts import REGISTRY
from medagent.compact_data import screening_issues,diverse_order,summarize


def main():
    data=ROOT/'runtime/small_model/data';manifest=data/'compact_manifest.json'
    if manifest.exists():raise ValueError('compact_selection_already_frozen')
    cfg=read_json(ROOT/'configs/small_training.json');selected={};report={};excluded=[]
    test={read_json(p)['group_id'] for p in (data/'cards/test').glob('*.json')}
    for split in ['train','validation']:
        selected[split]=[];report[split]={}
        candidates=[(p,read_json(p)) for p in sorted((data/'cards'/split).glob('*.json'))]
        for task in REGISTRY:
            pool=[]
            for p,c in candidates:
                if c['primary_specialty']!=task:continue
                issues=screening_issues(c)
                if c['group_id'] in test:issues.append('test_group_overlap')
                if issues:excluded.append({'split':split,'file':p.name,'issues':issues});continue
                pool.append(c)
            n=cfg['card_targets'][split+'_per_specialty']
            if len(pool)<n:raise ValueError(f'insufficient_screened_cards:{split}:{task}:{len(pool)}<{n}')
            chosen=diverse_order(pool)[:n];selected[split]+=chosen;report[split][task]=summarize(chosen)
    groups=[{c['group_id'] for c in selected[s]} for s in ['train','validation']]
    if groups[0]&groups[1]:raise ValueError('cross_split_group_overlap')
    archive=ROOT/'runtime/small_model/discarded'/('full_scale_pool_'+str(int(time.time())))
    archive.mkdir(parents=True)
    files={}
    # Planning and checks above finish before any source file moves.
    for split in ['train','validation']:
        source=data/'cards'/split;shutil.move(str(source),str(archive/split));source.mkdir()
        files[split]={}
        for card in selected[split]:
            name=card['id'].split('-')[-1]+'.json'
            original=archive/split/name
            if read_json(original)['group_id']!=card['group_id']:raise ValueError('source_identity_changed')
            shutil.copy2(original,source/name);files[split][name]=digest((source/name).read_bytes())
    write_json(manifest,{'version':'compact-diverse-v1','archive':str(archive),'counts':report,'excluded':excluded,
        'files':files,'test_cases_preserved':len(test),'selection':'greedy feature balance and lexical novelty',
        'clinical_validation':False,'screening':'source spans and conservative duration consistency only'})
    print(json.dumps({'report':report,'excluded':excluded,'archive':str(archive)},ensure_ascii=False))


if __name__=='__main__':main()
