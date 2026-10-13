import re
from .common import digest, norm, parse_json, read_json


def user_text(item):
    return '\n'.join(m['content'] for m in item['messages'] if m['role']=='user')


def case_body(text):
    if '<reference>' in text: text=text.split('<reference>',1)[1].split('</reference>',1)[0]
    text=re.sub(r'^分析此病例的牙隐裂分级[:：]\s*','',text.strip())
    return norm(text)


def mask_grade(text):
    return re.sub(r'((?:根管治疗)?难度(?:评估|分级)?\s*[:：为]?\s*)[ⅠⅡⅢⅣIV一二三四1-4]+\s*级',r'\1[已屏蔽旧分级]',text)


def prepare(task,cfg):
    from .rules import assess
    raw=read_json(cfg['test']); train=read_json(cfg['train'])
    train_hashes={digest(case_body(user_text(x))) for x in train}
    train_ids={str(x['id']) for x in train if 'id' in x}
    rows=[];seen=set(); issues=[];gold_conflicts=[]
    for i,item in enumerate(raw):
        text=user_text(item); h=digest(case_body(text)); rid=str(item.get('id',i))
        target=[m['content'] for m in item['messages'] if m['role']=='assistant'][-1]
        gold=parse_json(target)
        try:
            conflicts=[x for x in assess(task,gold)['issues'] if x['code']=='model_rule_disagreement']
            if conflicts:gold_conflicts.append({'id':rid,'conflicts':conflicts})
        except (ValueError,KeyError,TypeError) as exc:
            issues.append({'id':rid,'code':'invalid_gold_schema','detail':str(exc)})
        overlap=h in train_hashes
        duplicate=h in seen;seen.add(h)
        cleaned=mask_grade(text) if cfg.get('mask_existing_difficulty') else text
        rows.append({'id':rid,'text':cleaned,'gold':gold,'case_sha256':h,
            'train_text_overlap':overlap,'train_id_overlap':rid in train_ids,
            'duplicate_test':duplicate,'grade_masked':cleaned!=text})
    return rows,{'task':task,'test_samples':len(raw),'train_candidate_samples':len(train),
        'train_text_overlap':sum(x['train_text_overlap'] for x in rows),
        'train_id_overlap':sum(x['train_id_overlap'] for x in rows),
        'duplicates_in_test':sum(x['duplicate_test'] for x in rows),
        'masked_existing_grade':sum(x['grade_masked'] for x in rows),
        'gold_schema_issues':issues,'gold_rule_conflicts':gold_conflicts,
        'test_sha256':digest(open(cfg['test'],'rb').read()),
        'train_sha256':digest(open(cfg['train'],'rb').read()),
        'limitations':['Exact-text audit does not establish patient-level independence.','Training candidates may include validation rows; overlap treated conservatively.']}
