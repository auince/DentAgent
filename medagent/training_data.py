"""Traceable case groups and strict train/evaluation separation."""
import json
import re
from collections import defaultdict
from .common import digest,read_json
from .datasets import case_body,user_text,mask_grade

def shingles(text,n=5):
    text=re.sub(r'\s+','',text)
    return {text[i:i+n] for i in range(max(0,len(text)-n+1))}

class LeakageGuard:
    def __init__(self,heldout,threshold=.85):
        self.threshold=threshold;self.items=[];self.postings=defaultdict(set)
        self.exact=set();self.identities=set()
        for row in heldout:self.add(row)

    def add(self,row):
        text=case_body(row['text']);tokens=shingles(text);idx=len(self.items)
        self.exact.add(digest(text));self.items.append(tokens)
        for t in tokens:self.postings[t].add(idx)
        if row.get('source_identity'):self.identities.add(row['source_identity'])

    def overlaps(self,row):
        text=case_body(row['text'])
        if digest(text) in self.exact:return 'exact_text'
        if row.get('source_identity') and row['source_identity'] in self.identities:return 'source_identity'
        terms=shingles(text);counts=defaultdict(int)
        for t in terms:
            for idx in self.postings.get(t,()):counts[idx]+=1
        for idx,n in counts.items():
            union=len(terms)+len(self.items[idx])-n
            if union and n/union>=self.threshold:return 'near_duplicate'
        return None

def identity(row,task):
    # Namespace by source dataset; an unverified row number is not a patient ID.
    path=row.get('file_path') or row.get('filePath') or row.get('path')
    return task+':'+str(path) if path else None

def patient_excerpt(text):
    parts=[]
    # Conservative section extraction: never expose examination/diagnosis as a complaint.
    pattern=r'(?:主诉|现病史)\s*[:：]\s*(.*?)(?=(?:既往史|个人史|检查|查体|专科检查|辅助检查|诊断|治疗|处理|主诉|现病史)\s*[:：]|$)'
    for match in re.finditer(pattern,text,re.S):
        value=match.group(1).strip()
        if value:parts.append(value)
    return '\n'.join(parts)

def prepare_legacy(cfg,prepared_root):
    heldout=[]
    for task,spec in cfg['tasks'].items():
        for row in read_json(spec['test']):heldout.append({'text':user_text(row),'source_identity':identity(row,task)})
    guard=LeakageGuard(heldout)
    cards=[];rejected=[];seen={}
    for task,spec in cfg['tasks'].items():
        raw=read_json(spec['train'])
        by_hash={digest(case_body(user_text(r))):r for r in raw}
        for split in ['validation','train']:
            # Existing validation assignment takes precedence over any new derivations.
            for row in read_json(prepared_root/task/(split+'.json')):
                original=by_hash.get(row['case_sha256'])
                if original is None:raise ValueError('source_metadata_not_found')
                text=user_text(original);candidate={'text':text,'source_identity':identity(original,task)}
                reason=guard.overlaps(candidate)
                if reason:rejected.append({'id':row['id'],'task':task,'reason':reason});continue
                key=digest(case_body(text))
                if key in seen and seen[key]!=split:
                    rejected.append({'id':row['id'],'task':task,'reason':'cross_split_shared_case'});continue
                seen[key]=split
                cards.append({'id':task+':'+row['id'],'group_id':key,'split':split,'task':task,
                    'source':spec['train'],'source_sha256':digest(open(spec['train'],'rb').read()),
                    'source_identity':candidate['source_identity'],'source_flag_isReal':original.get('isReal'),
                    'origin':'legacy_source_unverified','label_version':'legacy-v1',
                    'text':mask_grade(text) if task=='root_canal' else text,
                    'patient_text':patient_excerpt(text),'clinical_record':text,
                    'gold':row['gold'],'knowledge_ids':[], 'validation':'legacy_rule_schema',
                    'prepared_row':row})
    # Enforce near-duplicate validation exclusion across BOTH task datasets.
    validation=LeakageGuard([{'text':c['text'],'source_identity':c['source_identity']} for c in cards if c['split']=='validation'])
    clean=[]
    for c in cards:
        reason=validation.overlaps(c) if c['split']=='train' else None
        if reason:rejected.append({'id':c['id'],'task':c['task'],'reason':'validation_'+reason})
        else:clean.append(c)
    return clean,{'heldout_cases':len(heldout),'excluded':rejected,'patient_identity_verified':False,
        'grouping':'source identity + exact normalized text + 5-gram Jaccard >= 0.85',
        'counts':{s:sum(c['split']==s for c in clean) for s in ['train','validation']}}

def assert_training_rows(rows,split='train'):
    if not rows:raise ValueError('empty_dataset')
    for r in rows:
        if r.get('split')!=split or not r.get('group_id') or not r.get('provenance'):raise ValueError('missing_split_group_or_provenance')
        if r.get('validation') not in {'source_checked','legacy_rule_schema','synthetic_reference'}:raise ValueError('unvalidated_row')

def assert_disjoint(train,validation):
    groups={r['group_id'] for r in train}&{r['group_id'] for r in validation}
    if groups:raise ValueError('train_validation_group_overlap')

def validate_augmentation(card,result):
    """Teacher returns verifiable source spans; unsupported assertions are rejected."""
    if not isinstance(result,dict) or not isinstance(result.get('complaint'),str) or not isinstance(result.get('facts'),list):raise ValueError('invalid_augmentation')
    allowed=card['patient_text'];complaint=result['complaint']
    if not allowed:raise ValueError('no_patient_observable_source')
    from .consult_contracts import validate_facts
    facts,issues=validate_facts({'facts':result['facts']},{'patient':{'text':complaint,'kind':'patient_report'}})
    if issues or not facts:raise ValueError('unanchored_augmented_facts')
    for f in facts:
        q=f.get('original_quote')
        if not isinstance(q,str) or not q or q not in allowed:raise ValueError('clinical_fact_leak_or_missing_original_span')
    if re.search(r'(?:根管弯曲|根尖透射|探诊深度|叩诊|电活力|X线|CT显示)',complaint) and not re.search(r'(?:报告|医生|检查)',complaint):raise ValueError('unobservable_patient_fact')
    return facts
