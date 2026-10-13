"""Deterministic, source-preserving diversity selection for the compact campaign."""
import re
from collections import Counter
from .common import digest


def features(card):
    facts=card.get('initial_facts',[])
    text=card['patient_text']
    result={('category',card['category']),('profile',str(card.get('profile'))),
        ('input','record' if card.get('record_available_initially') else 'oral'),
        ('length','short' if len(text)<40 else 'medium' if len(text)<100 else 'long'),
        ('route','multi' if len(card.get('required_tasks',[]))>1 else 'single_or_referral')}
    for f in facts:
        result.add(('state',f['status']))
        if f.get('tooth'):result.add(('tooth','explicit'))
        elif f.get('location'):result.add(('tooth','described'))
    for q in card.get('useful_questions',[]):result.add(('question',q))
    for word in ['凉','冷','热','咬','夜','肿','补','根管','不知道','没','不确定']:
        if word in text:result.add(('word',word))
    return result


def duration_ranges(text):
    # Conservative check: only compare a single stated duration in each message.
    values=[]
    numbers={'一':1,'两':2,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10}
    for m in re.finditer(r'(\d+|一|两|二|三|四|五|六|七|八|九|十)个?(多|余)?(天|周|星期|月|年)',text):
        n=int(m[1]) if m[1].isdigit() else numbers[m[1]]
        unit={'天':1,'周':7,'星期':7,'月':30,'年':365}[m[3]]
        values.append((max(0,n*unit-.25*unit),(n+(1 if m[2] else .25))*unit))
    return values


def screening_issues(card):
    issues=[]
    sources=card.get('sources',{})
    for f in card['initial_facts']:
        text=sources.get(f['source_id'],{}).get('text','')
        if not f.get('quote') or f['quote'] not in text:issues.append('fact_span_missing')
    known={k['id']:k['text'] for k in card['knowledge']}
    for e in card.get('knowledge_support',[]):
        if not e.get('quote') or e['quote'] not in known.get(e.get('id'),''):issues.append('knowledge_span_missing')
    answer=card.get('patient_answers',{}).get('duration')
    if isinstance(answer,dict):
        before=duration_ranges(card['patient_text']);after=duration_ranges(answer['text'])
        if len(before)==len(after)==1 and not any(w in answer['text'] for w in ['说错','更正','之前','这次','以前']):
            if max(before[0][0],after[0][0])>min(before[0][1],after[0][1]):issues.append('unmarked_duration_conflict')
    return sorted(set(issues))


def diverse_order(cards):
    """Balance feature coverage, then avoid repeated complaint vocabulary."""
    remaining=sorted(cards,key=lambda c:digest(c['group_id']))
    counts=Counter();seen=set();ordered=[]
    while remaining:
        def score(c):
            f=features(c);text=c['patient_text']
            grams={text[i:i+4] for i in range(max(1,len(text)-3))}
            novelty=len(grams-seen)/max(1,len(grams))
            return sum((3 if k in {'category','profile','input'} else 1)/(1+counts[(k,v)]) for k,v in f)+novelty
        best=max(remaining,key=score);remaining.remove(best);ordered.append(best)
        counts.update(features(best));text=best['patient_text'];seen.update(text[i:i+4] for i in range(max(1,len(text)-3)))
    return ordered


def summarize(cards):
    return {'count':len(cards),'independent_groups':len({c['group_id'] for c in cards}),
        'categories':dict(Counter(c['category'] for c in cards)),
        'profiles':dict(Counter(str(c.get('profile')) for c in cards)),
        'inputs':dict(Counter('record' if c.get('record_available_initially') else 'oral' for c in cards)),
        'states':dict(Counter(f['status'] for c in cards for f in c['initial_facts'])),
        'complaint_lengths':{'min':min(len(c['patient_text']) for c in cards),'max':max(len(c['patient_text']) for c in cards)}}
