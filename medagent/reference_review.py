"""Versioned weak-label review independent of generated candidate trajectories."""
import copy
import json
import re
from .common import ROOT,read_json,write_json,digest,parse_json
from .consult_contracts import validate_facts,merge_facts
from .generation_schemas import needs_schema,object_schema,references,response_schema
from .routing import NEEDS_PROMPT,validate_needs,selected_tasks
from .triage import urgent_fact_ids,RULE_VERSION

VERSION='routing-source-review-v4-separated-observations'
DATA=ROOT/'runtime/small_model/data'

def source_digest(card):
    return digest(json.dumps(card,ensure_ascii=False,sort_keys=True))

def reviewed_reference(card):
    proof=read_json(DATA/'reference_review'/(card['id']+'.json'))
    if proof.get('version')!=VERSION or proof.get('source_sha256')!=source_digest(card):
        raise ValueError('reference_review_binding_mismatch')
    if proof.get('triage_rule_version')!=RULE_VERSION:
        raise ValueError('reference_review_triage_rule_mismatch')
    return {**copy.deepcopy(card),**proof['reference'],
        'reference_review':{'version':VERSION,'source_sha256':proof['source_sha256'],
            'path':str((DATA/'reference_review'/(card['id']+'.json')).relative_to(ROOT)),
            'validation':'teacher_source_consistency_not_clinical_validity'}}

def review_card(card,teacher):
    """Two independent questions: final routing versus necessary initial information."""
    from .generation_teacher import META
    facts=copy.deepcopy(card['initial_facts']);sources=copy.deepcopy(card['sources']);available={}
    initial_text='\n'.join(s['text'] for s in sources.values())
    for key,answer in card.get('patient_answers',{}).items():
        if not isinstance(answer,dict) or not answer.get('fact'):continue
        f=copy.deepcopy(answer['fact']);sid='review-answer-'+key;f['source_id']=sid
        src={sid:{'text':answer['text'],'kind':'patient_report'}}
        incoming,issues=validate_facts({'facts':[f]},src)
        if issues:raise ValueError('reference_answer_not_grounded:'+key)
        facts=merge_facts(facts,incoming);sources.update(src)
        if f['quote'] not in initial_text:available[key]={'text':answer['text'],'fact_ids':[x['id'] for x in incoming]}
    cache_path=DATA/'reference_review_intermediate'/(card['id']+'.json')
    cache=read_json(cache_path) if cache_path.exists() else {}
    if cache.get('version')!=VERSION or cache.get('source_sha256')!=source_digest(card):cache={'version':VERSION,'source_sha256':source_digest(card)}
    requests=cache.get('teacher_requests',{})
    if urgent_fact_ids(card['initial_facts']):
        needs={'candidates':[]};question_review={'fields':[]}
    else:
        if urgent_fact_ids(facts):needs={'candidates':[]}
        else:
            needs=cache.get('final_needs')
            if needs is None:
                raw=teacher(NEEDS_PROMPT+'''\n这是独立参考审计的最终路由步骤。facts是已收到全部实际可提供补充后的最终事实，所有事实均已可见，绝不能只看最初主诉或重新等待已经给出的回答。
只评估这些最终事实支持哪些专业咨询。自发痛/夜间痛支持牙髓咨询；具体牙号或片子缺失不阻止咨询。定点咬合痛支持疑似隐裂咨询；根管难度必须有当前临床治疗依据。分别引用当前facts中的阳性证据，保留真正尚未解决的缺口。这不是确诊。
看不到旧参考标签、候选轨迹或数据文件所属病种。''',{'facts':facts,'sources':sources},needs_schema(facts))
                needs=validate_needs(raw,facts);requests['final_route']=META.get()
                cache.update(final_needs=needs,teacher_requests=requests);write_json(cache_path,cache)
        question_review=cache.get('question_review')
        if question_review is None:
            if not available:question_review={'fields':[]}
            else:
                quote_variants=[]
                for sid,source in card['sources'].items():
                    text=source['text']
                    spans={text,*[s.strip() for s in re.split(r'(?<=[。！？；\n])',text) if s.strip()]}
                    spans.update(f['quote'] for f in card['initial_facts'] if f.get('source_id')==sid and f.get('quote') and f['quote'] in text)
                    spans={s for s in spans if s.strip()}
                    if spans:quote_variants.append(object_schema({'source_id':{'const':sid},'quote':{'type':'string','enum':sorted(spans)}}))
                if not quote_variants:raise ValueError('reference_missing_initial_sources')
                quote_schema={'anyOf':quote_variants}
                field_schema=object_schema({'field':{'type':'string','enum':list(available)},
                    'state':{'type':'string','enum':['already_known','needed','optional']},
                    'initial_quotes':{'type':'array','items':quote_schema,'maxItems':2},
                    'new_fact_ids':references([i for v in available.values() for i in v['fact_ids']],0),
                    'reason':{'type':'string','minLength':1,'maxLength':160}})
                schema=response_schema('reference_questions',object_schema({'fields':{'type':'array','items':field_schema,'minItems':len(available),'maxItems':len(available)}}))
                question_review=teacher('''独立审计初始咨询真正需要哪些追问。每个available_answers字段必须出现一次，判断already_known、needed或optional；最多3个needed。
逐字相同并不是判断已知的要求：初始文本与回答同义也属于already_known，例如“快俩月”和“两个月”、“昨天肿”和“昨天突然肿”。引用initial_sources中的真实连续原句，不能因为initial_facts漏了时间等信息就假装患者没说过。
needed仅用于初始确实缺失、回答实际增加重要信息且影响专业派发/紧迫分流/当前咨询的字段。可问、能回答与必须追问不同；已有症状、未知牙位、未拍片不自动要求重复追问。不将能提供的病史字典全部列为必问。相同信息的换一种说法属于already_known。
answer包含新但不影响当前咨询的细节时用optional；例如不改变处理方向的天数细化。needed要引用对应available_answers.fact_ids，说明新增信息及其必要性。already_known必须给出初始原句。只评审信息缺口，不重新判断最终病种，不读取旧标签或任何候选轨迹。''',
                    {'initial_sources':card['sources'],'initial_facts':card['initial_facts'],'available_answers':available,'final_consultation_tasks':sorted(selected_tasks(needs))},schema)
                requests['initial_information']=META.get()
            fields=question_review.get('fields')
            if not isinstance(fields,list) or {f.get('field') for f in fields}!=set(available) or len(fields)!=len(available):raise ValueError('reference_question_field_coverage')
            for item in fields:
                field=item['field'];state=item.get('state');quotes=item.get('initial_quotes',[]);ids=item.get('new_fact_ids',[])
                if state not in {'already_known','needed','optional'}:raise ValueError('reference_question_state')
                if any(q.get('source_id') not in card['sources'] or not q.get('quote') or q['quote'] not in card['sources'][q['source_id']]['text'] for q in quotes):raise ValueError('reference_question_unlocated_initial_quote')
                if not set(ids)<=set(available[field]['fact_ids']):raise ValueError('reference_question_wrong_field_evidence')
                if state=='already_known' and not quotes:raise ValueError('reference_already_known_without_source')
                if state=='needed' and not ids:raise ValueError('reference_needed_without_new_fact')
            if sum(f['state']=='needed' for f in fields)>3:raise ValueError('reference_question_budget')
            cache.update(question_review=question_review,teacher_requests=requests);write_json(cache_path,cache)
    questions=[f['field'] for f in question_review['fields'] if f['state']=='needed']
    return {'version':VERSION,'source_sha256':source_digest(card),'id':card['id'],'split':card['split'],'group_id':card['group_id'],
        'reference':{'required_tasks':sorted(selected_tasks(needs)),'useful_questions':questions},
        'review':{'validated_needs':needs,'question_review':question_review},
        'original_reference':{k:card.get(k) for k in ('required_tasks','useful_questions')},
        'teacher_requests':requests,'review_prompt_revision':'final-routing-initial-information-separated-v4',
        'triage_rule_version':RULE_VERSION,
        'clinical_validation':False,'candidate_trajectory_visible':False}

def build(teacher,workers):
    from .generation_jobs import run_rows
    cards=[read_json(p) for s in ('train','validation') for p in sorted((DATA/'cards'/s).glob('*.json'))]
    if len(cards)!=540:raise ValueError('reference_review_source_count')
    def produce(i):
        result=review_card(cards[i],teacher)
        write_json(DATA/'reference_review'/(cards[i]['id']+'.json'),result)
        write_json(DATA/'generated'/'reference_review'/f'{i:06}.json',result)
    run_rows('reference_review',len(cards),produce,DATA/'generated'/'reference_review',DATA/'generation_status.json',workers)
    results=[read_json(DATA/'reference_review'/(c['id']+'.json')) for c in cards]
    for c in cards:reviewed_reference(c)
    changed=[r['id'] for r in results if r['reference']!=r['original_reference']]
    # Previously accepted traces must satisfy the reviewed reference, not just old labels.
    from .training_rewards import trajectory_score,trajectory_completion
    from .consult_contracts import question_field_issue,finish_issue
    import time
    by_group={c['group_id']:c for c in cards};retained=quarantined=restored=0;best={}
    current=list((DATA/'generated'/'leader').glob('*.json'))
    archived=list((DATA/'audit_history').glob('**/rows/*.json'))+list((DATA/'audit_history').glob('**/quarantined_leader/*.json'))
    for path in archived+current:
        row=read_json(path)
        if not isinstance(row.get('id'),str) or not row['id'].startswith('leader') or 'episode' not in row:continue
        card=by_group.get(row.get('group_id'))
        if not card or row['episode'].get('initial_facts')!=card['initial_facts'] or row['episode'].get('sources')!=card['sources'] or row.get('provenance')!=card['provenance']:continue
        new=reviewed_reference(card)
        ref=row['episode']|{k:new[k] for k in ('required_tasks','useful_questions')}
        actions=[parse_json(turn['completion'][0]['content']) for turn in row['trajectory'][1:]]
        score=trajectory_score(actions,row['tool_observations'],ref)
        parameter_issues=[question_field_issue(a) for a in actions if question_field_issue(a)]
        parameter_issues += [finish_issue(view,view['limits']) for a,turn in zip(actions,row['trajectory'][1:])
            if a.get('action')=='finish' for view in [parse_json(turn['prompt'][-1]['content'])]
            if finish_issue(view,view['limits'])]
        urgent_invalid=bool(urgent_fact_ids(card['initial_facts'])) and (len(actions)!=1 or actions[0].get('action')!='refer')
        empty_expert=any(not result.get('findings') or any(not isinstance(f.get('interpretation'),str) or not f['interpretation'].strip() for f in result['findings'])
            for o in row['tool_observations'] if o.get('tool')=='expert' for task in o.get('successful_tasks',[])
            for result in [o.get('results',{}).get(task,{})])
        if score<.6 or parameter_issues or urgent_invalid or empty_expert or not trajectory_completion(actions,row['tool_observations'],ref)['passed']:continue
        if path.name not in best or score>=best[path.name][0]:
            row.update(episode=ref,trajectory_reward=score,reference_review=new['reference_review'])
            best[path.name]=(score,row,path)
    archive=DATA/'reference_review'/'quarantined_leader'/str(int(time.time()))
    for path in current:
        if path.name not in best:
            archive.mkdir(parents=True,exist_ok=True);path.replace(archive/path.name);quarantined+=1
    for name,(_,row,source) in best.items():
        target=DATA/'generated/leader'/name
        if target.exists():retained+=1
        else:restored+=1
        write_json(target,row)
    write_json(DATA/'reference_review'/'complete.json',{'status':'complete','version':VERSION,'count':len(results),
        'changed_count':len(changed),'changed_ids':changed,'retained_leader_rows':retained,
        'quarantined_leader_rows':quarantined,'restored_valid_leader_rows':restored,'clinical_validation':False,
        'test_references_modified':False,'review_prompt_revision':'final-routing-initial-information-separated-v4',
        'triage_rule_version':RULE_VERSION,
        'source_manifest_sha256':digest((DATA/'compact_manifest.json').read_bytes())})
