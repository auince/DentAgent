"""Resumable local-teacher generation. Rejected rows never count toward targets."""
import argparse
import concurrent.futures
import base64
import json
import random
import os
import itertools
import sys
import time
from collections import deque
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json,parse_json,digest
from medagent.provider import LocalModelClient,ModelUnavailable
from medagent.consult_contracts import REGISTRY,FACT_PROMPT,LEADER_PROMPT,EXPERT_PROMPT,ANSWER_PROMPT,VISION_PROMPT,validate_facts,visible_messages
from medagent.training_data import validate_augmentation,LeakageGuard
from medagent.routing import NEEDS_PROMPT, needs_view, validate_needs
from medagent.datasets import user_text
from medagent.generation_contracts import card_format
from medagent.retrieval import KnowledgeIndex

OUT=ROOT/'runtime/small_model/data'
from medagent.generation_teacher import make_teacher,META,TeacherTransientError
teacher_client=make_teacher()
GENERATION_CONCURRENCY=teacher_client.cfg['concurrency']

class CardQualityReviewRequired(RuntimeError):
    pass

def teacher(prompt,payload,response_format=None):
    from medagent.generation_jobs import FEEDBACK
    error=FEEDBACK.get()
    if error and error.startswith('teacher_trajectory_did_not_satisfy_reference'):
        # Hidden reference labels belong in rejection logs, never model-visible feedback.
        error='轨迹质量未通过：请独立重查可见事实、信息缺口、工具结果和完成条件。'
    if error:prompt+='\n上次此训练样本未通过程序校验：'+error+'。请修复结构与证据引用，保持底稿事实不变，不补造检查。'
    schema_name=(response_format or {}).get('json_schema',{}).get('name')
    budget={'oral_needs':2048,'oral_expert':1536,'oral_action':256,'reference_review':1536,'reference_questions':2048,'oral_complaint_segments':2048,'oral_evidence_trace':2048,'oral_evidence_compression':1536}.get(schema_name)
    if budget and teacher_client.cfg['backend']=='local':
        return teacher_client.complete(prompt,payload,response_format,max_tokens=budget)
    return teacher_client.complete(prompt,payload,response_format)

def card_task(i,split):
    tasks=list(REGISTRY)
    if split=='test':
        category=next(k for end,k in [(600,'single'),(800,'multi'),(900,'insufficient'),(950,'urgent'),(1000,'followup')] if i<end)
    else:category=['single','single','single','multi','insufficient','followup','urgent'][i%7]
    selected=[tasks[i%len(tasks)]]
    if category=='multi':selected.append(tasks[(i+1)%len(tasks)])
    return category,selected

def create_cards():
    compact=OUT/'compact_manifest.json'
    if compact.exists():
        selection=read_json(compact)
        for split,files in selection['files'].items():
            for name,sha in files.items():
                if digest((OUT/'cards'/split/name).read_bytes())!=sha:raise ValueError('compact_card_source_changed')
        return
    index=KnowledgeIndex(ROOT/'knowledge')
    cfg=read_json(ROOT/'configs/small_training.json')
    curriculum=read_json(ROOT/'configs/specialty_curricula.json')['specialties']
    heldout=[]
    for spec in read_json(ROOT/'configs/experiments.json')['tasks'].values():
        heldout += [{'text':user_text(row)} for row in read_json(spec['test'])]
    development=OUT.parent/'development_scenarios/manifest.json'
    if development.exists():
        for fixture in read_json(development)['scenarios']:
            heldout.append({'text':'\n'.join(f['quote'] for f in fixture['reference']['facts'])})
    guard=LeakageGuard(heldout)
    counts={'test':cfg['card_targets']['test'],
        'train':cfg['card_targets']['train_per_specialty']*len(REGISTRY),
        'validation':cfg['card_targets']['validation_per_specialty']*len(REGISTRY)}
    def make(i,split,attempt):
        category,tasks=card_task(i,split);primary=tasks[0]
        chunks=[{k:v for k,v in c.items() if k!='tf'} for c in index.chunks if any(c['source'].startswith(t+'/') for t in tasks)]
        # Prefer substantive medical text over scraped menus and footer links.
        chunks.sort(key=lambda c:sum(c['text'].lower().count(w) for w in ['symptom','pain','swelling','bleeding','treatment','sensitive','difficulty','diagnos','因素','难度']),reverse=True)
        chosen=[]
        for task in tasks:
            choices=[c for c in chunks if c['source'].startswith(task+'/')][:8]
            if not choices:raise ValueError('no_reference_for:'+task)
            rng=random.Random(split+str(i));rng.shuffle(choices);chosen+=choices[:4]
        profile=curriculum[primary]['profiles'][(i//len(REGISTRY))%len(curriculum[primary]['profiles'])]
        record_visible=primary=='root_canal' or (i//len(REGISTRY))%5 in (1,2)
        request={'scenario_id':split+str(i),'category':category,'specialties':tasks,'references':chosen,
            'curriculum':curriculum[primary],'profile':profile,'record_available_initially':record_visible,
            'age_range':'adult','style':['简洁口语','含错字缩写','详细叙述','不确定部位'][(i//len(REGISTRY))%4],
            'scenario_variation':{'seed':split+str(i)+str(attempt),'age':18+(i*17)%65,'duration_days':1+(i*11)%90,
                'location':['左上后牙','右下后牙','前牙','说不清位置'][(i//len(REGISTRY))%4]}}
        previous=OUT/'teacher_candidates'/split/f'{i:05}-{attempt-1:02}.json'
        rejection=OUT/'rejections'/split/f'{i:05}-{attempt:02}.json'
        if attempt and attempt%3 and previous.exists() and rejection.exists():
            request['repair']={'previous_candidate':read_json(previous),'validation_error':read_json(rejection)['error'],
                'instruction':'只修复结构、引文和标签一致性错误。initial_facts的quote逐字复制自主诉或允许的病历；patient_answers.fact.quote必须逐字出现在该答案text中。未知答案用null，不补造缺失检查。'}
            available={c['id']:c['text'] for c in chosen}
            invalid=[]
            for evidence in request['repair']['previous_candidate'].get('knowledge_support',[]):
                quote=evidence.get('quote');source=available.get(evidence.get('id'),'')
                if not isinstance(quote,str) or not quote or quote not in source:
                    invalid.append({'id':evidence.get('id'),'rejected_quote':quote,'allowed_source_text':source})
            request['repair']['invalid_knowledge_quotes']=invalid
            request['repair']['knowledge_instruction']='知识quote必须从同一ID的allowed_source_text中逐字复制连续文字，保留空格、换行和标点，不能拼接、翻译、补字或跨片段。supports只解释所选文字支持的内容；缺少依据时重新构造有资料支持的病例，不能伪造引文。'
        result=teacher('''根据资料构造明确标记仿真的成人口腔咨询底稿。仅用资料支持的相关表现，不复制整段资料。
category: single单专业，multi多部位多专业，insufficient信息不足，urgent明确呼吸或吞咽困难需立即分流，followup需追问。初始主诉不得包含模拟患者未知的检查；检查单独放clinical_record。只有record_available_initially为真时才把检查因素放initial_facts。root_canal临床场景需明确当前治疗计划和检查，单纯牙痛不能作为根管难度依据。
返回JSON：{"patient_text":"患者原话","clinical_record":"单独的仿真临床记录，不能标为真实病人","initial_facts":[{"name":"因素","value":"原话信息","status":"positive|negative|unknown|conflict","location":null,"tooth":null,"quote":"来自patient_text或clinical_record连续引文","source_id":"patient或record"}],"required_tasks":["最终所需专业ID"],"useful_questions":["duration"],"patient_answers":{"duration":{"text":"底稿答案","fact":{"name":"持续时间","value":"三天","status":"positive","quote":"三天","source_id":"followup"}}},"knowledge_ids":["依据片段ID"],"knowledge_support":[{"id":"知识ID","quote":"该片段连续引文","supports":"仅一句中文转述"}]}。
useful_questions/答案键只能用duration/trigger/location/swelling/history/report。资料或底稿未知时答案为null。无需追问的字段也用null，不要给六个字段硬凑答案。未记录肿胀不能变成无肿胀，缺少报告不能变成阴性检查。quote必须是对应text的连续子串，不允许省略号拼接，不允许quote用null字符串；可以直接复制完整答案作为quote。urgent优先分流，required_tasks留空。不能假装真实病人或医生标注，不确诊。''',request,card_format(chosen,record_visible,category,tasks))
        if not isinstance(result,dict) or not isinstance(result.get('patient_text'),str):raise ValueError('invalid_card')
        write_json(OUT/'teacher_candidates'/split/f'{i:05}-{attempt:02}.json',result)
        if result.get('clinical_record') is None:result['clinical_record']=''
        if isinstance(result.get('clinical_record'),dict):
            result['clinical_record']=json.dumps(result['clinical_record'],ensure_ascii=False)
        if not isinstance(result.get('clinical_record',''),str):raise ValueError('invalid_record')
        sources={'patient':{'text':result['patient_text'],'kind':'patient_report'}}
        if record_visible:sources['record']={'text':result.get('clinical_record',''),'kind':'clinical_record'}
        repairs=[];candidate_facts=[];unknown_information=[]
        for fact in result.get('initial_facts',[]):
            source=sources.get(fact.get('source_id'),{}).get('text','')
            if fact.get('status')=='unknown' and (not fact.get('quote') or fact['quote'] not in source):
                # Absence of a record is a gap, not a fabricated quotation.
                unknown_information.append(fact.get('name'));repairs.append('unquoted_unknown_moved_to_gap');continue
            candidate_facts.append(fact)
        facts,issues=validate_facts({'facts':candidate_facts},sources)
        repairs+=['unsupported_tooth_reset_to_null']*issues.count('unsupported_tooth')
        hard=[issue for issue in issues if issue!='unsupported_tooth']
        if hard or not facts:raise ValueError('card_invalid_fact_spans:'+','.join(hard or ['no_facts']))
        required=result.get('required_tasks')
        if not isinstance(required,list) or any(t not in REGISTRY for t in required):raise ValueError('invalid_reference_route')
        if category=='single' and required!=[primary]:raise ValueError('single_specialty_curriculum_mismatch')
        if category=='multi' and not set(tasks)<=set(required):raise ValueError('multi_specialty_curriculum_mismatch')
        if category=='urgent' and required:raise ValueError('urgent_must_preempt_dispatch')
        known={c['id']:c for c in chosen}
        if not result.get('knowledge_ids') or not set(result['knowledge_ids'])<=known.keys():raise ValueError('invalid_knowledge_provenance')
        support=result.get('knowledge_support')
        if not isinstance(support,list) or not support:raise ValueError('missing_knowledge_support')
        for evidence in support:
            if evidence.get('id') not in known or not evidence.get('quote') or evidence['quote'] not in known[evidence['id']]['text']:
                raise ValueError('invalid_knowledge_support_span:'+str(evidence.get('id'))+':'+repr(evidence.get('quote'))[:160])
        answers=result.get('patient_answers',{})
        if not isinstance(answers,dict):raise ValueError('invalid_patient_answers')
        for key,answer in answers.items():
            if key not in {'duration','trigger','location','swelling','history','report'}:raise ValueError('invalid_answer_key')
            if answer is not None:
                if not isinstance(answer,dict) or not isinstance(answer.get('text'),str) or not isinstance(answer.get('fact'),dict):raise ValueError('invalid_answer_fact')
                fact={**answer['fact'],'source_id':'followup'}
                if fact.get('status')=='unknown':
                    answers[key]=None;repairs.append('unknown_patient_answer_preserved_as_unknown:'+key);continue
                if fact.get('quote') and fact['quote'] not in answer['text'] and fact['quote'] in result['patient_text']:
                    # Reuse an already supplied patient sentence verbatim, never invent an answer.
                    answer['text']=fact['quote'];repairs.append('followup_uses_verbatim_existing_report:'+key)
                normalized,bad=validate_facts({'facts':[fact]},{'followup':{'text':answer['text'],'kind':'patient_report'}})
                hard=[issue for issue in bad if issue!='unsupported_tooth']
                if hard or not normalized:raise ValueError('unlocated_followup_fact:'+key+':'+','.join(hard))
                if 'unsupported_tooth' in bad:repairs.append('followup_unsupported_tooth_reset_to_null:'+key)
                answer['fact']=normalized[0]
        useful=result.get('useful_questions',[])
        if not isinstance(useful,list) or any(not isinstance(k,str) for k in useful):raise ValueError('invalid_useful_questions')
        result['useful_questions']=[k for k in useful if answers.get(k) and answers[k]['text'] not in result['patient_text']]
        result.update(initial_facts=facts,sources=sources,split=split,category=category,id=split+f'-{i:05}',
            primary_specialty=primary,profile=profile,record_available_initially=record_visible,
            group_id=digest(split+'-oral-card-'+str(i)),origin='synthetic_reference',knowledge=chosen,
            provenance={'teacher':teacher_client.cfg['model'],'generator':teacher_client.cfg['campaign'],'api_request':META.get(),'references':[{'id':c['id'],'sha256':c['sha256']} for c in chosen]},
            validation='synthetic_reference',clinician_reviewed=False,normalization_repairs=repairs,unknown_information=unknown_information)
        return result
    # Keep requests in flight; individual quality failures go to the back of the
    # queue. They never count as accepted or stop unrelated candidates.
    blocked_splits={}
    # Load all already accepted splits before resuming an earlier split.
    # This keeps resumed training candidates isolated from saved validation cards.
    for path in (OUT/'cards').glob('*/*.json'):
        existing=read_json(path)
        if existing.get('provenance',{}).get('generator')!=teacher_client.cfg['campaign']:raise ValueError('discarded_teacher_data_in_active_dataset')
        guard.add({'text':existing['patient_text']})
    for split,count in counts.items():
        pending=deque(i for i in range(count) if not (OUT/'cards'/split/f'{i:05}.json').exists())
        attempts={};used={};blocked={}
        for i in pending:
            past=list((OUT/'rejections'/split).glob(f'{i:05}-*.json'))
            attempts[i]=max((int(p.stem.split('-')[-1]) for p in past),default=0)
            used[i]=0
        accepted=count-len(pending)
        workers=GENERATION_CONCURRENCY
        if workers<1:raise ValueError('invalid_generation_concurrency')
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            active={}
            while pending or active:
                while pending and len(active)<workers:
                    i=pending.popleft();active[pool.submit(make,i,split,attempts[i])]=i
                done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    i=active.pop(future)
                    try:
                        result=future.result()
                        if guard.overlaps({'text':result['patient_text']}):raise ValueError('generated_card_duplicate')
                        write_json(OUT/'cards'/split/f'{i:05}.json',result);guard.add({'text':result['patient_text']});accepted+=1
                    except (ValueError,KeyError,TypeError,ModelUnavailable) as exc:
                        if isinstance(exc,TeacherTransientError):
                            pending.append(i)
                            write_json(OUT/'provider_backoff.json',{'status':'retrying','error':str(exc),'time':time.time(),'concurrency':workers})
                            continue
                        attempts[i]+=1;used[i]+=1
                        write_json(OUT/'rejections'/split/f'{i:05}-{attempts[i]:02}.json',{'error':str(exc)[:300],'time':time.time()})
                        # Endpoint failures are infrastructural, not invalid medical cases.
                        if isinstance(exc,ModelUnavailable) and 'output_truncated' not in str(exc):raise
                        if used[i]<12:pending.append(i)
                        else:blocked[str(i)]=str(exc)[:300]
                write_json(OUT/'generation_status.json',{'phase':'cards','split':split,'completed':accepted,'target':count,
                    'concurrency':workers,'teacher':teacher_client.cfg['model'],'in_flight':len(active),'remaining_queued':len(pending),'blocked_count':len(blocked),'time':time.time()})
        write_json(OUT/'deferred'/f'{split}.json',{'split':split,'blocked':blocked,'time':time.time()})
        if blocked:blocked_splits[split]=blocked
        if split=='test' and not blocked:export_scenarios()
    if blocked_splits:
        raise CardQualityReviewRequired('card_quality_review_required:'+json.dumps({s:len(v) for s,v in blocked_splits.items()}))

def export_scenarios():
    from medagent.scenario_builder import build_scenarios
    return build_scenarios(OUT,chart_image)

def messages(system,text,completion):
    return {'prompt':[{'role':'system','content':system},{'role':'user','content':json.dumps(text,ensure_ascii=False)}],
        'completion':[{'role':'assistant','content':json.dumps(completion,ensure_ascii=False)}]}

def chart_image(text,target,seed):
    from PIL import Image,ImageDraw,ImageFont,ImageFilter
    fonts=[Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),Path('/System/Library/Fonts/PingFang.ttc')]
    fontpath=next((p for p in fonts if p.exists()),None)
    if fontpath is None:raise RuntimeError('CJK font required for readable medical chart simulation')
    font=ImageFont.truetype(str(fontpath),26);rng=random.Random(seed)
    lines=[]
    for line in ('仿真口腔病历（研究数据）\n'+text).splitlines():
        lines += [line[j:j+30] for j in range(0,len(line),30)] or ['']
    im=Image.new('RGB',(960,max(500,100+len(lines)*42)),(248,247,242));draw=ImageDraw.Draw(im)
    for n,line in enumerate(lines):draw.text((45,45+n*42),line,font=font,fill=(28,32,35))
    im=im.rotate(rng.uniform(-2,2),resample=Image.Resampling.BICUBIC,expand=True,fillcolor=(220,221,218))
    im=im.filter(ImageFilter.GaussianBlur(rng.uniform(0,.4)));target.parent.mkdir(parents=True,exist_ok=True);im.save(target,quality=85)

def generate_views(selected_views=None):
    from transformers import AutoTokenizer
    from medagent.rollout_env import ConsultationEnvironment
    tokenizer=AutoTokenizer.from_pretrained(OUT.parent/'models/Qwen3.5-4B')
    index=KnowledgeIndex(ROOT/'knowledge')
    cfg=read_json(ROOT/'configs/small_training.json')
    cards=[read_json(p) for s in ['train','validation'] for p in sorted((OUT/'cards'/s).glob('*.json'))]
    expected=(cfg['card_targets']['train_per_specialty']+cfg['card_targets']['validation_per_specialty'])*len(REGISTRY)
    if len(cards)!=expected:raise ValueError('synthetic_cards_incomplete')
    # External requests contain only newly generated synthetic cards and public knowledge.
    if any(c.get('provenance',{}).get('generator')!=teacher_client.cfg['campaign'] for c in cards):raise ValueError('mixed_teacher_campaign')
    from medagent.compact_data import diverse_order
    pools={(task,split):diverse_order([c for c in cards if c['split']==split and c.get('primary_specialty')==task]) for task in REGISTRY for split in ['train','validation']}
    for view,count in cfg['data_targets'].items():
        if selected_views and view not in selected_views:continue
        counters={};assignments=[]
        for i in range(count):
            task=list(REGISTRY)[i%len(REGISTRY)];ordinal=i//len(REGISTRY)
            divisor=9 if view=='specialist' else 10
            split='validation' if ordinal%divisor==0 else 'train'
            key=(task,split);offset=counters.get(key,0);counters[key]=offset+1
            pool=pools[(task,split)]
            if not pool:raise ValueError('missing_specialty_cards:'+str(key))
            assignments.append((task,split,pool[offset%len(pool)]))
        def generate_one(i):
            task,split,card=assignments[i]
            path=OUT/'generated'/view/f'{i:06}.json'
            if path.exists():return
            provenance=card.get('provenance',{'source':card.get('source'),'source_sha256':card.get('source_sha256')})
            row={'id':view+str(i),'split':split,'group_id':card['group_id'],'provenance':provenance,'validation':'source_checked','synthetic':True,'clinician_reviewed':False,'specialty':task}
            if view=='frontend':
                from medagent.generation_schemas import complaint_schema
                segmented=teacher('''把patient_text改写成自然口语片段，只改措辞，不增加症状、检查、牙位或否定。
返回JSON {"segments":[{"text":"口语片段","original_quote":"来源原句","name":"一个因素","value":"片段表达的信息","status":"positive|negative|unknown|conflict","location":null,"tooth":null}],"intent":"咨询"}。
每段只表达可由original_quote支持的一项事实；original_quote逐字复制来源可用引文。各text顺序拼起来应像完整患者口述，不写因素名清单。
程序会用整个text作为连续引文，不需要另写quote。否定、不确定、时间和左右部位与来源一致；缺失信息不凭空新建unknown事实。明确不存在和不知道不可互换。''',
                    {'patient_text':card['patient_text'],'patient_facts':[f for f in card['initial_facts'] if f.get('source_kind')=='patient_report'],
                        'variation':i,'style':['错字缩写','平实口语','保留原有否定','保留部位明确或不明确的程度'][i%4]},complaint_schema(card))
                segments=segmented.get('segments',[])
                if not isinstance(segments,list) or not segments or any(not isinstance(s.get('text'),str) or not s['text'].strip() for s in segments):raise ValueError('empty_complaint_segment')
                result={'complaint':'。'.join(s['text'] for s in segments),'intent':segmented.get('intent','咨询'),
                    'facts':[{k:v for k,v in s.items() if k!='text'}|{'quote':s['text'],'source_id':'patient'} for s in segments]}
                facts=validate_augmentation(card,result)
                target={'facts':[{k:v for k,v in f.items() if k in {'name','value','status','location','tooth','quote','source_id'}} for f in facts],'intent':result.get('intent','咨询')}
                row.update(messages(FACT_PROMPT,{'sources':{'patient':{'text':result['complaint'],'kind':'patient_report'}}},target),gold=target,text=result['complaint'])
                row['augmentation_method']='source_anchored_segments_v2'
            elif view=='vision':
                text=card['patient_text']+'\n检查记录：'+card.get('clinical_record','未提供')
                image=OUT/'images'/f'{i:06}.jpg';chart_image(text,image,i)
                row.update(messages(VISION_PROMPT,{'instruction':'转写仿真病历'}, {'text':'仿真口腔病历（研究数据）\n'+text,'uncertain_regions':[]}),images=[str(image.relative_to(ROOT))])
            elif view=='retrieval':
                positive=[k['text'] for k in card['knowledge'] if k['id'] in card['knowledge_ids']]
                other=next(c for c in cards if c['split']==split and c.get('knowledge')
                    and c['knowledge'][0]['id'] not in card['knowledge_ids']
                    and not set(c.get('required_tasks',[]))&set(card.get('required_tasks',[])))
                query=teacher('只将patient_text改写为一句检索问题，不补造症状或确诊，返回JSON {"query":"问题"}。',{'patient_text':card['patient_text'],'style':i%7,'variant':i})
                if not isinstance(query.get('query'),str) or not query['query']:raise ValueError('retrieval_query_invalid')
                row.update(query=query['query'],positives=positive,negatives=[other['knowledge'][0]['text']],positive_ids=card['knowledge_ids'],negative_source_ids=[other['knowledge'][0]['id']])
            elif view=='leader':
                from medagent.reference_review import reviewed_reference
                card=reviewed_reference(card)
                row['reference_review']=card['reference_review']
                episode={k:card[k] for k in ['initial_facts','required_tasks','useful_questions','patient_answers']}
                episode['sources']=card.get('sources',{'patient':{'text':card['patient_text'],'kind':'patient_report'}})
                def expert(task,state):
                    spec=REGISTRY[task]
                    from medagent.generation_schemas import expert_schema
                    context={k:state[k] for k in ('facts','sources','knowledge')}
                    result=teacher(EXPERT_PROMPT.format(specialty=spec['name'],scope=spec['scope'])+'\n仅输出findings、directions、missing_information，facts由引用ID回填原文对象。',context,expert_schema(context))
                    from medagent.evidence import validate_expert
                    result['reference_validation']=validate_expert(result,state)
                    used={fid for f in result['findings'] for fid in f['fact_ids']}
                    result['facts']=[f for f in state['facts'] if f['id'] in used]
                    return result
                def analyse(state):
                    from medagent.generation_schemas import needs_schema
                    return validate_needs(teacher(NEEDS_PROMPT,needs_view(state['facts'],recent_messages=visible_messages(state.get('messages',[]))),needs_schema(state['facts'])),state['facts'])
                env=ConsultationEnvironment(episode,expert,lambda q,t:index.search(q,t,3),read_json(ROOT/'configs/consultation.json'),analyse=analyse)
                needs_turn=messages(NEEDS_PROMPT,needs_view(env.state['facts'],recent_messages=visible_messages(env.state.get('messages',[]))),env.state['needs'])
                turns=[]
                while not env.done:
                    from medagent.generation_schemas import action_schema
                    observation=env.view();action=teacher(LEADER_PROMPT,observation,action_schema(env.state,env.config))
                    outcome=env.step(action)
                    if outcome.get('status')=='failed':raise ValueError('invalid_teacher_trajectory:'+json.dumps({'action':action,'outcome':outcome},ensure_ascii=False)[:800])
                    turns.append(messages(LEADER_PROMPT,observation,action))
                from medagent.training_rewards import trajectory_completion
                completion=trajectory_completion(env.actions,env.observations,episode)
                if env.reward()<.6 or not completion['passed']:raise ValueError('teacher_trajectory_did_not_satisfy_reference:'+json.dumps({'reward':env.reward(),'completion':completion,'actions':env.actions,'required':episode['required_tasks'],'useful_questions':episode['useful_questions']},ensure_ascii=False)[:800])
                row.update(turns[0],episode=episode,trajectory=[needs_turn]+turns,trajectory_reward=env.reward(),tool_observations=env.observations)
            else:
                spec=REGISTRY[task];context={'facts':card['initial_facts'],'sources':card.get('sources',{'patient':{'text':card['patient_text'],'kind':'patient_report'}}),'knowledge':card['knowledge']}
                prompt=EXPERT_PROMPT.format(specialty=spec['name'],scope=spec['scope']) if view=='specialist' else ANSWER_PROMPT
                if view=='specialist':
                    from medagent.generation_schemas import expert_schema
                    result=teacher(prompt+'\n只输出findings、directions、missing_information；facts由引用ID回填原文对象。',context,expert_schema(context))
                else:result=teacher(prompt,context)
                if not isinstance(result,dict):raise ValueError('invalid_generated_view')
                if view=='specialist':
                    if any(not isinstance(result.get(k),list) for k in ['findings','directions','missing_information']):raise ValueError('specialist_schema')
                    factids={f['id'] for f in card['initial_facts']};kids={k['id'] for k in card['knowledge']}
                    if any(not f.get('fact_ids') or not set(f['fact_ids'])<=factids for f in result['findings']):raise ValueError('specialist_invalid_fact')
                    if any(not d.get('evidence_ids') or not set(d['evidence_ids'])<=kids for d in result['directions']):raise ValueError('specialist_invalid_knowledge')
                    used={fid for finding in result['findings'] for fid in finding['fact_ids']}
                    result['facts']=[f for f in card['initial_facts'] if f['id'] in used]
                    from medagent.evidence import validate_expert
                    result['reference_validation']=validate_expert(result,context)
                    row.update(messages(prompt,context,result),gold=result,text='\n'.join(s['text'] for s in context['sources'].values()))
                else:
                    if not isinstance(result.get('answer'),str):raise ValueError('answer_schema')
                    if not isinstance(result.get('citations'),list) or not set(result['citations'])<={k['id'] for k in card['knowledge']}:raise ValueError('answer_unknown_citation')
                    rejected=teacher('只改写输入答案的一处患者事实使其与事实底稿不符，用于偏好训练的负例。其余保持一致，长度差小于10%。返回JSON {"answer":"修改后答案","citations":[],"changed_fact":"明确指出改了什么"}。',{'answer':result,'facts':card['initial_facts']})
                    if not rejected.get('changed_fact') or rejected.get('answer')==result['answer']:raise ValueError('preference_not_changed')
                    rejected.pop('changed_fact',None)
                    row.update(prompt=messages(prompt,context,result)['prompt'],chosen=[{'role':'assistant','content':json.dumps(result,ensure_ascii=False)}],rejected=[{'role':'assistant','content':json.dumps(rejected,ensure_ascii=False)}])
                    chosen_len=len(tokenizer.encode(row['chosen'][0]['content']));rejected_len=len(tokenizer.encode(row['rejected'][0]['content']))
                    if abs(chosen_len-rejected_len)/max(chosen_len,rejected_len,1)>.2:raise ValueError('preference_length_mismatch')
            if view!='vision':row['generation_request']=META.get()
            write_json(path,row)
        from medagent.generation_jobs import run_rows
        run_rows(view,count,generate_one,OUT/'generated'/view,OUT/'generation_status.json',GENERATION_CONCURRENCY)
        rows=[read_json(p) for p in sorted((OUT/'generated'/view).glob('*.json'))]
        if len(rows)!=count:raise ValueError('view_count_mismatch')
        if view=='specialist':
            for split in ['train','validation']:
                rows += [{**r,'legacy_task':r['task'],'text':r['prompt'][-1]['content']} for r in read_json(OUT/'legacy_specialist'/(split+'.json'))]
            for row in rows:
                if not isinstance(row.get('gold'),str):row['gold']=json.dumps(row['gold'],ensure_ascii=False)
        for split in ['train','validation']:write_json(OUT/view/(split+'.json'),[r for r in rows if r['split']==split])
        if view=='specialist':
            from medagent.data_coverage import audit_specialist_dataset
            audit_specialist_dataset(OUT/view)
        if view=='answer_dpo':
            for split in ['train','validation']:
                write_json(OUT/'answer'/(split+'.json'),[{**r,'completion':r['chosen']} for r in rows if r['split']==split])
        if view!='specialist':
            from medagent.view_audit import audit_view
            audit_view(OUT,view)
    export_scenarios()
    if all((OUT/name/'train.json').exists() and (OUT/name/'validation.json').exists() for name in cfg['data_targets']):
        from medagent.view_audit import audit_view
        audits={name:audit_view(OUT,name) for name in cfg['data_targets']}
        counts={name:{split:len(read_json(OUT/name/(split+'.json'))) for split in ['train','validation']} for name in cfg['data_targets']}
        write_json(OUT/'complete.json',{'status':'complete','counts':counts,'test_count':1000,'clinical_validation':False,'view_audits':audits})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['cards','views','all','scenarios','ablation','reference_review','preflight']);p.add_argument('--views',nargs='+');a=p.parse_args()
    if a.phase=='preflight':
        from medagent.data_preflight import build
        build();raise SystemExit(0)
    if a.phase=='reference_review':
        from medagent.reference_review import build
        build(teacher,GENERATION_CONCURRENCY);raise SystemExit(0)
    if a.phase=='ablation':
        from medagent.ablation_data import build
        for view in a.views or []:
            while True:
                try:
                    build(view,teacher);break
                except TeacherTransientError as exc:
                    write_json(OUT/'provider_backoff.json',{'status':'retrying','phase':'ablation','view':view,'error':str(exc),'time':time.time(),'concurrency':GENERATION_CONCURRENCY})
                    time.sleep(30)
        raise SystemExit(0)
    if a.phase=='scenarios':export_scenarios();raise SystemExit(0)
    for phase,fn in [('cards',create_cards),('views',generate_views)]:
        if a.phase not in {phase,'all'}:continue
        failures=0;previous=-1;quality_rounds=0
        while True:
            try:
                fn(a.views) if phase=='views' else fn();break
            except TeacherTransientError as exc:
                write_json(OUT/'provider_backoff.json',{'status':'retrying','phase':phase,'error':str(exc),'time':time.time(),'concurrency':GENERATION_CONCURRENCY})
                time.sleep(30)
                continue
            except CardQualityReviewRequired as exc:
                quality_rounds+=1
                write_json(OUT/'quality_gate.json',{'phase':phase,'status':'retrying_deferred_cards','round':quality_rounds,'error':str(exc),'time':time.time()})
                if quality_rounds>=3:raise
                # Every split was attempted; resume only missing cards with fresh feedback.
                continue
            except (ValueError,KeyError,TypeError) as exc:
                accepted=sum(1 for _ in (OUT/('cards' if phase=='cards' else 'generated')).rglob('*.json'))
                failures=failures+1 if accepted==previous else 1;previous=accepted
                write_json(OUT/'quality_gate.json',{'phase':phase,'accepted':accepted,'consecutive_failures_at_same_position':failures,'error':str(exc)[:200],'time':time.time()})
                if failures>=12:raise
