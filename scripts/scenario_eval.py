"""All 60 authored cases through real HTTP -> Leader -> LangGraph -> model tools.
No oracle routing, replay predictions, supplied image transcription, or task tags.
"""
import argparse,base64,collections,fcntl,json,os,re,sys,time,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json,digest
from medagent.evaluation import evaluate
from medagent.rules import assess

CASES=ROOT/'scenarios/implicit_v1/cases.json'
OUT=ROOT/'runtime/scenario_eval_v1'
API='http://deployment.invalid'


def request_for(case):
    payload={'instruction':case['instruction']}
    if case['modality']=='image_only':
        raw=(ROOT/case['image']).read_bytes()
        payload.update(text='',images=['data:image/png;base64,'+base64.b64encode(raw).decode()])
    else:payload.update(text=case['text'],images=[])
    assert set(payload)=={'instruction','text','images'} and 'tasks' not in payload
    return payload


def call(path,payload=None):
    headers={'Content-Type':'application/json'}
    if payload is not None:headers['Idempotency-Key']=digest(json.dumps(payload,ensure_ascii=False,sort_keys=True))
    if os.environ.get('MEDAGENT_TOKEN'):headers['Authorization']='Bearer '+os.environ['MEDAGENT_TOKEN']
    req=urllib.request.Request(API+path,None if payload is None else json.dumps(payload,ensure_ascii=False).encode(),headers)
    with urllib.request.urlopen(req,timeout=60) as r:return json.load(r)


def freeze(cases):
    assert len(cases)==60 and len({c['id'] for c in cases})==60
    assert sum(c['modality']=='image_only' for c in cases)==30
    assert collections.Counter(tuple(c['expected_tasks']) for c in cases)=={('root_canal',):20,('crack',):20,('crack','root_canal'):20}
    for c in cases:
        assert not any(t in c['instruction'] for t in ('根管','隐裂','root_canal','crack'))
        for task,gold in c['gold'].items():
            result=assess(task,gold)
            assert result['assessments']==c['expected_assessments'][task]
        if c['modality']=='image_only':
            assert (ROOT/c['image']).is_file()
            assert (ROOT/c['image']).with_suffix('.ocr.txt').is_file()
    files=[CASES,ROOT/'configs/agent.json',ROOT/'configs/models.json',ROOT/'configs/runtime.json',ROOT/'scripts/model_gateway.py',Path(__file__),
           *sorted((ROOT/'medagent').glob('*.py'))]
    files += [ROOT/c['image'] for c in cases if c['image']]
    manifest={'suite':'implicit-v1','count':60,'api':API,'synthetic':True,
              'runtime':read_json(ROOT/'configs/runtime.json'),
              'files':{str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in files},
              'clinical_validation':'Project-rule author fixtures, not independent clinical adjudication',
              'request_policy':'No tasks key; image-only requests have empty text and only embedded image bytes.'}
    path=OUT/'manifest.json'
    if path.exists() and read_json(path)!=manifest:raise ValueError('scenario/code changed after freeze')
    write_json(path,manifest)


def distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]


def score(cases):
    records=[];task_rows={'root_canal':[],'crack':[]};ocr_edits=ocr_chars=0
    for c in cases:
        p=OUT/'reports'/(c['id']+'.json')
        if not p.exists():continue
        r=read_json(p);trace=r.get('trace',[]);expected=set(c['expected_tasks']);actual=set(r.get('tasks',[]))
        recognized=[e for e in trace if e['step']=='leader.intent' and e['status']=='recognized']
        stages={e['step'] for e in trace}
        full=all(t+'.explain' in stages for t in expected) and {'document_parse','information_review','leader'}<=stages and any(e['status']=='aggregated' for e in trace)
        success=all(r.get('results',{}).get(t,{}).get('status') in ('assessed','needs_review') for t in expected)
        explanation_ok=all(isinstance(r.get('results',{}).get(t,{}).get('explanation'),dict) for t in expected)
        all_grades=True;all_factors=True
        for task in expected:
            pred=r.get('results',{}).get(task,{}).get('factors','')
            if task=='root_canal' and isinstance(pred,dict):
                import copy
                pred=copy.deepcopy(pred)
                computed={x['tooth']:x['grade'] for x in r['results'][task].get('assessments',[])}
                from medagent.common import tooth_key
                for tooth in pred.get('toothLabel',[]):tooth['difficulty']=computed.get(tooth_key(tooth.get('position','')))
            row={'id':c['id'],'gold':c['gold'][task],'prediction':pred};task_rows[task].append(row)
            m=evaluate(task,[row],False)
            all_grades &= m['grade']['case_exact_accuracy']==1
            all_factors &= m['symptom_field_exact']['accuracy']==1
        row={'id':c['id'],'modality':c['modality'],'expected_tasks':sorted(expected),'actual_tasks':sorted(actual),
             'route_exact':expected==actual,'initial_route_exact':bool(recognized) and set(recognized[0].get('tasks',[]))==expected,
             'llm_route_used':bool(recognized),'status':r.get('status'),'stages_reached':full,'specialties_returned':success,
             'grade_exact':all_grades,'symptom_fields_exact':all_factors,'elapsed_seconds':r.get('elapsed_seconds'),
             'explanations_valid':explanation_ok,'automatic_pass':r.get('status')=='completed',
             'end_to_end_correct':expected==actual and full and success and explanation_ok and all_grades and all_factors,
             'changed_source':any(e.get('changed') for e in trace if e['step']=='information_review'),
             'rerouted':any(e['status']=='rerouted' for e in trace),'run_id':r.get('run_id')}
        if c['modality']=='image_only':
            gt=(ROOT/c['image']).with_suffix('.ocr.txt').read_text()
            # Keep punctuation, negations, tooth numbers; normalize only whitespace.
            gt=re.sub(r'\s+','',gt);pred=re.sub(r'\s+','',r.get('document',{}).get('text',''))
            edits=distance(gt,pred);ocr_edits+=edits;ocr_chars+=len(gt)
            row.update(ocr_char_edits=edits,ocr_reference_chars=len(gt),ocr_method=r.get('document',{}).get('method'))
        records.append(row)
    completed=len(records);counts={key:sum(bool(r[key]) for r in records) for key in
        ('route_exact','initial_route_exact','llm_route_used','stages_reached','specialties_returned','grade_exact','symptom_fields_exact','explanations_valid','automatic_pass','end_to_end_correct','changed_source','rerouted')}
    result={'suite':'implicit-v1','synthetic':True,'expected_cases':60,'completed':completed,'counts':counts,'cases':records,
            'graph_version':read_json(ROOT/'configs/agent.json')['graph_version'],
            'task_metrics':{t:evaluate(t,rows,False) for t,rows in task_rows.items()},
            'ocr':{'completed_images':sum(c['modality']=='image_only' for c in records),'expected_images':30,
                   'char_edits':ocr_edits,'reference_chars':ocr_chars,'cer':ocr_edits/ocr_chars if ocr_chars else None},
            'note':'Partial reports describe completed cases only; final acceptance requires all 60. Wrong routes and failed calls remain errors. needs_review is not synonymous with incorrect.'}
    write_json(OUT/'metrics.json',result)
    lines=['# 隐含需求与病历拍照：60 场景端到端验证','',f'完成 {completed}/60。仿真数据，不是临床独立测试集。所有请求不传 tasks，30 个图片场景不附转写文字。','',
           '| 指标 | 已完成病例中计数 |','|---|---:|']
    for k,v in counts.items():lines.append(f'| {k} | {v}/{completed} |')
    lines+=['','## 分项准确率','','| 专病 | 应评估病例数（最终各 40） | 症状字段准确率 | 规则等级准确率 |','|---|---:|---:|---:|']
    def pct(x):return '—' if x is None else f'{100*x:.2f}%'
    for t,m in result['task_metrics'].items():lines.append(f"| {t} | {m['samples']} | {pct(m['symptom_field_exact']['accuracy'])} | {pct(m['grade']['accuracy'])} |")
    lines += ['',f"图片 OCR：已完成 {result['ocr']['completed_images']}/30；字符错误率 CER={pct(result['ocr']['cer'])}，只忽略空白字符，保留标点与否定词。",'',
              '## 解释边界','','- stages_reached 表示走到所需分支的解释节点；节点执行并不保证输出有效。',
              '- specialties_returned 表示所需分支有评估结果；结果可能被标记需复核。',
              '- route_exact 表示最终任务集合完全一致；双任务漏掉一个即失败，额外执行一个也失败。',
              '- end_to_end_correct 要求路由正确、完整到达解释与汇总、所需分支返回、说明通过结构引用校验、所有患牙等级及症状字段均正确；不等同于临床专家验收。',
              '- 图像由病历模板和 CSS 模拟拍摄角度、阴影、轻微模糊、低对比度；不代表真实手机或手写病历分布。',
              '- 仿真标签由项目现有规则定义，未经过独立医生裁决，不能替代真实临床准确率。',
              '- 完整报告逐例保留失败与错路由，不挑选通过病例。']
    (ROOT/'docs/scenario_results.md').write_text('\n'.join(lines)+'\n')
    return result


def run():
    cases=read_json(CASES);freeze(cases)
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read_json(OUT/'state.json') if (OUT/'state.json').exists() else {'submitted':{},'errors':{}}
    concurrency=read_json(ROOT/'configs/runtime.json')['scenario_concurrency']
    assert concurrency==1, 'Scenario acceptance is sequential'
    # Interleave task and modality groups. All 60 still execute; no selection.
    queue=[cases[g*20+i] for i in range(20) for g in range(3)]
    first=['S002','S041']  # Exercise an image and a dual-specialty case first, without dropping any case.
    queue=[next(c for c in cases if c['id']==sid) for sid in first]+[c for c in queue if c['id'] not in first]
    try:
        while True:
            completed={p.stem for p in (OUT/'reports').glob('*.json')}
            pending={sid:rid for sid,rid in state['submitted'].items() if sid not in completed}
            if len(completed)==len(cases):break
            for c in queue:
                if len(pending)>=concurrency:break
                if c['id'] in state['submitted'] or c['id'] in completed:continue
                payload=request_for(c)
                try:response=call('/api/runs',payload)
                except Exception as exc:
                    state['errors'][c['id']]=str(exc);write_json(OUT/'state.json',state)
                    time.sleep(5);break
                rid=response['run_id']
                state['submitted'][c['id']]=rid;pending[c['id']]=rid;write_json(OUT/'state.json',state)
                write_json(OUT/'requests'/(c['id']+'.json'),{'run_id':rid,'keys':list(payload),
                    'text_chars':len(payload['text']),'images':len(payload['images']),
                    'request_sha256':digest(json.dumps(payload,ensure_ascii=False,sort_keys=True))})
            changed=False
            for sid,rid in list(pending.items()):
                try:r=call('/api/runs/'+rid)
                except Exception as exc:
                    state['errors'][sid]=str(exc);write_json(OUT/'state.json',state);continue
                if r['status'] not in ('queued','running'):
                    write_json(OUT/'reports'/(sid+'.json'),r);changed=True
            if changed:score(cases)
            completed={p.stem for p in (OUT/'reports').glob('*.json')}
            write_json(OUT/'status.json',{'status':'running','completed':len(completed),'total':len(cases),
                       'pending':{sid:rid for sid,rid in state['submitted'].items() if sid not in completed},
                       'pid':os.getpid(),'updated_at':time.time(),'graph_version':read_json(ROOT/'configs/agent.json')['graph_version']})
            time.sleep(5)
        result=score(cases)
        write_json(OUT/'status.json',{'status':'completed','completed':len(cases),'total':len(cases),'updated_at':time.time(),'counts':result['counts']})
    except BaseException as exc:
        write_json(OUT/'status.json',{'status':'failed','error':str(exc),'updated_at':time.time()});raise


if __name__=='__main__':
    os.umask(0o077);OUT.mkdir(parents=True,exist_ok=True,mode=0o700)
    for d in ('requests','reports'):(OUT/d).mkdir(exist_ok=True,mode=0o700)
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','run','score']);args=p.parse_args()
    if args.action=='check':freeze(read_json(CASES));print('Validated 60 cases, 30 image-only requests, disjoint request/label payloads')
    elif args.action=='score':score(read_json(CASES))
    else:run()
