"""Full original-test evaluation. No sampling, decontamination filter, or gold in prompts.

Separate processes release GPU memory between stages. Skill ablations use actual
production requests; they measure offline modules, not HTTP/LangGraph latency.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT, digest, read_json, write_json
from medagent.datasets import prepare
from medagent.evaluation import evaluate
from medagent.extraction_policy import ExplicitFieldExtractSkill, TrainingAlignedExtractSkill
from medagent.format_repair import close_containers
from medagent.review_policy import ConservativeReviewSkill
from medagent.rules import assess
from medagent.skills import ReviewSkill, SymptomExtractSkill
from medagent.retrieval import KnowledgeIndex
import ablation_worker as worker
from ablation_worker import Runner, Replay, capture
from summarize_ablations import rule_rows
from schema_adapter_audit import adapt

OUT = ROOT / 'runtime/full_test'
worker.OUT = OUT
ERRORS = (ValueError, TypeError, KeyError)
POLICIES = {'selected': TrainingAlignedExtractSkill, 'explicit': ExplicitFieldExtractSkill,
            'original_contract': SymptomExtractSkill}


def freeze():
    cfg = read_json(ROOT / 'configs/experiments.json')
    protocol = read_json(ROOT / 'configs/full_test.json')
    assert protocol['sample_limit'] is None and not protocol['exclude_overlap']
    rows, audits = {}, {}
    for task, tc in cfg['tasks'].items():
        rows[task], audits[task] = prepare(task, tc)
        assert len(rows[task]) == protocol['expected_cases'][task]
        assert len({r['id'] for r in rows[task]}) == len(rows[task])
    manifest = {'protocol': protocol, 'audit': audits,
                'ids': {t: [r['id'] for r in rr] for t, rr in rows.items()},
                'data_sha256': digest(json.dumps(rows, ensure_ascii=False, sort_keys=True)),
                'code_sha256': {str(p.relative_to(ROOT)): digest(p.read_bytes())
                                for p in [Path(__file__), ROOT/'scripts/ablation_worker.py',
                                          *sorted((ROOT/'medagent').glob('*.py'))]}}
    path = OUT / 'manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Frozen full-test data/code changed; do not mix experiment versions')
    write_json(path, manifest)
    write_json(OUT / 'cases.json', rows)
    return rows


def ensure_complete(task, rows):
    expected = read_json(OUT/'cases.json')[task]
    assert [r['id'] for r in rows] == [r['id'] for r in expected], 'incomplete/reordered test'
    assert [r['gold'] for r in rows] == [r['gold'] for r in expected], 'gold changed'


def extract(task, variant, policy, gpu):
    rows = read_json(OUT/'cases.json')[task]
    cfg = read_json(ROOT/'configs/experiments.json')
    model = cfg['base_model'] if variant == 'base' else cfg['tasks'][task]['model']
    skill = POLICIES[policy]
    name = f'{task}_{variant}_{policy}'
    path = OUT/(name+'_predictions.json')
    if path.exists():
        ensure_complete(task, read_json(path)); return
    jobs = [{'key':r['id'], **capture(lambda c:skill(c).run(task,r['text']))} for r in rows]
    # Reuse only original FULL-test model predictions whose exact requests and
    # on-disk model identity still match. Never reuse historical subset answers.
    if task == 'root_canal' and policy == 'selected' and not (OUT/(name+'.jsonl')).exists():
        old = ROOT/'runtime/evaluation'/f'{task}_{variant}.jsonl'
        metadata = read_json(old.with_name(old.stem+'_manifest.json'))
        identity = {p.name:{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,
                    **({'sha256':digest(p.read_bytes())} if p.suffix=='.json' else {})}
                    for p in sorted(Path(model).iterdir()) if p.suffix in ('.json','.safetensors')}
        previous = {r['id']:r for r in map(json.loads,old.read_text().splitlines())}
        assert metadata['model_files'] == identity and metadata['model_path'] == model
        cached = []
        for row, job in zip(rows, jobs):
            r = previous[row['id']]
            assert r['text'] == row['text'] == job['text'] and r['gold'] == row['gold']
            assert r['prompt_sha256'] == digest(job['system'])
            assert r['generation']['max_new_tokens'] == 4096 and not r['generation']['do_sample']
            fp=digest(json.dumps({'request':job,'model':model,'max_tokens':4096,'do_sample':False},ensure_ascii=False,sort_keys=True))
            cached.append({**job,'fingerprint':fp,'answer':r['prediction'],
                           'reused_full_test_source':str(old),'source_experiment':r['experiment_sha256']})
        with (OUT/(name+'.jsonl')).open('w') as f:
            for r in cached:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with Runner(model,gpu,name) as runner:
        initial=runner.generate(jobs,name); repairs=[]
        for row in rows:
            try:skill(Replay(initial[row['id']]['answer'])).run(task,row['text'])
            except ERRORS as exc:
                repairs.append({'key':row['id'],**capture(lambda c:skill(c).run(task,row['text'],str(exc)))})
        corrected=runner.generate(repairs,name+'_retry')
    results=[]
    for row in rows:
        raw=initial[row['id']]['answer']; chosen=corrected.get(row['id'],initial[row['id']])['answer']
        # Only the root selected/explicit skills apply deterministic brace repair.
        fixed=close_containers(task,raw)['text'] if task=='root_canal' and policy!='original_contract' else raw
        final=close_containers(task,chosen)['text'] if task=='root_canal' and policy!='original_contract' else chosen
        try:skill(Replay(chosen)).run(task,row['text']);error=None
        except ERRORS as exc:error=str(exc)
        results.append({**row,'prediction':final,'initial_prediction':raw,'format_prediction':fixed,
                        'repair_attempted':row['id'] in corrected,'skill_error':error})
    ensure_complete(task,results);write_json(path,results)


def review(gpu):
    rows=read_json(OUT/'cases.json'); jobs=[]
    policies={'conservative':ConservativeReviewSkill,'original':ReviewSkill}
    for task, rr in rows.items():
        for row in rr:
            for policy,skill in policies.items():
                jobs.append({'key':task+':'+row['id']+':'+policy,**capture(lambda c:skill(c).run(row['text']))})
    model=read_json(ROOT/'configs/gateway.json')['models']['dental-vision']
    with Runner(model,gpu,'review',vision=True) as runner:answers=runner.generate(jobs,'review')
    result={}
    for task,rr in rows.items():
        for row in rr:
            for policy,skill in policies.items():
                key=task+':'+row['id']+':'+policy
                try:result[key]=skill(Replay(answers[key]['answer'])).run(row['text'])
                except ERRORS as exc:result[key]={'text':row['text'],'approved':False,'error':str(exc)}
    write_json(OUT/'reviews.json',result)


def pipeline(task,gpu):
    rows=read_json(OUT/f'{task}_finetuned_selected_predictions.json')
    reviews=read_json(OUT/'reviews.json');results={};jobs=[];index=KnowledgeIndex(ROOT/'knowledge')
    for row in rows:
        key=row['id']
        try:
            if row['skill_error']:raise ValueError(row['skill_error'])
            result=assess(task,row['prediction']);results[key]=result
        except ERRORS as exc:results[key]={'status':'failed','error':str(exc)};continue
        req=capture(lambda c:SymptomExtractSkill(c).ground(task,row['text'],result['factors']))
        if req:jobs.append({'key':key,**req})
    model=read_json(ROOT/'configs/gateway.json')['models']['dental-vision']
    retrieval=[]
    with Runner(model,gpu,task+'_pipeline',vision=True) as runner:
        answers=runner.generate(jobs,task+'_ground');explain=[]
        for row in rows:
            key=row['id'];result=results[key]
            if result.get('status')=='failed':continue
            try:
                if key not in answers:result['evidence'],issues=[],[]
                else:result['evidence'],issues=SymptomExtractSkill(Replay(answers[key]['answer'])).ground(task,row['text'],result['factors'])
            except ERRORS as exc:result['evidence']=[];issues=[{'code':'evidence_failed','detail':str(exc)}]
            result['issues']+=issues
            if not reviews[task+':'+key+':conservative'].get('approved'):
                result['issues'].append({'code':'unresolved_document_review'})
            query='\n'.join(str(a) for a in result['assessments'])+'\n'+row['text'][:2000]
            result['knowledge']=index.search(query,task=task,limit=4)
            retrieval.append({'id':key,'filtered':result['knowledge'],'unfiltered':index.search(query,limit=4)})
            for mode in ('full','no_rag','no_evidence','no_review'):
                rr=copy.deepcopy(result)
                if mode=='no_rag':rr['knowledge']=[]
                if mode=='no_evidence':
                    rr['evidence']=[];rr['issues']=[i for i in rr['issues'] if i['code'] not in ('unsupported_factor','evidence_failed')]
                if mode=='no_review':rr['issues']=[i for i in rr['issues'] if i['code']!='unresolved_document_review']
                req=capture(lambda c:SymptomExtractSkill(c).explain(task,row['text'],rr))
                explain.append({'key':key+':'+mode,**req})
        explained=runner.generate(explain,task+'_explain')
        for row in rows:
            result=results[row['id']]
            if result.get('status')=='failed':continue
            result['explanation_variants']={}
            for mode in ('full','no_rag','no_evidence','no_review'):
                item=explained[row['id']+':'+mode];payload=json.loads(item['text'])
                rr={**result,'knowledge':payload['references'],'evidence':payload['evidence'],'issues':payload['issues']}
                try:answer=SymptomExtractSkill(Replay(item['answer'])).explain(task,row['text'],rr)
                except ERRORS as exc:answer={'error':str(exc)}
                result['explanation_variants'][mode]=answer
            full=result['explanation_variants']['full']
            if 'error' in full:result['issues'].append({'code':'explanation_failed'})
            if result['knowledge'] and not full.get('citations'):result['issues'].append({'code':'no_citation_in_explanation'})
            result['status']='needs_review' if result['issues'] else 'assessed'
    ensure_complete(task,rows)
    write_json(OUT/(task+'_pipeline.json'),{'rows':rows,'results':results,'retrieval':retrieval})


def report():
    cases=read_json(OUT/'cases.json');result={'manifest':read_json(OUT/'manifest.json'),'tasks':{}}
    reviews=read_json(OUT/'reviews.json')
    lines=['# 完整测试集模型对比与模块消融','','统一使用根管 101 例、牙隐裂 20 例原始测试集。不抽样、不剔除失败或重叠病例。牙隐裂 2 例与训练候选文本重叠，结果可能偏乐观。','',
           '以下为实际模型输出的离线模块对照，不是交互延迟或临床专家验收。根管的旧等级仍按原协议屏蔽，避免标签泄漏。','']
    pct=lambda v:'—' if v is None else f'{v*100:.2f}%'
    for task,expected in cases.items():
        variants={}
        for kind in ('base','finetuned'):
            old=[json.loads(s) for s in (ROOT/'runtime/evaluation'/f'{task}_{kind}.jsonl').read_text().splitlines()]
            ensure_complete(task,old);variants[kind+'_legacy_raw']=evaluate(task,old,False)
            if task=='root_canal':
                adapted=[]
                for row in old:
                    try:pred,_=adapt(row['prediction'])
                    except ERRORS:pred=''
                    adapted.append({**row,'prediction':pred})
                variants[kind+'_legacy_alias']=evaluate(task,adapted,False)
        for path in sorted(OUT.glob(task+'_*_predictions.json')):
            rows=read_json(path);ensure_complete(task,rows);name=path.stem.removeprefix(task+'_').removesuffix('_predictions')
            for stage,field in [('raw','initial_prediction'),('format','format_prediction'),('retry','prediction')]:
                rr=[{**r,'prediction':r[field]} for r in rows]
                variants[name+'_'+stage]=evaluate(task,rr,False)
                if task=='root_canal':variants[name+'_'+stage+'_rules']=evaluate(task,rule_rows(task,rr),False)
        bundle=read_json(OUT/(task+'_pipeline.json'));counts=Counter(total=len(expected));issues=Counter()
        explanation={m:Counter(total=len(expected)) for m in ('full','no_rag','no_evidence','no_review')}
        for r in bundle['results'].values():
            if r['status']=='failed':counts['failed']+=1;continue
            codes=[i['code'] for i in r['issues']];issues.update(codes)
            counts['accepted']+=not codes
            counts['without_evidence_gate']+=not any(c not in ('unsupported_factor','evidence_failed') for c in codes)
            counts['without_review_gate']+=not any(c!='unresolved_document_review' for c in codes)
            counts['factors']+=len(r['evidence']);counts['supported_factors']+=sum(e['supported'] for e in r['evidence'])
            for mode,answer in r['explanation_variants'].items():
                explanation[mode]['attempted']+=1;explanation[mode]['valid']+='error' not in answer
                explanation[mode]['cited']+=bool(answer.get('citations'))
        review_stats={}
        for policy in ('original','conservative'):
            counter=Counter(total=len(expected))
            for row in expected:
                review=reviews[task+':'+row['id']+':'+policy]
                counter['valid']+='error' not in review
                counter['approved']+=bool(review.get('approved'))
                counter['changed_text']+=review.get('text',row['text'])!=row['text']
            review_stats[policy]=dict(counter)
        retrieval_stats={}
        for mode in ('filtered','unfiltered'):
            hits=[hit for row in bundle['retrieval'] for hit in row[mode]]
            retrieval_stats[mode]={'total_cases':len(expected),'executed_queries':len(bundle['retrieval']),
                                   'hits':len(hits),'cross_task_hits':sum(not hit['source'].startswith(task+'/') for hit in hits)}
        result['tasks'][task]={'variants':variants,'gate':dict(counts),'issues':dict(issues),
                              'review':review_stats,'retrieval':retrieval_stats,
                              'explanations':{m:dict(c) for m,c in explanation.items()}}
        lines += [f'## {task}（{len(expected)} 例）','','| 配置 | 症状字段准确率 | 症状 F1 | 分级准确率 |','|---|---:|---:|---:|']
        for name,m in variants.items():lines.append(f"| {name} | {pct(m['symptom_field_exact']['accuracy'])} | {pct(m['symptom_extraction']['micro_f1'])} | {pct(m['grade']['accuracy'])} |")
        lines += ['',f"整例自动通过 {counts['accepted']}/{len(expected)}；输出失败 {counts['failed']}/{len(expected)}；逐字证据通过 {counts['supported_factors']}/{counts['factors']}。",'',
                  '| 说明配置 | 有效输出/完整病例数 | 实际执行病例数 | 带引用 |','|---|---:|---:|---:|']
        for mode,c in explanation.items():lines.append(f"| {mode} | {c['valid']}/{len(expected)} | {c['attempted']} | {c['cited']} |")
        lines += ['','| 审校策略 | 结构有效 | 标记通过 | 改写原文 |','|---|---:|---:|---:|']
        for policy,c in review_stats.items():lines.append(f"| {policy} | {c.get('valid',0)}/{len(expected)} | {c.get('approved',0)} | {c.get('changed_text',0)} |")
        lines += ['','| 检索配置 | 实际查询/完整病例数 | 返回片段 | 跨专病片段 |','|---|---:|---:|---:|']
        for mode,c in retrieval_stats.items():lines.append(f"| {mode} | {c['executed_queries']}/{len(expected)} | {c['hits']} | {c['cross_task_hits']} |")
    lines += ['','## 口径说明','','- raw/format/retry 分别表示原始输出、确定性格式修复、最多一次模型重试；rules 表示规则重算。',
              '- legacy_alias 是同一确定性字段适配同时用于两模型的辅助指标；原始字段不兼容不能解读成医学能力为零。',
              '- 因素证据、引用合法性与标签准确率分开报告；RAG 位于分级之后，不能改变当前分类结果。',
              '- 格式修复和重试对照使用相同提示词与原始请求；失败记录仍进入全部准确率分母。',
              '- 图片缺少覆盖完整测试集的图文标签配对，无法给出全量 OCR/VLM 图像增益；不以少量图片另建子集替代。',
              '- 路由与并行执行需要匹配的任务意图/时延评测；这些病例标签不能支持复杂用户意图准确率或多 Agent 加速结论。',
              '- 当前未提供 CPT、VLM 微调或 RL 的完整测试集实验，不宣称其收益。']
    write_json(OUT/'results.json',result)
    (ROOT/'docs/full_test_results.md').write_text('\n'.join(lines)+'\n')


def run_suite():
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    freeze()
    # Each slot is a normal evaluation process, not another autonomous agent.
    stages=[[
        ['extract','--task','root_canal','--variant','finetuned','--policy','selected'],
        ['extract','--task','crack','--variant','finetuned','--policy','selected'],
        ['review'],
        ['extract','--task','root_canal','--variant','base','--policy','selected'],
        ['extract','--task','root_canal','--variant','finetuned','--policy','explicit'],
        ['extract','--task','root_canal','--variant','base','--policy','explicit'],
        ['extract','--task','crack','--variant','base','--policy','selected'],
        ['extract','--task','root_canal','--variant','finetuned','--policy','original_contract'],
    ],[['pipeline','--task','root_canal'],['pipeline','--task','crack']]]
    active={};completed=[]
    try:
        for stage,jobs in enumerate(stages):
            queue=list(jobs)
            while queue or active:
                for gpu in (0,1):
                    if gpu not in active and queue:
                        job=queue.pop(0);name='_'.join(x for x in job if not x.startswith('--'))
                        log=(OUT/(name+'.log')).open('ab')
                        process=subprocess.Popen([sys.executable,__file__,*job,'--gpu',str(gpu)],stdout=log,stderr=log)
                        log.close();active[gpu]=(process,name)
                write_json(OUT/'status.json',{'status':'running','stage':stage,'pid':os.getpid(),'updated_at':time.time(),
                    'cases':{'root_canal':101,'crack':20},'completed_jobs':completed,
                    'active':{str(g):{'job':n,'pid':p.pid} for g,(p,n) in active.items()},'queued_jobs':len(queue)})
                time.sleep(5)
                for gpu,(process,name) in list(active.items()):
                    if process.poll() is not None:
                        if process.returncode:raise RuntimeError('failed full-test job: '+name)
                        completed.append(name);del active[gpu]
        report()
        write_json(OUT/'status.json',{'status':'completed','cases':{'root_canal':101,'crack':20},'updated_at':time.time(),'completed_jobs':completed})
    except BaseException as exc:
        for process,_ in active.values():
            if process.poll() is None:process.terminate()
        write_json(OUT/'status.json',{'status':'failed','error':str(exc),'updated_at':time.time()});raise


if __name__=='__main__':
    os.umask(0o077);OUT.mkdir(parents=True,exist_ok=True,mode=0o700)
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','extract','review','pipeline','report','all'])
    p.add_argument('--task',choices=['root_canal','crack']);p.add_argument('--variant',choices=['base','finetuned'],default='finetuned')
    p.add_argument('--policy',choices=list(POLICIES),default='selected');p.add_argument('--gpu',type=int,default=1);a=p.parse_args()
    if a.phase=='freeze':freeze()
    elif a.phase=='extract':extract(a.task,a.variant,a.policy,a.gpu)
    elif a.phase=='review':review(a.gpu)
    elif a.phase=='pipeline':pipeline(a.task,a.gpu)
    elif a.phase=='report':report()
    else:run_suite()
