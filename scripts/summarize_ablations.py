"""Score paired ablations, preserving failures and separating coverage from accuracy."""
import copy,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json,digest
from medagent.evaluation import evaluate
from medagent.rules import assess,normalize_output,factor_items
from medagent.skills import ReviewSkill,SymptomExtractSkill
from ablation_worker import Replay,records,OUT

def rule_rows(task,rows):
    out=[]
    for r in rows:
        r=copy.deepcopy(r)
        if task=='root_canal':
            try:
                pred=normalize_output(task,r['prediction'])
                try:computed={a['tooth']:a['grade'] for a in assess(task,pred)['assessments']}
                except (ValueError,TypeError,KeyError):computed={}
                from medagent.common import tooth_key
                for t in pred['toothLabel']:t['difficulty']=computed.get(tooth_key(t['position']))
                r['prediction']=pred
            except (ValueError,TypeError,KeyError):pass
        out.append(r)
    return out

def grade_vector(task,rows):
    from medagent.evaluation import grades
    out={}
    for row in rows:
        try:out[row['id']]=grades(task,row['prediction'])
        except (ValueError,TypeError,KeyError):out[row['id']]={}
    return out

def main():
    selected=read_json(OUT/'selection.json');report={'manifest':read_json(OUT/'manifest.json'),'tasks':{},'limitations':[
        'Only text branches tested; no verified image-text-label test pairs, so no OCR/VLM image gain claim.',
        'No human-blinded medical explanation ground truth; citation checks measure traceability, not clinical correctness.',
        'Offline batch inference durations exclude gateway model cold starts; not interactive end-to-end latency.',
        'No CPT/GRPO result included without a verified matched parent checkpoint and compatible test protocol.',
        'Ablation subset differs from the full baseline benchmark; compare only within each table.',
        'Grouped tooth records use exact identity; merged/split groups are penalized, and are separately audited.',
        'RAG is downstream of classification, so changing RAG cannot change the current pipeline classification result.'
    ]}
    reviews={r['key']:r for r in records('review')};grounds={r['key']:r for r in records('ground')};explanations={r['key']:r for r in records('explain')}
    for task,selection in selected.items():
        ids={r['id'] for r in selection};variants={};data={};evaluation=ROOT/'runtime/evaluation'
        for kind in ('base','finetuned'):
            rows=[json.loads(s) for s in (evaluation/(task+'_'+kind+'.jsonl')).read_text().splitlines()]
            data[kind+'_legacy']=[r for r in rows if r['id'] in ids]
            if len(data[kind+'_legacy'])!=len(ids):raise ValueError('baseline_incomplete:'+task+kind)
        for kind,suffix in [('base','_base'),('finetuned','')]:
            rows=read_json(OUT/(task+'_contract'+suffix+'_predictions.json'))
            data[kind+'_contract']=[{**r,'prediction':r['initial_prediction']} for r in rows]
            data[kind+'_repair']=rows
        original=data['finetuned_repair'];changes={r['id']:r for r in read_json(OUT/(task+'_reflow_predictions.json'))} if (OUT/(task+'_reflow_predictions.json')).exists() else {}
        final=[changes.get(r['id'],r) for r in original]
        data['full_no_rules']=final
        data['full_no_review']=rule_rows(task,original)
        data['full_no_repair']=rule_rows(task,[{**r,'prediction':r['initial_prediction']} for r in final])
        data['full_before_evidence_gate']=rule_rows(task,final)
        for name,rows in data.items():
            if task=='crack' and name=='full_no_rules':continue
            variants[name]=evaluate(task,rows,False)
        evidence_stats=Counter();review_stats=Counter();accept=[];accept_no_evidence=[];explain_stats={m:Counter() for m in ('full','no_rag','no_evidence')};badcases=[];issue_counts=Counter()
        for row,scored in zip(final,data['full_before_evidence_gate']):
            key=task+':'+row['id'];review_ok=False;issues=[]
            try:
                review=ReviewSkill(Replay(reviews[key]['answer'])).run(next(x['text'] for x in selection if x['id']==row['id']))
                review_ok=review['approved'] and not review['issues'];review_stats['valid']+=1;review_stats['approved']+=review['approved'];review_stats['changed']+=row['id'] in changes
            except (ValueError,KeyError,TypeError):review_stats['failed']+=1
            try:result=assess(task,row['prediction']);issues+=result['issues']
            except (ValueError,KeyError,TypeError) as exc:
                issue_counts['invalid_final_schema']+=1;badcases.append({'id':row['id'],'issues':['invalid_final_schema'],'detail':str(exc)});continue
            if review_ok and not issues:accept_no_evidence.append(scored)
            try:
                evidence,ev_issues=SymptomExtractSkill(Replay(grounds[key]['answer'])).ground(task,row['text'],result['factors'])
                evidence_stats['factors']+=len(evidence);evidence_stats['located_supported']+=sum(e['supported'] for e in evidence);issues+=ev_issues
            except (ValueError,KeyError,TypeError):
                evidence=[];issues.append({'code':'ground_failed'});evidence_stats['failed_cases']+=1
            if review_ok and not issues:accept.append(scored)
            for issue in issues:issue_counts[issue['code']]+=1
            for mode in explain_stats:
                counter=explain_stats[mode];counter['expected']+=1
                item=explanations.get(key+':'+mode)
                if not item:counter['missing']+=1;continue
                payload=json.loads(item['text']);refs=payload['references'];counter['with_references']+=bool(refs)
                rr={**result,'evidence':payload['evidence'],'knowledge':refs,'issues':payload['issues']}
                try:
                    ans=SymptomExtractSkill(Replay(item['answer'])).explain(task,row['text'],rr)
                    counter['valid_output']+=1;counter['with_citations']+=bool(ans['citations']);counter['citations']+=len(ans['citations']);counter['summary_characters']+=len(ans['summary'])
                    if refs and not ans['citations']:counter['missing_citation']+=1
                except (ValueError,KeyError,TypeError):counter['invalid_output']+=1
            metric=evaluate(task,[scored],False)
            if metric['grade']['accuracy']!=1 or metric['symptom_extraction']['fp'] or metric['symptom_extraction']['fn']:
                badcases.append({'id':row['id'],'symptom_tp_fp_fn':{k:metric['symptom_extraction'][k] for k in ('tp','fp','fn')},'grade':metric['grade'],'issues':sorted({i['code'] for i in issues})})
        # RAG and grounding alter provenance/admission, not predictions. Explicitly reuse unchanged classifications.
        pairs={}
        full=grade_vector(task,data['full_before_evidence_gate'])
        for name in ('full_no_rules','full_no_review','full_no_repair'):
            if task=='crack' and name=='full_no_rules':continue
            other=grade_vector(task,data[name]);pairs[name]={'changed_cases':sum(full[k]!=other[k] for k in full),'total_cases':len(full)}
        report['tasks'][task]={'samples':len(selection),'variants':variants,'paired_grade_changes':pairs,
            'review':dict(review_stats),'evidence':dict(evidence_stats),'issues':dict(issue_counts),'explanations':{k:dict(v) for k,v in explain_stats.items()},
            'selective_gate':{'accepted_cases':len(accept),'total_cases':len(final),'coverage':len(accept)/len(final),'accepted_metrics':evaluate(task,accept,False),
                'without_evidence_accepted_cases':len(accept_no_evidence),'without_evidence_metrics':evaluate(task,accept_no_evidence,False),
                'note':'Accuracy on accepted cases is selective accuracy, not full-test accuracy. Explanatory validation errors are separately counted.'},'badcases':badcases}
    report['evaluator_sha256']=digest((ROOT/'medagent/evaluation.py').read_bytes());write_json(OUT/'results.json',report)
    lines=['# 模块消融结果','', '本报告为固定样本、真实模型推理的开发集消融。根管 20 例按原标签最大难度轮转抽取；牙隐裂使用排除重复后的全部 18 例。它与完整模型对比的分母不同，不能直接串成一张提升曲线。','']
    names={'base_legacy':'基座 + 原提示词','finetuned_legacy':'SFT + 原提示词','base_contract':'基座 + 结构契约','base_repair':'基座 + 契约 + 有界修复','finetuned_contract':'SFT + 结构契约','finetuned_repair':'SFT + 契约 + 有界修复','full_no_review':'去除审校回流，保留规则','full_no_rules':'去除规则重算，保留审校','full_no_repair':'去除输出修复，保留审校和规则','full_before_evidence_gate':'完整提取/修复/审校/规则链路'}
    def pct(x):return '—' if x is None else f'{x*100:.2f}%'
    for task,t in report['tasks'].items():
        lines+=['## '+('根管治疗' if task=='root_canal' else '牙隐裂'),'','| 配置 | 症状字段准确率 | 症状 F1 | 分级准确率 | 完整枚举通过 |','|---|---:|---:|---:|---:|']
        for name,m in t['variants'].items():lines.append(f"| {names[name]} | {pct(m['symptom_field_exact']['accuracy'])} | {pct(m['symptom_extraction']['micro_f1'])} | {pct(m['grade']['accuracy'])} | {m['parse_schema_ok']}/{m['samples']} |")
        ev=t['evidence'];gate=t['selective_gate'];lines+=['',f"审校统计：{json.dumps(t['review'],ensure_ascii=False)}。",f"逐字证据定位通过 {ev.get('located_supported',0)}/{ev.get('factors',0)}；失败病例 {ev.get('failed_cases',0)}。证据通过不等于医学语义正确。",f"加入证据门控后可自动通过 {gate['accepted_cases']}/{gate['total_cases']}，覆盖率 {pct(gate['coverage'])}；去除证据门控为 {gate['without_evidence_accepted_cases']}/{gate['total_cases']}。需把人工复核比例一并汇报。",'', '| 说明配置 | 有效结构 | 带引用 | 无效输出 |','|---|---:|---:|---:|']
        for mode,e in t['explanations'].items():lines.append(f"| {mode} | {e.get('valid_output',0)}/{e.get('expected',0)} | {e.get('with_citations',0)} | {e.get('invalid_output',0)} |")
        lines+=['']
    lines+=['## 解释边界','']+['- '+x for x in report['limitations']]
    (ROOT/'docs/ablation_results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({task:{'variants':list(t['variants']),'evidence':t['evidence'],'review':t['review']} for task,t in report['tasks'].items()},ensure_ascii=False))
if __name__=='__main__':main()
