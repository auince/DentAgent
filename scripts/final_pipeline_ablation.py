"""Follow-up experiment after observing literal fields and over-correction failures."""
import os,sys,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
from medagent.rules import assess
from medagent.skills import SymptomExtractSkill
from medagent.retrieval import KnowledgeIndex
from ablation_worker import Runner,Replay,capture,OUT
os.umask(0o077)
while not (OUT/'root_canal_contract_v2_predictions.json').exists():
 state=read_json(OUT/'explicit_field_contract_status.json') if (OUT/'explicit_field_contract_status.json').exists() else {}
 if state.get('status')=='failed':raise RuntimeError('explicit_fields_experiment_failed')
 time.sleep(15)
rows={'root_canal':read_json(OUT/'root_canal_contract_v2_predictions.json'),'crack':read_json(OUT/'crack_contract_predictions.json')}
from medagent.format_repair import close_containers
from medagent.extraction_policy import ExplicitFieldExtractSkill
for row in rows['root_canal']:
 row['before_format_repair']=row['prediction']
 initial=close_containers('root_canal',row['initial_prediction'])
 try:
  ExplicitFieldExtractSkill(Replay(initial['text'])).run('root_canal',row['text'])
  chosen=initial;row['model_repair_used']=False
 except (ValueError,KeyError,TypeError):
  chosen=close_containers('root_canal',row['prediction']);row['model_repair_used']=row['repair_attempted']
 row['prediction']=chosen['text'];row['format_repaired']=chosen['repaired']
review=read_json(OUT/'conservative_review_results.json');review={t:{r['id']:r for r in v} for t,v in review.items()};index=KnowledgeIndex(ROOT/'knowledge');jobs=[];results={}
for task,cases in rows.items():
 for row in cases:
  key=task+':'+row['id']
  try:result=assess(task,row['prediction'])
  except (ValueError,KeyError,TypeError) as exc:results[key]={'status':'failed','error':str(exc)};continue
  results[key]=result
  req=capture(lambda c:SymptomExtractSkill(c).ground(task,row['text'],result['factors']))
  if req:jobs.append({'key':key,**req})
model=read_json(ROOT/'configs/gateway.json')['models']['dental-vision']
with Runner(model,0,'pipeline_v2',vision=True) as runner:
 grounded=runner.generate(jobs,'v2_ground');explain=[]
 for task,cases in rows.items():
  for row in cases:
   key=task+':'+row['id'];result=results[key]
   if result.get('status')=='failed':continue
   try:result['evidence'],issues=SymptomExtractSkill(Replay(grounded[key]['answer'])).ground(task,row['text'],result['factors'])
   except (ValueError,KeyError,TypeError):result['evidence']=[];issues=[{'code':'evidence_failed'}]
   result['issues']+=issues
   rr=review[task][row['id']]
   if not rr.get('approved',False):result['issues'].append({'code':'unresolved_document_review'})
   query='\n'.join(str(x) for x in result['assessments'])+'\n'+row['text'][:2000]
   result['knowledge']=index.search(query,task=task,limit=4)
   for mode in ('full','no_rag','no_evidence'):
    variant=copy.deepcopy(result)
    if mode=='no_rag':variant['knowledge']=[]
    if mode=='no_evidence':variant['evidence']=[]
    req=capture(lambda c:SymptomExtractSkill(c).explain(task,row['text'],variant))
    explain.append({'key':key+':'+mode,**req})
 answers=runner.generate(explain,'v2_explain')
 for task,cases in rows.items():
  for row in cases:
   key=task+':'+row['id'];result=results[key]
   if result.get('status')=='failed':continue
   result['explanation_variants']={}
   for mode in ('full','no_rag','no_evidence'):
    variant=copy.deepcopy(result)
    if mode=='no_rag':variant['knowledge']=[]
    if mode=='no_evidence':variant['evidence']=[]
    try:result['explanation_variants'][mode]=SymptomExtractSkill(Replay(answers[key+':'+mode]['answer'])).explain(task,row['text'],variant)
    except (ValueError,KeyError,TypeError) as exc:result['explanation_variants'][mode]={'error':str(exc)}
   if 'error' in result['explanation_variants']['full']:result['issues'].append({'code':'explanation_failed'})
   if result['knowledge'] and not result['explanation_variants']['full'].get('citations'):result['issues'].append({'code':'no_citation_in_explanation'})
   result['status']='needs_review' if result['issues'] else 'assessed'
write_json(OUT/'pipeline_v2_results.json',{'rows':rows,'results':results,'review_policy':'conservative_text_v1','extraction_policy':'explicit_fields_v1','note':'Development follow-up after original ablation failures; same held development cases, not a fresh generalization test.'})
