"""Task-scope retrieval control on actual predicted-factor queries; not clinical relevance gold."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
from medagent.rules import assess
from medagent.retrieval import KnowledgeIndex
index=KnowledgeIndex(ROOT/'knowledge');out=ROOT/'runtime/ablation';report={'corpus_documents':len(set(c['source'] for c in index.chunks)),'corpus_chunks':len(index.chunks),'queries':[]}
for task in ('root_canal','crack'):
 for row in read_json(out/(task+'_contract_predictions.json')):
  try:result=assess(task,row['prediction'])
  except (ValueError,KeyError,TypeError):continue
  query='\n'.join(str(x) for x in result['assessments'])+'\n'+row['text'][:2000]
  for mode,scope in [('task_filter',task),('no_task_filter',None)]:
   hits=index.search(query,task=scope,limit=4)
   report['queries'].append({'task':task,'id':row['id'],'mode':mode,'hits':len(hits),
    'outside_task':sum(not h['source'].startswith(task+'/') for h in hits),
    'valid_source_offsets':sum((ROOT/'knowledge'/h['source']).read_text()[h['start']:h['end']]==h['text'] for h in hits),
    'sources':[h['source'] for h in hits]})
report['summary']={}
for mode in ('task_filter','no_task_filter'):
 rows=[r for r in report['queries'] if r['mode']==mode];report['summary'][mode]={'queries':len(rows),'hits':sum(r['hits'] for r in rows),'outside_task':sum(r['outside_task'] for r in rows),'queries_with_cross_task':sum(r['outside_task']>0 for r in rows),'valid_source_offsets':sum(r['valid_source_offsets'] for r in rows)}
report['limitations']=['Only two legacy rule documents, not a medical guideline knowledge base.','Task-scope adherence and offsets are engineering metrics, not clinical retrieval accuracy.','No gold relevance labels, so Recall@K and answer clinical quality are not claimed.']
write_json(out/'retrieval_results.json',report);print(json.dumps(report['summary'],ensure_ascii=False))
