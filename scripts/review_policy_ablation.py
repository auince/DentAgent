import argparse,os,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
from medagent.review_policy import ConservativeReviewSkill
from ablation_worker import Runner,capture,Replay,OUT,select
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=2);args=p.parse_args();os.umask(0o077)
jobs=[];selected=select()
for task,rows in selected.items():
 for row in rows:jobs.append({'key':task+':'+row['id'],**capture(lambda c:ConservativeReviewSkill(c).run(row['text']))})
model=read_json(ROOT/'configs/gateway.json')['models']['dental-vision']
with Runner(model,args.gpu,'conservative_review',vision=True) as runner:answers=runner.generate(jobs,'conservative_review')
results={}
for task,rows in selected.items():
 results[task]=[]
 for row in rows:
  try:
   out=ConservativeReviewSkill(Replay(answers[task+':'+row['id']]['answer'])).run(row['text'])
   results[task].append({'id':row['id'],**out})
  except (ValueError,TypeError,KeyError) as exc:results[task].append({'id':row['id'],'error':str(exc)})
write_json(OUT/'conservative_review_results.json',results)
