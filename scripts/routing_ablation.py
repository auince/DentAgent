"""Synthetic routing-intent controls; not a medical accuracy benchmark."""
import argparse,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.agents import LeaderAgent,SharedContext
from medagent.common import ROOT,read_json,write_json,parse_json
from ablation_worker import Runner,capture,Replay,OUT
CASES=[
 ('explicit_root','评估根管治疗难度',['root_canal']),
 ('explicit_crack','评估牙隐裂风险',['crack']),
 ('explicit_both','分析根管难度和牙隐裂风险',['root_canal','crack']),
 ('implicit_root','这颗牙的首次根管操作会不会很难',['root_canal']),
 ('implicit_crack','判断这颗牙出现裂纹的风险',['crack']),
 ('comprehensive','做一下综合口腔评估',['root_canal','crack']),
 ('unsupported','请帮我评估高血压用药',[]),
 ('negated_crack','只做根管难度评估，不评估牙隐裂',['root_canal']),
 ('negated_root','只评估牙隐裂，暂不做根管难度评估',['crack']),
 ('ambiguous','帮我看看这段病历',['root_canal','crack']),
]
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=2);args=p.parse_args();os.umask(0o077)
text='主诉为右后牙咬物不适，需结合原始检查记录开展评估。';jobs=[];fixed={}
for key,instruction,expected in CASES:
 req=capture(lambda c:LeaderAgent(c).plan(instruction,text,SharedContext(key)))
 if req:jobs.append({'key':key,**req})
 else:fixed[key]=LeaderAgent(Replay('')).plan(instruction,text,SharedContext(key))
answers={}
if jobs:
 with Runner(read_json(ROOT/'configs/gateway.json')['models']['dental-vision'],args.gpu,'routing',vision=True) as run:answers=run.generate(jobs,'routing')
rows=[]
for key,instruction,expected in CASES:
 try:pred=fixed[key] if key in fixed else LeaderAgent(Replay(answers[key]['answer'])).plan(instruction,text,SharedContext(key));error=None
 except ValueError as exc:pred=[];error=str(exc)
 rows.append({'id':key,'instruction':instruction,'expected':expected,'predicted':pred,'match':set(pred)==set(expected),'method':'keyword' if key in fixed else 'model','error':error})
write_json(OUT/'routing_results.json',{'cases':rows,'correct':sum(r['match'] for r in rows),'total':len(rows),'limitations':['10 authored intent checks, not a naturalistic user benchmark.','Ambiguous request policy is defined here as both specialties and must be reviewed as a product decision.','Routing correctness does not measure diagnosis quality or multi-agent speed.']})
