import argparse,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
from medagent.extraction_policy import ExplicitFieldExtractSkill
from ablation_worker import Runner,capture,Replay,OUT,select
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=2);args=p.parse_args();os.umask(0o077)
rows=select()['root_canal'];jobs=[{'key':r['id'],**capture(lambda c:ExplicitFieldExtractSkill(c).run('root_canal',r['text']))} for r in rows]
model=read_json(ROOT/'configs/experiments.json')['base_model']
with Runner(model,args.gpu,'explicit_field_contract_base') as run:
 initial=run.generate(jobs,'root_canal_contract_v2_base');repairs=[]
 for r in rows:
  try:ExplicitFieldExtractSkill(Replay(initial[r['id']]['answer'])).run('root_canal',r['text'])
  except (ValueError,KeyError,TypeError) as exc:repairs.append({'key':r['id'],**capture(lambda c:ExplicitFieldExtractSkill(c).run('root_canal',r['text'],str(exc)))})
 corrected=run.generate(repairs,'root_canal_contract_v2_base_repair')
write_json(OUT/'root_canal_contract_v2_base_predictions.json',[{**r,'prediction':corrected.get(r['id'],initial[r['id']])['answer'],'initial_prediction':initial[r['id']]['answer'],'repair_attempted':r['id'] in corrected} for r in rows])
