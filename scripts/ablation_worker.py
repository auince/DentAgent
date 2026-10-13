"""Reproducible, cached module ablations using production skill prompts.
Raw cases and model responses stay in runtime (private); never uploads data.
"""
import argparse, copy, json, os, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT, digest, read_json, write_json, parse_json
from medagent.datasets import prepare
from medagent.evaluation import evaluate, grades
from medagent.rules import assess, normalize_output
from medagent.skills import ReviewSkill, SymptomExtractSkill
from medagent.retrieval import KnowledgeIndex
from medagent.gpu import reserve_specific

OUT=ROOT/'runtime/ablation'
class Captured(Exception): pass
class Capture:
    def complete(self,role,system,text,images=()):
        self.request={'role':role,'system':system,'text':text,'images':list(images)}
        raise Captured()
class Replay:
    def __init__(self,answer): self.answer=answer
    def complete(self,*args,**kwargs): return self.answer

def capture(fn):
    c=Capture()
    try: fn(c)
    except Captured: return c.request
    return None

def records(name):
    path=OUT/(name+'.jsonl')
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

def select():
    path=OUT/'selection.json'
    if path.exists(): return read_json(path)
    cfg=read_json(ROOT/'configs/experiments.json'); selected={}
    for task,tc in cfg['tasks'].items():
        rows,_=prepare(task,tc)
        rows=[r for r in rows if not any(r[k] for k in ('train_text_overlap','train_id_overlap','duplicate_test'))]
        if task=='root_canal':
            buckets={}
            for r in rows: buckets.setdefault(str(max(grades(task,r['gold']).values())),[]).append(r)
            for bucket in buckets.values():bucket.sort(key=lambda r:digest('ablation-v1:'+r['id']))
            chosen=[]
            while len(chosen)<20 and any(buckets.values()):
                for key in sorted(buckets):
                    if buckets[key] and len(chosen)<20:chosen.append(buckets[key].pop(0))
            rows=chosen
        selected[task]=rows
    write_json(path,selected)
    write_json(OUT/'manifest.json',{'selection':'root 20 round-robin by maximum gold difficulty, SHA256 id ordering; crack all 18 decontaminated cases',
        'selected_ids':{k:[r['id'] for r in v] for k,v in selected.items()},
        'selection_sha256':digest(json.dumps(selected,ensure_ascii=False,sort_keys=True)),
        'code_sha256':{str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in [Path(__file__),ROOT/'medagent/skills.py',ROOT/'medagent/rules.py',ROOT/'medagent/retrieval.py']},
        'limitations':['Stratified development ablation subset, not population accuracy.','No patient identifiers available for patient-level decontamination.','Frozen before module predictions; gold labels never enter inference prompts.']})
    return selected

class Runner:
    def __init__(self,model_path,gpu,name,vision=False):
        self.model_path,self.gpu,self.name,self.vision=model_path,gpu,name,vision
    def __enter__(self):
        self.lease=reserve_specific(self.gpu,lambda:self.status('waiting_for_gpu'))
        self.lease.__enter__();os.environ['CUDA_VISIBLE_DEVICES']=str(self.gpu)
        self.status('loading_model')
        import torch
        from transformers import AutoTokenizer,AutoProcessor,AutoModelForCausalLM,AutoModelForImageTextToText
        self.torch=torch
        self.processor=(AutoProcessor if self.vision else AutoTokenizer).from_pretrained(self.model_path,trust_remote_code=False)
        self.tokenizer=self.processor.tokenizer if self.vision else self.processor
        self.tokenizer.padding_side='left'
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        cls=AutoModelForImageTextToText if self.vision else AutoModelForCausalLM
        self.model=cls.from_pretrained(self.model_path,torch_dtype=torch.bfloat16,device_map='cuda:0',trust_remote_code=False).eval()
        return self
    def __exit__(self,*args):
        self.status('completed' if args[0] is None else 'failed');self.lease.__exit__(*args)
    def status(self,status,**kw):
        write_json(OUT/(self.name+'_status.json'),{'status':status,'gpu':self.gpu,'updated_at':time.time(),**kw})
    def generate(self,jobs,stage):
        prior=records(stage); lookup={r['key']:r for r in prior}; pending=[]
        for job in jobs:
            fp=digest(json.dumps({'request':job,'model':self.model_path,'max_tokens':4096,'do_sample':False},ensure_ascii=False,sort_keys=True))
            job={**job,'fingerprint':fp}
            if job['key'] in lookup:
                if lookup[job['key']]['fingerprint']!=fp:raise ValueError('cache_fingerprint_changed:'+job['key'])
            else:pending.append(job)
        with (OUT/(stage+'.jsonl')).open('a') as out:
            batch_size=1 if any(j['images'] for j in pending) else 4
            for offset in range(0,len(pending),batch_size):
                group=pending[offset:offset+batch_size];texts=[];images=[]
                for job in group:
                    content=job['text']
                    if job['images']:
                        if not self.vision:raise ValueError('text_model_cannot_read_images')
                        from PIL import Image
                        content=[{'type':'text','text':job['text']}]
                        for path in job['images']:
                            im=Image.open(path).convert('RGB');im.thumbnail((1536,1536));images.append(im);content.append({'type':'image'})
                    messages=[{'role':'system','content':job['system']},{'role':'user','content':content}]
                    texts.append(self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False))
                inputs=self.processor(text=texts,images=images or None,padding=True,return_tensors='pt') if self.vision else self.processor(texts,padding=True,return_tensors='pt')
                n=inputs['input_ids'].shape[1]
                if n>16000:raise ValueError('ablation_input_too_long')
                start=time.monotonic()
                with self.torch.inference_mode():ids=self.model.generate(**inputs.to('cuda:0'),do_sample=False,max_new_tokens=4096,pad_token_id=self.tokenizer.pad_token_id)
                elapsed=time.monotonic()-start;tokens=ids[:,n:].cpu().tolist();eos=self.model.generation_config.eos_token_id
                eos=set(eos if isinstance(eos,list) else [eos])
                for job,t in zip(group,tokens):
                    raw=self.tokenizer.decode(t,skip_special_tokens=True);error=None if any(x in eos for x in t) else 'output_truncated'
                    row={**job,'answer':raw if error is None else '', 'raw_answer':raw,'error':error,'model_path':self.model_path,'batch_wall_seconds':elapsed,'batch_size':len(group)}
                    out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();os.fsync(out.fileno());lookup[job['key']]=row
                self.status('running',stage=stage,completed=len(lookup),total=len(jobs));del ids,inputs
        return lookup

def extraction(task,gpu,reflow=False,variant="finetuned"):
    selected=select()[task];cfg=read_json(ROOT/'configs/experiments.json'); stage=task+('_reflow' if reflow else '_contract')+('_base' if variant=='base' else '')
    if reflow:
        reviews={r['key']:r for r in records('review')};changed=[]
        for row in selected:
            rr=reviews[task+':'+row['id']]
            try: review=ReviewSkill(Replay(rr['answer'])).run(row['text'])
            except (ValueError,TypeError,KeyError):continue
            if review['text']!=row['text']:changed.append({**row,'text':review['text']})
        selected=changed
    jobs=[]
    for row in selected:
        req=capture(lambda c:SymptomExtractSkill(c).run(task,row['text']))
        jobs.append({'key':row['id'],**req})
    if not jobs:
        write_json(OUT/(stage+'_predictions.json'),[])
        write_json(OUT/(stage+'_status.json'),{'status':'completed','total':0});return
    cached_path=OUT/(stage+'_predictions.json')
    if cached_path.exists():
        cached=read_json(cached_path)
        if [(r['id'],r['text']) for r in cached]!=[(r['id'],r['text']) for r in selected]:raise ValueError('cached_selection_changed')
        return
    with Runner(cfg['base_model'] if variant=='base' else cfg['tasks'][task]['model'],gpu,stage) as run:
        initial=run.generate(jobs,stage); repairs=[]
        for row in selected:
            try: assess(task,initial[row['id']]['answer'])
            except (ValueError,KeyError,TypeError) as exc:
                req=capture(lambda c:SymptomExtractSkill(c).run(task,row['text'],str(exc)))
                repairs.append({'key':row['id'],**req})
        repaired=run.generate(repairs,stage+'_repair')
    write_json(OUT/(stage+'_predictions.json'),[{**row,'prediction':repaired.get(row['id'],initial[row['id']])['answer'],
        'initial_prediction':initial[row['id']]['answer'],'repair_attempted':row['id'] in repaired} for row in selected])

def review_and_explain(gpu,phase='review'):
    selected=select();jobs=[]
    if phase=='review':
        for task,rows in selected.items():
            for row in rows:jobs.append({'key':task+':'+row['id'],**capture(lambda c:ReviewSkill(c).run(row['text']))})
    else:
        index=KnowledgeIndex(ROOT/'knowledge')
        for task,rows in selected.items():
            predictions=read_json(OUT/(task+'_contract_predictions.json'))
            changed={r['id']:r for r in read_json(OUT/(task+'_reflow_predictions.json'))} if (OUT/(task+'_reflow_predictions.json')).exists() else {}
            # Grounding must be recomputed for the final reviewed text version.
            for row in predictions:
                row=changed.get(row['id'],row)
                try: result=assess(task,row['prediction'])
                except (ValueError,KeyError,TypeError):continue
                key=task+':'+row['id']
                if phase=='ground':
                    req=capture(lambda c:SymptomExtractSkill(c).ground(task,row['text'],result['factors']))
                    if req:jobs.append({'key':key,**req})
                else:
                    answers={r['key']:r for r in records('ground')}
                    try: evidence,issues=SymptomExtractSkill(Replay(answers[key]['answer'])).ground(task,row['text'],result['factors'])
                    except (ValueError,KeyError,TypeError):evidence=[];issues=[{'code':'ground_failed'}]
                    query='\n'.join(str(x) for x in result['assessments'])+'\n'+row['text'][:2000]
                    refs=index.search(query,task=task,limit=4)
                    result.update(evidence=evidence,issues=result['issues']+issues)
                    for mode in ('full','no_rag','no_evidence'):
                        rr=copy.deepcopy(result);rr['knowledge']=refs if mode!='no_rag' else []
                        if mode=='no_evidence':rr['evidence']=[]
                        req=capture(lambda c:SymptomExtractSkill(c).explain(task,row['text'],rr))
                        jobs.append({'key':key+':'+mode,**req})
    if not jobs:return
    path=read_json(ROOT/'configs/gateway.json')['models']['dental-vision']
    previous={r['key']:r for r in records(phase)}
    if all(job['key'] in previous and previous[job['key']]['fingerprint']==digest(json.dumps({'request':job,'model':path,'max_tokens':4096,'do_sample':False},ensure_ascii=False,sort_keys=True)) for job in jobs):return
    with Runner(path,gpu,phase,vision=True) as run:run.generate(jobs,phase)

if __name__=='__main__':
    os.umask(0o077);OUT.mkdir(parents=True,exist_ok=True,mode=0o700)
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['select','extract','reflow','review','ground','explain']);p.add_argument('--task');p.add_argument('--variant',choices=['base','finetuned'],default='finetuned');p.add_argument('--gpu',type=int,default=2);args=p.parse_args()
    if args.phase=='select':select()
    elif args.phase in ('extract','reflow'):extraction(args.task,args.gpu,args.phase=='reflow',args.variant)
    else:review_and_explain(args.gpu,args.phase)
