"""Persistent one-shot baseline/finetune comparison; waits for a genuinely idle GPU."""
import argparse
import fcntl
import gc
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT, digest, read_json, write_json
from medagent.datasets import prepare
from medagent.evaluation import evaluate
from medagent.gpu import reserve_idle, reserve_specific, exclusive, snapshot

OUT=ROOT/'runtime/evaluation'
JOB_NAME=None


def update(**fields):
    write_json(OUT/('jobs/'+JOB_NAME+'.json' if JOB_NAME else 'status.json'),{'updated_at':time.time(),**fields})


def load_progress(path):
    if not path.exists(): return []
    content=path.read_bytes();lines=content.splitlines(keepends=True);rows=[];offset=0
    for i,line in enumerate(lines):
        try: rows.append(json.loads(line))
        except (ValueError,UnicodeDecodeError):
            if i==len(lines)-1 and not line.endswith(b'\n'):
                # Preserve an interrupted tail and resume only complete records.
                path.with_suffix('.interrupted-tail').write_bytes(line)
                with path.open('r+b') as f:f.truncate(offset)
                break
            raise
        offset+=len(line)
    return rows


def infer_job(task,variant,config,rows,gpu_index=None):
    cfg=config['tasks'][task]; model_path=config['base_model'] if variant=='base' else cfg['model']
    prompt=(ROOT/cfg['prompt']).read_text()
    if task=='root_canal': prompt+='\n本次任务是 /json。只输出包含 personLabel 和 toothLabel 的 JSON 对象，不输出分析过程。'
    settings=config['generation']
    model_dir=Path(model_path)
    identity={p.name:{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,
        **({'sha256':digest(p.read_bytes())} if p.suffix=='.json' else {})}
        for p in sorted(model_dir.iterdir()) if p.suffix in ('.json','.safetensors')}
    fingerprint=digest(json.dumps({'model':model_path,'model_files':identity,'prompt':prompt,'generation':settings,
        'rows':[(r['id'],r['text'],r['gold']) for r in rows]},ensure_ascii=False,sort_keys=True))
    path=OUT/(task+'_'+variant+'.jsonl')
    done=load_progress(path)
    for record in done:
        if record.get('experiment_sha256')!=fingerprint: raise RuntimeError('experiment_changed: use a new output directory')
    write_json(OUT/(task+'_'+variant+'_manifest.json'),{'experiment_sha256':fingerprint,
        'model_path':model_path,'model_files':identity,'prompt_sha256':digest(prompt),'generation':settings})
    if len(done)==len(rows): return done
    completed_ids={r['id'] for r in done}
    def waiting():
        update(status='waiting_for_gpu',task=task,variant=variant,completed=len(done),total=len(rows),gpu=snapshot())
    lease=reserve_specific(gpu_index,waiting) if gpu_index is not None else reserve_idle(waiting)
    with lease as gpu:
        os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        update(status='loading_model',task=task,variant=variant,gpu_index=gpu,completed=len(done),total=len(rows))
        tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map='cuda:0',trust_remote_code=False)
        model.eval()
        tokenizer.padding_side='left'
        if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
        remaining=[r for r in rows if r['id'] not in completed_ids]
        with path.open('a') as out:
            path.chmod(0o600)
            def save(row,prediction,error,elapsed,batch_size):
                record={**row,'prediction':prediction if not error else '',
                    'raw_prediction':prediction,'error':error,'batch_wall_seconds':elapsed,'batch_size':batch_size,
                    'experiment_sha256':fingerprint,'model_path':model_path,'prompt_sha256':digest(prompt),'generation':settings}
                out.write(json.dumps(record,ensure_ascii=False)+'\n');out.flush();os.fsync(out.fileno())
                done.append(record)
                update(status='running',task=task,variant=variant,gpu_index=gpu,completed=len(done),total=len(rows))
            for offset in range(0,len(remaining),settings.get('batch_size',4)):
                group=remaining[offset:offset+settings.get('batch_size',4)]
                texts=[];valid=[]
                for row in group:
                    messages=[{'role':'system','content':prompt},{'role':'user','content':row['text']}]
                    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    if len(tokenizer.encode(rendered))>settings['max_input_tokens']:
                        save(row,'','input_too_long',0,1)
                    else: texts.append(rendered);valid.append(row)
                if not valid: continue
                inputs=tokenizer(texts,return_tensors='pt',padding=True).to('cuda:0')
                n=inputs['input_ids'].shape[1];start=time.monotonic()
                with torch.inference_mode():
                    ids=model.generate(**inputs,do_sample=False,max_new_tokens=settings['max_new_tokens'],pad_token_id=tokenizer.pad_token_id)
                generated=ids[:,n:].cpu().tolist()
                eos=model.generation_config.eos_token_id
                eos=set(eos if isinstance(eos,list) else [eos])
                for row,tokens in zip(valid,generated):
                    ended=any(token in eos for token in tokens)
                    prediction=tokenizer.decode(tokens,skip_special_tokens=True)
                    save(row,prediction,'output_truncated' if not ended and len(tokens)>=settings['max_new_tokens'] else None,time.monotonic()-start,len(valid))
                del ids,inputs
        update(status='completed',task=task,variant=variant,gpu_index=gpu,completed=len(done),total=len(rows))
        del model,tokenizer;gc.collect();torch.cuda.empty_cache()
    return done


def main():
    OUT.mkdir(parents=True,exist_ok=True,mode=0o700)
    lock=(OUT/'worker.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('evaluation worker already running')
    config=read_json(ROOT/'configs/experiments.json')
    if (OUT/'comparison.json').exists():
        update(status='completed',report='comparison.json');return
    prepared={};audits={}
    for task,cfg in config['tasks'].items(): prepared[task],audits[task]=prepare(task,cfg)
    write_json(OUT/'dataset_audit.json',audits)
    # Four subprocesses on separate GPUs; parent holds the application GPU gate.
    import subprocess
    jobs=[(task,variant) for task in prepared for variant in ('base','finetuned')]
    with exclusive():
        processes={}
        try:
            for index,(task,variant) in enumerate(jobs):
                name=task+'_'+variant
                write_json(OUT/'jobs'/(name+'.json'),{'status':'starting','task':task,'variant':variant,'completed':0,'total':len(prepared[task])})
                log=(OUT/'jobs'/(name+'.log')).open('ab')
                processes[name]=subprocess.Popen([sys.executable,__file__,'--job',task,variant,'--gpu',str(index%2)],stdout=log,stderr=log)
                log.close()
            while any(p.poll() is None for p in processes.values()):
                state={name:read_json(OUT/'jobs'/(name+'.json')) for name in processes}
                update(status='running',jobs=state,completed=sum(v.get('completed',0) for v in state.values()),total=sum(len(prepared[t]) for t,v in jobs))
                time.sleep(10)
            failed={name:p.returncode for name,p in processes.items() if p.returncode}
            if failed: raise RuntimeError('failed_jobs:'+json.dumps(failed))
        except BaseException:
            for child in processes.values():
                if child.poll() is None:child.terminate()
            raise
    report={'dataset_audit':audits,'results':{},'metric_note':'Strict schema; missing and extra teeth both penalized; symptom F1 is not accuracy.'}
    for task in prepared:
        results={}
        for variant in ('base','finetuned'):
            rows=[json.loads(x) for x in (OUT/(task+'_'+variant+'.jsonl')).read_text().splitlines()]
            clean=[r for r in rows if not r['train_text_overlap'] and not r['train_id_overlap'] and not r['duplicate_test']]
            results[variant]={'all':evaluate(task,rows),'decontaminated':evaluate(task,clean)}
        for subset in ('all','decontaminated'):
            a,b=results['base'][subset],results['finetuned'][subset]
            def delta(x,y): return 100*(y-x) if x is not None and y is not None else None
            results[subset+'_delta_pp']={'symptom_micro_f1':delta(a['symptom_extraction']['micro_f1'],b['symptom_extraction']['micro_f1']),
                'symptom_field_accuracy':delta(a['symptom_field_exact']['accuracy'],b['symptom_field_exact']['accuracy']),
                'grade_accuracy':delta(a['grade']['accuracy'],b['grade']['accuracy'])}
        report['results'][task]=results
    write_json(OUT/'comparison.json',report)
    update(status='completed',report='comparison.json')


if __name__=='__main__':
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument('--job',nargs=2);p.add_argument('--gpu',type=int);args=p.parse_args()
    try:
        if args.job:
            task,variant=args.job;JOB_NAME=task+'_'+variant;config=read_json(ROOT/'configs/experiments.json')
            rows,_=prepare(task,config['tasks'][task]);infer_job(task,variant,config,rows,args.gpu)
        else: main()
    except Exception as exc:
        update(status='failed',error=str(exc)[:600]);raise
