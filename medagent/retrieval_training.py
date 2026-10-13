"""Qwen retrieval supervision: multi-positive InfoNCE and yes/no pairwise ranking."""
import math
import os
import time
import random
from pathlib import Path
from .common import write_json

def ranking_prompt(query,document):
    return '<|im_start|>system\nJudge whether the document answers the query. Answer yes or no.<|im_end|>\n<|im_start|>user\n<Query>: '+query+'\n<Document>: '+document+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

def train_retriever(stage,rows,validation,out,seed,max_steps,resume):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel,AutoModelForCausalLM,AutoTokenizer
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    world=int(os.environ.get('WORLD_SIZE','1'));rank_id=int(os.environ.get('RANK','0'));local_rank=int(os.environ.get('LOCAL_RANK','0'))
    if world!=2:raise ValueError('retrieval_training_requires_two_ranks')
    if not dist.is_initialized():dist.init_process_group('nccl')
    torch.cuda.set_device(local_rank);device=torch.device('cuda',local_rank)
    rng=random.Random(seed);torch.manual_seed(seed)
    tokenizer=AutoTokenizer.from_pretrained(stage['model'],padding_side='left')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    is_rank=stage['method']=='pairwise'
    model=(AutoModelForCausalLM if is_rank else AutoModel).from_pretrained(stage['model'],dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
    raw_model=model
    model=DistributedDataParallel(raw_model,device_ids=[local_rank],broadcast_buffers=False)
    out.mkdir(parents=True,exist_ok=True);start=0
    state_path=out/'resume.pt'
    if resume and state_path.exists():
        state=torch.load(state_path,map_location='cpu',weights_only=False);raw_model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start=state['step'];rng.setstate(state['rng'])
    def embed(texts):
        batch=tokenizer(texts,padding=True,truncation=True,max_length=768,return_tensors='pt').to(device)
        h=(model if model.training else raw_model)(**batch).last_hidden_state[:,-1]
        return F.normalize(h.float(),dim=-1)
    def rank(query,documents):
        prompts=[ranking_prompt(query,d) for d in documents]
        batch=tokenizer(prompts,padding=True,truncation=True,max_length=1024,return_tensors='pt').to(device)
        logits=(model if model.training else raw_model)(**batch).logits[:,-1].float()
        yes=tokenizer.encode('yes',add_special_tokens=False);no=tokenizer.encode('no',add_special_tokens=False)
        if len(yes)!=1 or len(no)!=1:raise ValueError('reranker_yes_no_tokenization')
        return logits[:,yes[0]]-logits[:,no[0]]
    def loss(row):
        pos=row['positives'];neg=row['negatives'];documents=pos+neg
        if not pos or not neg:raise ValueError('retrieval_missing_positive_or_negative')
        if is_rank:scores=rank(row['query'],documents)
        else:
            vectors=embed(['Instruct: Retrieve oral medical evidence relevant to the patient query.\nQuery: '+row['query']]+documents)
            scores=(vectors[:1]@vectors[1:].T)[0]/.05
        if is_rank:return F.softplus(-(scores[:len(pos),None]-scores[None,len(pos):])).mean()
        return torch.logsumexp(scores,0)-torch.logsumexp(scores[:len(pos)],0)
    steps=max_steps if max_steps>0 else math.ceil(len(rows)/world)*2
    order=list(range(len(rows)));rng.shuffle(order)
    if resume and state_path.exists():order=state.get('order',order)
    best=state.get('best',math.inf) if resume and state_path.exists() else math.inf
    for step in range(start,steps):
        model.train();optimizer.zero_grad();value=loss(rows[order[(step*world+rank_id)%len(order)]]);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if (step+1)%100==0 or step+1==steps:
            model.eval()
            with torch.no_grad():
                local=validation[rank_id::world]
                aggregate=torch.tensor([sum(float(loss(r)) for r in local),len(local)],device=device,dtype=torch.float64)
                dist.all_reduce(aggregate)
                v=float(aggregate[0]/aggregate[1])
            if rank_id==0:write_json(out/'status.json',{'step':step+1,'steps':steps,'validation_ranking_loss':v,'world_size':world,'updated_at':time.time()})
            if v<best:
                best=v
                if rank_id==0:raw_model.save_pretrained(out/'final');tokenizer.save_pretrained(out/'final')
            if rank_id==0:torch.save({'model':raw_model.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,'rng':rng.getstate(),'best':best,'order':order},state_path)
            dist.barrier()
    result={'status':'complete','steps':steps,'index_rebuild_required':not is_rank,'business_evaluation':'pending'}
    result['world_size']=world
    if rank_id==0:write_json(out/'complete.json',result)
    dist.barrier();return result
