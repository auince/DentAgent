"""Training implementations; torch/TRL imported only inside training processes."""
import json
import os
import re
import time
from pathlib import Path
from .common import ROOT,read_json,write_json,digest
from .training_data import assert_training_rows,assert_disjoint

def enforce_devices():
    from .gpu import allowed_indices
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','')
    parts=devices.split(',')
    allowed={str(i) for i in allowed_indices()}
    if not devices or len(set(parts))!=len(parts) or not set(parts)<=allowed:
        raise RuntimeError('CUDA_VISIBLE_DEVICES must use only the configured physical GPU pool')
    expected=read_json(ROOT/'configs/small_training.json').get('execution',{}).get('training_world_size',1)
    if expected==2:
        if devices not in {'0,1','2,3'} or int(os.environ.get('WORLD_SIZE','1'))!=2:
            raise RuntimeError('training_requires_torchrun_nproc_2_and_an_allowed_physical_gpu_pair')
        import torch
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def resolve(path):
    p=Path(path);return str(p if p.is_absolute() else ROOT/p)

def lora_config():
    from peft import LoraConfig
    # Explicit language module suffixes, including the Qwen3.5 DeltaNet projection.
    # Use a regex so visual qkv/proj modules are not trained inadvertently.
    targets=r'.*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)$'
    return LoraConfig(r=32,lora_alpha=64,lora_dropout=.05,task_type='CAUSAL_LM',target_modules=targets)

def load_model(path,adapter=None,full=False):
    import torch
    from transformers import AutoModelForImageTextToText
    from peft import PeftModel,get_peft_model
    model=AutoModelForImageTextToText.from_pretrained(path,dtype=torch.bfloat16,attn_implementation='sdpa')
    count=sum(p.numel() for p in model.parameters())
    if count>=7_000_000_000:raise ValueError('online_model_exceeds_parameter_limit')
    if not full:model=PeftModel.from_pretrained(model,adapter,is_trainable=True) if adapter else get_peft_model(model,lora_config())
    model.config.use_cache=False
    return model,count

class CompletionCollator:
    def __init__(self,processor,max_length=4096,critical_weight=1.):
        self.processor=processor;self.max_length=max_length;self.critical_weight=critical_weight
    def __call__(self,rows):
        import torch
        from PIL import Image
        if len(rows)!=1:raise ValueError('multimodal_collator_requires_microbatch_one')
        row=rows[0];prompt=row['prompt'];completion=row['completion']
        if isinstance(completion,str):completion=[{'role':'assistant','content':completion}]
        messages=json.loads(json.dumps(prompt+completion))
        if row.get('images'):
            images=[Image.open(resolve(p)).convert('RGB') for p in row['images']]
            idx=max(i for i,m in enumerate(messages) if m['role']=='user')
            messages[idx]['content']=[{'type':'image','image':im} for im in images]+[{'type':'text','text':messages[idx]['content']}]
        full=self.processor.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False)
        prefix=self.processor.apply_chat_template(messages[:-len(completion)],tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True,enable_thinking=False)
        ids=full['input_ids'];n=prefix['input_ids'].shape[1]
        if ids.shape[1]>self.max_length:raise ValueError('training_sequence_exceeds_max_length; use audited chunking, never truncate labels')
        if not torch.equal(ids[0,:n],prefix['input_ids'][0]):raise ValueError('chat_template_prefix_mismatch')
        labels=ids.clone();labels[:,:n]=-100
        weights=torch.ones_like(ids,dtype=torch.float32);weights[:,:n]=0
        if self.critical_weight>1:
            target=completion[-1]['content']
            critical=re.findall(r'[1-4][1-8]|[+＋−-]|无|未|否认|\d+(?:\.\d+)?\s*(?:mm|cm|毫米|厘米|度)',target)
            tokenizer=getattr(self.processor,'tokenizer',self.processor)
            for word in critical:
                tokenids=tokenizer.encode(word,add_special_tokens=False)
                for start in range(n,ids.shape[1]-len(tokenids)+1):
                    if ids[0,start:start+len(tokenids)].tolist()==tokenids:weights[0,start:start+len(tokenids)]=self.critical_weight
        return {**full,'labels':labels,'loss_weights':weights}

def weighted_trainer_class():
    import torch
    import torch.nn.functional as F
    from transformers import Trainer
    class WeightedTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            weights=inputs.pop('loss_weights');labels=inputs.pop('labels')
            outputs=model(**inputs,use_cache=False)
            target=labels[:,1:];weight=weights[:,1:]*(target!=-100)
            # Chunk the vocabulary projection's CE consumption to limit float32 copies.
            loss=outputs.logits.new_zeros((),dtype=torch.float32)
            for start in range(0,target.shape[1],128):
                stop=min(start+128,target.shape[1])
                ce=F.cross_entropy(outputs.logits[:,start:stop].float().reshape(-1,outputs.logits.shape[-1]),target[:,start:stop].reshape(-1),reduction='none',ignore_index=-100)
                loss=loss+(ce*weight[:,start:stop].reshape(-1)).sum()
            loss=loss/weight.sum().clamp_min(1)
            return (loss,outputs) if return_outputs else loss
    return WeightedTrainer

def train(stage_id,seed=42,max_steps=-1,resume=False):
    enforce_devices()
    import torch
    from datasets import Dataset
    from transformers import AutoProcessor,AutoTokenizer,TrainingArguments,set_seed,TrainerCallback
    cfg=read_json(ROOT/'configs/small_training.json');stage=next(x for x in cfg['stages'] if x['id']==stage_id)
    method=stage['method'];out=ROOT/'runtime/small_model/checkpoints'/stage_id/str(seed)
    if (out/'complete.json').exists():return read_json(out/'complete.json')
    if out.exists() and any(out.iterdir()) and not resume:raise ValueError('nonempty_training_output_requires_resume')
    data=ROOT/'runtime/small_model/data'/stage['data']
    train_rows=read_json(data/'train.json');val_rows=read_json(data/'validation.json')
    assert_training_rows(train_rows);assert_training_rows(val_rows,'validation');assert_disjoint(train_rows,val_rows)
    if stage['data'] in {'specialist','direct_sft','short_evidence_sft','compressed_teacher_sft'}:
        from .data_coverage import audit_specialist_dataset
        audit_specialist_dataset(data)
    if stage_id=='leader_sft':
        def flatten(rows):
            return [{**r,**turn,'id':r['id']+':'+str(i)} for r in rows for i,turn in enumerate(r['trajectory'])]
        train_rows=flatten(train_rows);val_rows=flatten(val_rows)
    if method in {'contrastive','pairwise'}:
        from .retrieval_training import train_retriever
        return train_retriever(stage,train_rows,val_rows,out,seed,max_steps,resume)
    base=resolve(cfg['models'][stage['role']]);parent=None
    if stage.get('parent'):
        parent=ROOT/'runtime/small_model/checkpoints'/stage['parent']/str(seed)/'final'
        if not (parent.parent/'complete.json').exists():raise ValueError('parent_training_not_complete')
        if stage.get('full_finetune'):base=str(parent);parent=None
    set_seed(seed)
    processor=AutoProcessor.from_pretrained(base)
    tokenizer=processor.tokenizer
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    tokenizer.chat_template='{% set enable_thinking = false %}'+tokenizer.chat_template
    if isinstance(processor.chat_template,str):processor.chat_template='{% set enable_thinking = false %}'+processor.chat_template
    model,nparams=load_model(base,str(parent) if parent else None,stage.get('full_finetune',False))
    default=cfg['defaults'];lr=stage.get('learning_rate',default['learning_rate'])
    world=int(os.environ.get('WORLD_SIZE','1'))
    global_batch=stage.get('global_batch_size',cfg.get('execution',{}).get('global_batch_size',8))
    if global_batch%world:raise ValueError('global_batch_not_divisible_by_world_size')
    common=dict(output_dir=str(out),num_train_epochs=2,max_steps=max_steps,learning_rate=lr,
        per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=global_batch//world,
        bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        report_to='none',logging_steps=stage.get('logging_steps',5),save_strategy='steps',save_steps=stage.get('save_steps',50),eval_strategy='epoch',
        seed=seed,remove_unused_columns=False,save_total_limit=3,
        ddp_find_unused_parameters=bool(stage.get('full_finetune')),ddp_timeout=1800)
    provenance={'stage':stage,'base':base,'base_config_sha256':digest(Path(base,'config.json').read_bytes()),
        'train_sha256':digest((data/'train.json').read_bytes()),'validation_sha256':digest((data/'validation.json').read_bytes()),
        'seed':seed,'visible_physical_gpus':os.environ['CUDA_VISIBLE_DEVICES'],'full_parameter_count':nparams,'max_steps':max_steps,
        'world_size':world,'global_batch_size':global_batch,
        'label_quality':'research_reference_not_clinically_validated'}
    if int(os.environ.get('RANK','0'))==0:write_json(out/'provenance.json',provenance)
    class Progress(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            if state.is_world_process_zero:
                write_json(out/'progress.json',{'status':'training','stage':stage_id,'seed':seed,'step':state.global_step,'total_steps':state.max_steps,'epoch':state.epoch,'updated_at':time.time(),'physical_gpus':os.environ['CUDA_VISIBLE_DEVICES'],'world_size':world})
    if method in {'sft','vision_sft'}:
        Trainer=weighted_trainer_class()
        trainer=Trainer(model=model,args=TrainingArguments(**common),train_dataset=Dataset.from_list(train_rows),eval_dataset=Dataset.from_list(val_rows),
            data_collator=CompletionCollator(processor,stage.get('max_length',default['max_length']),stage.get('critical_weight',1.)),processing_class=processor,callbacks=[Progress()])
    elif method=='dpo':
        from trl import DPOTrainer,DPOConfig
        for r in train_rows+val_rows:
            a=len(tokenizer.encode(r['chosen'][-1]['content']));b=len(tokenizer.encode(r['rejected'][-1]['content']))
            if abs(a-b)/max(a,b,1)>.2:raise ValueError('preference_pair_length_mismatch')
        args=DPOConfig(**common,beta=.1,loss_type=['sigmoid','sft'],loss_weights=[1.,.1],max_length=default['max_length'])
        trainer=DPOTrainer(model=model,args=args,train_dataset=Dataset.from_list(train_rows),eval_dataset=Dataset.from_list(val_rows),processing_class=processor,callbacks=[Progress()])
    elif method=='opd':
        from trl import DistillationTrainer,DistillationConfig
        teacher_path=cfg['teacher'];teacher_tokenizer=AutoTokenizer.from_pretrained(teacher_path)
        if tokenizer.get_vocab()!=teacher_tokenizer.get_vocab():raise ValueError('teacher_student_tokenizer_mismatch')
        template_probe=train_rows[0]['prompt']
        if tokenizer.apply_chat_template(template_probe,tokenize=True,add_generation_prompt=True,enable_thinking=False)!=teacher_tokenizer.apply_chat_template(template_probe,tokenize=True,add_generation_prompt=True,enable_thinking=False):
            raise ValueError('teacher_student_prompt_tokenization_mismatch')
        gate=read_json(ROOT/'runtime/small_model/evaluation/teacher_gate.json')
        if gate.get('split')!='validation' or gate.get('teacher_f1',0)<=gate.get('student_f1',1):raise ValueError('teacher_not_better_on_validation')
        args=DistillationConfig(**common,max_completion_length=256,beta=.5,teacher_model_init_kwargs={'dtype':torch.bfloat16})
        trainer=DistillationTrainer(model=model,teacher_model=teacher_path,args=args,train_dataset=Dataset.from_list(train_rows),eval_dataset=Dataset.from_list(val_rows),processing_class=processor,callbacks=[Progress()])
    else:
        from trl import GRPOTrainer,GRPOConfig
        from .training_rewards import specialist_reward
        rl_common={**common,'per_device_eval_batch_size':4//world,'eval_accumulation_steps':1}
        args=GRPOConfig(**rl_common,num_generations=4,num_generations_eval=4,generation_batch_size=4,
            max_completion_length=stage.get('max_completion_length',2048 if method.startswith('agent_') else 512),beta=.02,
            importance_sampling_level='sequence' if method in {'agent_gspo','agent_vespo'} else 'token',
            loss_type='vespo' if method=='agent_vespo' else 'grpo',mask_truncated_completions=True,
            temperature=.8,generation_kwargs={'do_sample':True,'top_k':0,'top_p':1.})
        extra={};reward=specialist_reward
        if method.startswith('agent_'):
            from .agent_training import AgentRollouts,trajectory_reward
            from .evaluation_memory import memory_bounded_grpo_trainer
            GRPOTrainer=memory_bounded_grpo_trainer(GRPOTrainer)
            extra['rollout_func']=AgentRollouts(train_rows+val_rows,out/'trajectories',{**cfg,'active_stage':stage})
            reward=trajectory_reward
            # Arrow cannot represent heterogeneous tool argument shapes. Complete
            # episodes and hidden references stay inside the rollout environment.
            train_rows=[{'prompt':r['prompt']} for r in train_rows]
            val_rows=[{'prompt':r['prompt']} for r in val_rows]
        trainer=GRPOTrainer(model=model,args=args,train_dataset=Dataset.from_list(train_rows),eval_dataset=Dataset.from_list(val_rows),processing_class=processor,reward_funcs=[reward],callbacks=[Progress()],**extra)
    checkpoint=None
    if resume:
        from .checkpoint_integrity import select_checkpoint
        checkpoint,rejected=select_checkpoint(out,world,
            require_reference_adapter=(method.startswith('agent_') or method=='grpo') and not stage.get('full_finetune',False))
        if trainer.is_world_process_zero():
            run_id=os.environ.get('TORCHELASTIC_RUN_ID','standalone')
            audit={'checked_at':time.time(),'training_run_id':run_id,
                'selected':str(checkpoint) if checkpoint else None,
                'selected_step':int(checkpoint.name.split('-')[-1]) if checkpoint else 0,
                'rejected':rejected,'world_size':world,'validation':'file_structure_and_complete_payloads',
                'source_sha256':{name:digest((ROOT/'medagent'/name).read_bytes()) for name in
                    ('small_training.py','agent_training.py','training_rewards.py','checkpoint_integrity.py','evaluation_memory.py')}}
            write_json(out/'resume_audit.json',audit)
            write_json(out/'resume_history'/(run_id+'.json'),audit)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    trainer.save_model(str(out/'final'))
    if trainer.is_world_process_zero():processor.save_pretrained(out/'final')
    trainer.accelerator.wait_for_everyone()
    result={'status':'complete','stage':stage_id,'seed':seed,'steps':trainer.state.global_step,'training_only':True,'business_evaluation':'pending'}
    result['world_size']=world
    if trainer.is_world_process_zero():write_json(out/'complete.json',result)
    return result
