"""Explicit training entry points. Does not run during service startup.

SFT/GRPO currently train the verified text label tasks. CPT requires an explicit,
authorized plain-text corpus. VLM training is intentionally gated on paired data.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json,digest
from medagent.rewards import grpo_reward


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['sft','grpo','cpt'],required=True)
    p.add_argument('--data',required=True,help='Prepared training split JSON, or authorized text corpus for CPT')
    p.add_argument('--model',default='LOCAL_RESOURCE')
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=100)
    a=p.parse_args()
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
    from trl import SFTTrainer,SFTConfig,GRPOTrainer,GRPOConfig
    if AutoConfig.from_pretrained(a.model).model_type!='qwen3':
        raise ValueError('This entry point supports text Qwen3. Paired image supervision and separate VLM validation are required for VLM training.')
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    if not isinstance(tokenizer.chat_template,str):raise ValueError('A single explicit Qwen3 chat template is required')
    # Match serving and baseline evaluation, including prompt/completion rendering.
    tokenizer.chat_template='{% set enable_thinking = false %}'+tokenizer.chat_template
    if a.mode=='cpt':
        raw=Path(a.data).read_text()
        data=Dataset.from_list([{'text':raw[i:i+3000]} for i in range(0,len(raw),3000)])
    else:
        rows=read_json(a.data)
        if not rows or any(r.get('split')!='train' for r in rows):
            raise ValueError('Explicit train split manifest required; do not supply test or validation data')
        data=Dataset.from_list(rows)
    out=Path(a.output)
    if out.exists() and any(out.iterdir()): raise ValueError('output must be new/empty')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'provenance.json',{'mode':a.mode,'model':a.model,'data':str(Path(a.data).resolve()),'data_sha256':digest(Path(a.data).read_bytes()),'max_steps':a.steps,'enable_thinking':False,'chat_template_sha256':digest(tokenizer.chat_template)})
    lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    common=dict(output_dir=str(out),max_steps=a.steps,per_device_train_batch_size=1,gradient_accumulation_steps=8,
        learning_rate=2e-5,bf16=True,gradient_checkpointing=True,save_steps=50,logging_steps=5,report_to='none')
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.bfloat16)
    if a.mode=='grpo':
        args=GRPOConfig(**common,num_generations=2,generation_batch_size=2,max_completion_length=2048)
        trainer=GRPOTrainer(model=model,args=args,train_dataset=data,processing_class=tokenizer,peft_config=lora,reward_funcs=[grpo_reward])
    else:
        extra={'dataset_text_field':'text','completion_only_loss':False} if a.mode=='cpt' else {'completion_only_loss':True}
        args=SFTConfig(**common,max_length=8192,**extra)
        trainer=SFTTrainer(model=model,args=args,train_dataset=data,processing_class=tokenizer,peft_config=lora)
    trainer.train();trainer.save_model(str(out/'final'))


if __name__=='__main__':main()
