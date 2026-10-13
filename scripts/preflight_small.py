"""CPU compatibility audit; optional GPU forward/backward is an engineering check."""
import argparse
import inspect
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json

def main(gpu_check=False):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig,AutoModelForImageTextToText,AutoProcessor
    from peft import get_peft_model
    from medagent.small_training import lora_config,CompletionCollator,load_model,enforce_devices,weighted_trainer_class
    from trl import GRPOConfig,DistillationTrainer,DPOConfig
    result={'status':'running','models':{},'gpu_check':gpu_check}
    for name in ['Qwen3.5-2B','Qwen3.5-4B']:
        path=ROOT/'runtime/small_model/models'/name
        config=AutoConfig.from_pretrained(path)
        with init_empty_weights():model=AutoModelForImageTextToText.from_config(config)
        model.tie_weights();count=sum(p.numel() for p in model.parameters())
        if count>=7_000_000_000:raise ValueError('parameter_cap_exceeded')
        model=get_peft_model(model,lora_config())
        trainable=[n for n,p in model.named_parameters() if p.requires_grad]
        if not trainable or any('visual' in n for n in trainable):raise ValueError('lora_language_target_mismatch')
        processor=AutoProcessor.from_pretrained(path)
        sample={'prompt':[{'role':'system','content':'仅提取原文事实。'},{'role':'user','content':'左边后牙喝凉水疼，没有肿。'}],
            'completion':[{'role':'assistant','content':'{"症状":"冷痛","肿胀":"阴性","牙位":null}'}]}
        batch=CompletionCollator(processor)([sample])
        if not (batch['labels']!=-100).any():raise ValueError('all_labels_masked')
        result['models'][name]={'parameters':count,'trainable_adapter_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'adapter_modules':trainable,'sample_tokens':batch['input_ids'].shape[1],'label_tokens':int((batch['labels']!=-100).sum()),'prefix_mask_verified':True}
        del model
    result['trl']={'sequence_importance': 'importance_sampling_level' in GRPOConfig.__dataclass_fields__,
        'vespo':'vespo' in inspect.getsource(GRPOConfig),'distillation':DistillationTrainer.__name__,'dpo_multi_loss':'loss_weights' in DPOConfig.__dataclass_fields__}
    if gpu_check:
        enforce_devices()
        model,_=load_model(str(ROOT/'runtime/small_model/models/Qwen3.5-4B'));model.cuda();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        batch={k:v.cuda() for k,v in batch.items()}
        trainer=object.__new__(weighted_trainer_class())
        loss=trainer.compute_loss(model,batch);loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('invalid_gradient')
        result['gpu']={'forward_backward_finite':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'physical_devices':os.environ['CUDA_VISIBLE_DEVICES']}
        model.save_pretrained(ROOT/'runtime/small_model/preflight_adapter')
    result['status']='passed';write_json(ROOT/'runtime/small_model/preflight.json',result)
    print({k:{a:b for a,b in v.items() if a!='adapter_modules'} for k,v in result['models'].items()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu-check',action='store_true');main(p.parse_args().gpu_check)
