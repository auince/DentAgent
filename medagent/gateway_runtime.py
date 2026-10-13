"""Warm model cache with a single GPU-resident model and CPU-resident inactive weights."""
import base64,copy,gc,io,json,time
from .common import ROOT,read_json


class InferenceCache:
    def __init__(self):
        import torch
        self.torch=torch;self.models={};self.active=None

    def generate(self,request):
        from transformers import AutoConfig,AutoTokenizer,AutoProcessor,AutoModelForCausalLM,AutoModelForImageTextToText
        torch=self.torch;name=request['model'];started=time.monotonic()
        model_path=read_json(ROOT/'configs/gateway.json')['models'][name]
        if self.active and self.active!=name:
            self.models[self.active]['model'].to('cpu');gc.collect();torch.cuda.empty_cache()
        if name not in self.models:
            config=AutoConfig.from_pretrained(model_path,trust_remote_code=False)
            vision=config.model_type in ('qwen3_vl','qwen2_5_vl')
            processor=(AutoProcessor if vision else AutoTokenizer).from_pretrained(model_path,trust_remote_code=False)
            cls=AutoModelForImageTextToText if vision else AutoModelForCausalLM
            model=cls.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map='cpu',trust_remote_code=False).eval()
            self.models[name]={'model':model,'processor':processor,'vision':vision}
        item=self.models[name];model=item['model'];processor=item['processor'];vision=item['vision']
        if self.active!=name:model.to('cuda:0')
        self.active=name;messages=copy.deepcopy(request['messages']);images=[]
        for msg in messages:
            if isinstance(msg['content'],list):
                content=[]
                for part in msg['content']:
                    if part['type']=='image_url':
                        if not vision:raise ValueError('text_model_cannot_accept_images')
                        from PIL import Image
                        url=part['image_url']['url']
                        if not url.startswith('data:image/'):raise ValueError('remote_image_not_allowed')
                        im=Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1],validate=True))).convert('RGB')
                        im.thumbnail((1536,1536));images.append(im);content.append({'type':'image'})
                    else:content.append(part)
                msg['content']=content
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=processor(text=[text],images=images or None,return_tensors='pt') if vision else processor(text,return_tensors='pt')
        count=inputs['input_ids'].shape[1]
        if count>16000:raise ValueError('input_too_long')
        maximum=min(int(request.get('max_tokens',4096)),4096)
        with torch.inference_mode():ids=model.generate(**inputs.to('cuda:0'),max_new_tokens=maximum,do_sample=False)
        tokens=ids[0,count:].cpu().tolist()
        tokenizer=processor.tokenizer if vision else processor
        completion=tokenizer.decode(tokens,skip_special_tokens=True)
        eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        ended=any(t in eos for t in tokens)
        result={'choices':[{'message':{'role':'assistant','content':completion},'finish_reason':'stop' if ended else 'length'}],
                'usage':{'prompt_tokens':count,'completion_tokens':len(tokens)},
                'runtime':{'cached_models':list(self.models),'elapsed_seconds':round(time.monotonic()-started,3)}}
        del ids,inputs;gc.collect();torch.cuda.empty_cache()
        return result


def worker_loop():
    import sys,traceback
    cache=InferenceCache()
    for line in sys.stdin:
        try:value={'ok':True,'result':cache.generate(json.loads(line))}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            value={'ok':False,'error':type(exc).__name__+':'+str(exc)[:600]}
        print('MEDAGENT_RESULT:'+json.dumps(value,ensure_ascii=False),flush=True)
