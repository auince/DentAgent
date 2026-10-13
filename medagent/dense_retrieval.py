"""Version-bound dense retrieval with reciprocal-rank fusion and optional reranker."""
import json
import hashlib
from pathlib import Path
from .common import digest,read_json,write_json
from .retrieval import KnowledgeIndex

def weight_version(path):
    result={}
    for p in sorted(Path(path).glob('*.safetensors')):
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        result[p.name]=h.hexdigest()
    if not result:raise ValueError('missing_encoder_weights')
    return result

class DenseKnowledgeIndex(KnowledgeIndex):
    def __init__(self,directory,index_path,encoder_path,device='cpu'):
        super().__init__(directory)
        import numpy as np
        from transformers import AutoTokenizer,AutoModel
        self.manifest=read_json(Path(index_path)/'manifest.json')
        fingerprint=digest(json.dumps([{k:v for k,v in c.items() if k!='tf'} for c in self.chunks],sort_keys=True,ensure_ascii=False))
        if self.manifest['corpus_sha256']!=fingerprint:raise ValueError('dense_index_stale_corpus')
        if self.manifest['encoder_path']!=str(Path(encoder_path).resolve()):raise ValueError('dense_index_encoder_mismatch')
        if self.manifest['encoder_config_sha256']!=digest(Path(encoder_path,'config.json').read_bytes()):raise ValueError('dense_index_encoder_config_changed')
        if self.manifest['encoder_weights']!=weight_version(encoder_path):raise ValueError('dense_index_encoder_weights_changed')
        self.matrix=np.load(Path(index_path)/'vectors.npy',allow_pickle=False)
        self.tokenizer=AutoTokenizer.from_pretrained(encoder_path,padding_side='left')
        self.encoder=AutoModel.from_pretrained(encoder_path).to(device).eval();self.device=device
    def embed(self,texts):
        import torch
        import torch.nn.functional as F
        batch=self.tokenizer(texts,padding=True,truncation=True,max_length=1024,return_tensors='pt').to(self.device)
        with torch.inference_mode():return F.normalize(self.encoder(**batch).last_hidden_state[:,-1].float(),dim=-1).cpu().numpy()
    def search(self,query,task=None,limit=4):
        import numpy as np
        score=self.matrix@self.embed(['Instruct: Retrieve oral medical evidence relevant to the patient query.\nQuery: '+query])[0]
        ids=[i for i in np.argsort(-score) if not task or self.chunks[i]['source'].startswith(task+'/')][:20]
        fused={};docs={c['id']:{k:v for k,v in c.items() if k!='tf'} for c in self.chunks}
        for ranking in [[self.chunks[i]['id'] for i in ids],[c['id'] for c in super().search(query,task,20)]]:
            for rank,key in enumerate(ranking):fused[key]=fused.get(key,0)+1/(60+rank+1)
        return [{**docs[k],'score':v,'retrieval':'dense_bm25_rrf'} for k,v in sorted(fused.items(),key=lambda x:-x[1])[:limit]]

def build_index(directory,index_path,encoder_path,device='cpu'):
    import numpy as np
    from transformers import AutoTokenizer,AutoModel
    index=object.__new__(DenseKnowledgeIndex);KnowledgeIndex.__init__(index,directory)
    index.tokenizer=AutoTokenizer.from_pretrained(encoder_path,padding_side='left');index.encoder=AutoModel.from_pretrained(encoder_path).to(device).eval();index.device=device
    matrix=np.concatenate([index.embed([c['text'] for c in index.chunks[i:i+4]]) for i in range(0,len(index.chunks),4)])
    out=Path(index_path);out.mkdir(parents=True,exist_ok=True)
    np.save(out/'vectors.npy',matrix,allow_pickle=False)
    chunks=[{k:v for k,v in c.items() if k!='tf'} for c in index.chunks]
    write_json(out/'manifest.json',{'encoder_path':str(Path(encoder_path).resolve()),'encoder_config_sha256':digest(Path(encoder_path,'config.json').read_bytes()),
        'encoder_weights':weight_version(encoder_path),'corpus_sha256':digest(json.dumps(chunks,sort_keys=True,ensure_ascii=False)),'chunks':len(chunks),'dimension':matrix.shape[1]})

class RerankedIndex:
    def __init__(self,index,model_path,device='cpu'):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        self.index=index;self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,padding_side='left')
        self.model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16 if device!='cpu' else torch.float32).to(device).eval()
        self.yes=self.tokenizer.encode('yes',add_special_tokens=False);self.no=self.tokenizer.encode('no',add_special_tokens=False)
        if len(self.yes)!=1 or len(self.no)!=1:raise ValueError('reranker_answer_tokens')
    def search(self,query,task=None,limit=4):
        import torch
        from .retrieval_training import ranking_prompt
        candidates=self.index.search(query,task,max(12,limit))
        if not candidates:return []
        batch=self.tokenizer([ranking_prompt(query,c['text']) for c in candidates],padding=True,truncation=True,max_length=1024,return_tensors='pt').to(self.device)
        with torch.inference_mode():
            logits=self.model(**batch).logits[:,-1].float()
            scores=(logits[:,self.yes[0]]-logits[:,self.no[0]]).cpu().tolist()
        return sorted([{**c,'rerank_score':s} for c,s in zip(candidates,scores)],key=lambda c:-c['rerank_score'])[:limit]
