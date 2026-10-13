"""Explicit model protocols for embeddings and reranking; no local GPU loading."""
import json
import math
import os
import time
import urllib.error
import urllib.request

class RetrievalUnavailable(RuntimeError):pass

def post_json(settings,config,body):
    service=settings.data['model_services'][config['service']]
    url=service['base_url'].rstrip('/')+config['request_path']
    headers={'Content-Type':'application/json'}
    key=os.environ.get(service.get('api_key_env',''))
    if key:headers['Authorization']='Bearer '+key
    retries=config.get('retries',0)
    for attempt in range(retries+1):
        req=urllib.request.Request(url,json.dumps(body,ensure_ascii=False).encode('utf-8'),headers)
        try:
            with urllib.request.urlopen(req,timeout=config['timeout']) as response:
                raw=response.read(config['max_response_bytes']+1)
            if len(raw)>config['max_response_bytes']:raise RetrievalUnavailable('retrieval_response_too_large')
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            if exc.code not in {429,500,502,503,504} or attempt==retries:
                raise RetrievalUnavailable('retrieval_model_HTTP_'+str(exc.code)) from None
        except (urllib.error.URLError,TimeoutError,ConnectionError):
            if attempt==retries:raise RetrievalUnavailable('retrieval_model_endpoint_unavailable') from None
        except (ValueError,UnicodeDecodeError):raise RetrievalUnavailable('invalid_retrieval_model_JSON') from None
        time.sleep(config.get('retry_delay_seconds',1))
    raise RetrievalUnavailable('retrieval_request_failed')


def normalize_vector(value):
    if not isinstance(value,list) or not value or any(type(x) not in (int,float) or not math.isfinite(x) for x in value):
        raise RetrievalUnavailable('invalid_embedding_vector')
    norm=math.sqrt(sum(x*x for x in value))
    if not math.isfinite(norm) or norm==0:raise RetrievalUnavailable('zero_or_overflow_embedding_vector')
    return [x/norm for x in value]

class EmbeddingClient:
    def __init__(self,settings):self.settings,self.config=settings,settings.data['retrieval']['embedding']
    def identity(self):
        cfg=self.config;svc=self.settings.data['model_services'][cfg['service']]
        return {'model':cfg['model'],'declared_revision':cfg['revision'],
            'endpoint':svc['base_url'],'request_path':cfg['request_path'],
            'query_prefix':cfg['query_prefix'],'document_prefix':cfg['document_prefix'],
            'dimensions':cfg.get('dimensions'),'normalization':'l2','protocol':'openai_embeddings'}
    def embed(self,texts,query=False):
        if not texts:return []
        cfg=self.config;prefix=cfg['query_prefix'] if query else cfg['document_prefix']
        inputs=[prefix+t for t in texts]
        if any(not isinstance(t,str) or len(t)>cfg['max_input_chars'] for t in inputs):raise RetrievalUnavailable('embedding_input_too_large')
        vectors=[];dimension=None
        for start in range(0,len(inputs),cfg['batch_size']):
            batch=inputs[start:start+cfg['batch_size']]
            payload={'model':cfg['model'],'input':batch,'encoding_format':'float'}
            if cfg.get('dimensions') is not None:payload['dimensions']=cfg['dimensions']
            response=post_json(self.settings,cfg,payload)
            data=response.get('data') if isinstance(response,dict) else None
            if not isinstance(data,list) or len(data)!=len(batch):raise RetrievalUnavailable('embedding_count_mismatch')
            indexed={}
            for item in data:
                if not isinstance(item,dict) or type(item.get('index')) is not int or not 0<=item['index']<len(batch) or item['index'] in indexed:
                    raise RetrievalUnavailable('invalid_embedding_index')
                row=normalize_vector(item.get('embedding'))
                if dimension is None:dimension=len(row)
                if len(row)!=dimension or (cfg.get('dimensions') is not None and len(row)!=cfg['dimensions']):raise RetrievalUnavailable('embedding_dimension_mismatch')
                indexed[item['index']]=row
            vectors.extend(indexed[i] for i in range(len(batch)))
        return vectors

class RerankerClient:
    def __init__(self,settings):self.settings,self.config=settings,settings.data['retrieval']['reranker']
    def rerank(self,query,candidates):
        if not candidates:return []
        cfg=self.config
        if len(query)>cfg['max_query_chars'] or any(len(c['text'])>cfg['max_document_chars'] for c in candidates):
            raise RetrievalUnavailable('reranker_input_too_large')
        documents=[c['text'] for c in candidates]
        if cfg['protocol']=='tei':payload={'query':query,'texts':documents,'raw_scores':True}
        else:payload={'model':cfg['model'],'query':query,'documents':documents,'top_n':len(documents)}
        response=post_json(self.settings,cfg,payload)
        rows=response if cfg['protocol']=='tei' else response.get('results') if isinstance(response,dict) else None
        if not isinstance(rows,list) or len(rows)!=len(candidates):raise RetrievalUnavailable('reranker_count_mismatch')
        indexed={}
        for row in rows:
            if not isinstance(row,dict) or type(row.get('index')) is not int or not 0<=row['index']<len(candidates) or row['index'] in indexed:
                raise RetrievalUnavailable('invalid_reranker_index')
            score=row.get('score') if cfg['protocol']=='tei' else row.get('relevance_score')
            if type(score) not in (int,float) or not math.isfinite(score):raise RetrievalUnavailable('invalid_reranker_score')
            indexed[row['index']]=float(score)
        results=[dict(candidate,rerank_score=indexed[i]) for i,candidate in enumerate(candidates)]
        return sorted(results,key=lambda c:(-c['rerank_score'],-c['score'],c['id']))
