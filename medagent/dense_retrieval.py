"""Version-bound vector index and BM25/dense fusion with explicit reranking."""
import fcntl
import json
import math
import threading
import time
from .common import digest,read_json,write_json
from .rag_clients import EmbeddingClient,RerankerClient,RetrievalUnavailable


def fingerprint(value):return digest(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')))

class DenseIndex:
    def __init__(self,settings,lexical):
        self.settings,self.lexical=settings,lexical
        self.client=EmbeddingClient(settings);self.path=settings.path('dense_index')
        self.corpus=[{k:v for k,v in c.items() if k!='tf'} for c in lexical.chunks]
        self.corpus_sha=fingerprint(self.corpus);self.encoder_sha=fingerprint(self.client.identity())
        self.matrix=None;self.manifest=None;self.guard=threading.RLock()

    def _read(self):
        if self.path.stat().st_size>self.settings.data['retrieval']['max_index_bytes']:raise RetrievalUnavailable('dense_index_too_large')
        index=read_json(self.path);manifest=index.get('manifest',{})
        if manifest.get('schema_version')!=1:raise RetrievalUnavailable('unsupported_dense_index_schema')
        if manifest.get('corpus_sha256')!=self.corpus_sha:raise RetrievalUnavailable('dense_index_stale_corpus_rebuild_required')
        if manifest.get('embedding_config_sha256')!=self.encoder_sha:raise RetrievalUnavailable('dense_index_stale_embedding_config_rebuild_required')
        if manifest.get('chunk_ids')!=[c['id'] for c in self.corpus]:raise RetrievalUnavailable('dense_index_chunk_order_mismatch')
        matrix=index.get('vectors');dimension=manifest.get('dimension')
        if not isinstance(matrix,list) or len(matrix)!=len(self.corpus) or type(dimension) is not int or dimension<1:
            raise RetrievalUnavailable('invalid_dense_index_shape')
        if fingerprint(matrix)!=manifest.get('vectors_sha256'):raise RetrievalUnavailable('dense_index_checksum_mismatch')
        for row in matrix:
            if not isinstance(row,list) or len(row)!=dimension or any(type(x) not in (int,float) or not math.isfinite(x) for x in row):
                raise RetrievalUnavailable('invalid_dense_index_values')
            if abs(sum(x*x for x in row)-1)>0.001:raise RetrievalUnavailable('dense_index_not_normalized')
        self.matrix,self.manifest=matrix,manifest

    def build(self,force=False):
        # Cross-thread and cross-process locking prevent overlapping writes/model builds.
        with self.guard:
            self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            lock_path=self.path.with_suffix('.lock')
            with lock_path.open('a+') as lock:
                lock_path.chmod(0o600);fcntl.flock(lock,fcntl.LOCK_EX)
                if self.path.exists() and not force:
                    self._read();return dict(self.manifest,reused=True)
                if not self.corpus:raise RetrievalUnavailable('empty_knowledge_corpus')
                matrix=self.client.embed([c['text'] for c in self.corpus])
                if len(matrix)!=len(self.corpus):raise RetrievalUnavailable('dense_index_build_count_mismatch')
                manifest={'schema_version':1,'corpus_sha256':self.corpus_sha,'embedding_config_sha256':self.encoder_sha,
                    'embedding_identity':self.client.identity(),'chunk_ids':[c['id'] for c in self.corpus],
                    'dimension':len(matrix[0]),'chunks':len(matrix),'vectors_sha256':fingerprint(matrix),'built_at':time.time(),
                    'identity_verification':'configuration_and_declared_revision_not_independent_weight_audit'}
                index={'manifest':manifest,'vectors':matrix}
                if len(json.dumps(index,ensure_ascii=False).encode())>self.settings.data['retrieval']['max_index_bytes']:
                    raise RetrievalUnavailable('dense_index_build_exceeds_size_limit')
                write_json(self.path,index);self.matrix,self.manifest=matrix,manifest
                return dict(manifest,reused=False)

    def ensure_loaded(self):
        with self.guard:
            if self.matrix is not None:return
            if self.path.exists():self._read()
            elif self.settings.data['retrieval']['index_policy']=='build_if_missing':self.build()
            else:raise RetrievalUnavailable('dense_index_missing_explicit_build_required')

    def search(self,query,task=None,limit=20):
        ids=[i for i,c in enumerate(self.corpus) if not task or c['source'].startswith(task+'/')]
        if not ids:return []
        self.ensure_loaded()
        vector=self.client.embed([query],query=True)[0]
        if len(vector)!=self.manifest['dimension']:raise RetrievalUnavailable('query_embedding_dimension_mismatch')
        hits=[]
        for i in ids:
            score=sum(a*b for a,b in zip(self.matrix[i],vector))
            hits.append(dict(self.corpus[i],dense_score=score))
        return sorted(hits,key=lambda c:(-c['dense_score'],c['id']))[:limit]

    def describe(self):
        return {'index_file_present':self.path.exists(),'loaded_and_validated':self.matrix is not None,
            'corpus_sha256':self.corpus_sha,'embedding_config_sha256':self.encoder_sha,
            'dimension':self.manifest['dimension'] if self.manifest else None,'chunks':len(self.corpus)}

class HybridRetrievalService:
    def __init__(self,settings,lexical):
        self.settings,self.index=settings,lexical
        self.dense=DenseIndex(settings,lexical);self.reranker=RerankerClient(settings)

    def describe(self):
        cfg=self.settings.data['retrieval']
        return {'mode':'hybrid','reranker_enabled':cfg['reranker']['enabled'],
            'failure_policy':cfg['failure_policy'],'inference_executed_at_startup':False,**self.dense.describe()}

    def build(self,force=False):return self.dense.build(force)

    def search_with_trace(self,query,task=None,limit=None):
        cfg=self.settings.data['retrieval'];limit=limit or cfg['top_k'];candidate_k=max(limit,cfg['candidate_k'])
        trace={'mode':'hybrid','task':task,'limit':limit,'candidate_k':candidate_k,
            'corpus_sha256':self.dense.corpus_sha,'stages':[],'degraded':False}
        lexical=self.index.search(query,task,candidate_k)
        trace['stages'].append({'name':'bm25','status':'ok','candidates':len(lexical)})
        try:
            dense=self.dense.search(query,task,candidate_k)
            trace['stages'].append({'name':'dense','status':'ok','candidates':len(dense),
                'embedding_model':cfg['embedding']['model'],'declared_revision':cfg['embedding']['revision']})
            fused={};fusion=cfg['fusion']
            for name,ranking,weight in [('bm25',lexical,fusion['bm25_weight']),('dense',dense,fusion['dense_weight'])]:
                for rank,hit in enumerate(ranking,1):
                    key=hit['id'];record=fused.setdefault(key,{**hit,'score':0.,'retrieval_scores':{},'retrieval_ranks':{}})
                    record['retrieval_scores'][name]=hit['score'] if name=='bm25' else hit['dense_score']
                    record['retrieval_ranks'][name]=rank
                    record['score']+=weight/(fusion['rrf_k']+rank)
            candidates=sorted(fused.values(),key=lambda c:(-c['score'],c['id']))[:candidate_k]
            trace['stages'].append({'name':'weighted_rrf','status':'ok','candidates':len(candidates)})
            if cfg['reranker']['enabled']:
                candidates=self.reranker.rerank(query,candidates)
                trace['stages'].append({'name':'reranker','status':'ok','candidates':len(candidates),
                    'model':cfg['reranker']['model'],'declared_revision':cfg['reranker']['revision']})
            else:trace['stages'].append({'name':'reranker','status':'disabled'})
            trace['reranked']=cfg['reranker']['enabled'] and bool(candidates)
            hits=candidates[:limit]
        except Exception as exc:
            failed='reranker' if trace['stages'][-1]['name']=='weighted_rrf' else 'dense_or_fusion'
            trace['stages'].append({'name':failed,'status':'failed','error':str(exc)[:240]})
            if cfg['failure_policy']=='fail':raise RetrievalUnavailable('hybrid_retrieval_failed:'+str(exc)[:240]) from None
            trace.update(mode='bm25_fallback',degraded=True,reranked=False,fallback_reason=str(exc)[:240])
            hits=lexical[:limit]
        return {'results':[dict(hit,retrieval=trace['mode'],retrieval_audit=trace) for hit in hits], 'retrieval':trace}

    def search(self,query,task=None,limit=None):return self.search_with_trace(query,task,limit)['results']
