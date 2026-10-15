"""Single configuration file; relative paths resolve against that file's directory."""
import copy
import json
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_CONFIG=Path(__file__).resolve().parents[1]/'configs/service.json'

class Settings:
    def __init__(self,path):
        self.file=Path(path).expanduser().resolve()
        self.data=json.loads(self.file.read_text(encoding='utf-8'))
        if not isinstance(self.data,dict):raise ValueError('configuration_must_be_object')
        for key in ('http','paths','model_services','models','agent','consultation','limits','retrieval'):
            if not isinstance(self.data.get(key),dict):raise ValueError('missing_configuration:'+key)
        h=self.data['http']
        if not isinstance(h['host'],str) or not h['host'] or type(h['port']) is not int or not 1<=h['port']<=65535:raise ValueError('invalid_http_binding')
        for name in ('api_prefix','docs_path','openapi_path'):
            value=h[name]
            if not isinstance(value,str) or not value.startswith('/') or value=='/' or value.endswith('/'):
                raise ValueError('invalid_http_path:'+name)
        if len({h['api_prefix'],h['docs_path'],h['openapi_path']})!=3:raise ValueError('overlapping_http_paths')
        for key in ('workers','max_pending_runs','max_request_bytes'):
            if type(h[key]) is not int or h[key]<1:raise ValueError('invalid_http_limit:'+key)
        if h['max_pending_runs']<h['workers']:raise ValueError('queue_capacity_below_workers')
        for key,value in self.data['limits'].items():
            if type(value) is not int or value<1:raise ValueError('invalid_input_limit:'+key)
        for name,service in self.data['model_services'].items():
            url=service.get('base_url','');parts=urlsplit(url)
            if parts.scheme not in {'http','https'} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
                raise ValueError('invalid_model_service_url:'+name)
            if 'chat_path' in service and not service['chat_path'].startswith('/'):raise ValueError('invalid_chat_path:'+name)
        roles=self.model_roles()
        from .token_budget import DEFAULT_CONTEXT
        context=self.data.get('context',{})
        if not isinstance(context,dict) or set(context)-set(DEFAULT_CONTEXT):
            raise ValueError('invalid_context_configuration')
        for name,value in context.items():
            if type(value) is not int or value<1:
                raise ValueError('invalid_context_limit:'+name)
        context={**DEFAULT_CONTEXT,**context}
        if context['context_window_tokens']<=context['output_reserve_tokens']+context['safety_tokens']:
            raise ValueError('invalid_context_window')
        for role,cfg in roles.items():
            window=cfg.get('context_window_tokens',context['context_window_tokens'])
            output=cfg.get('max_tokens',context['output_reserve_tokens'])
            if type(window) is not int or type(output) is not int or output<1 or window<=output+context['safety_tokens']:
                raise ValueError('invalid_model_context_window:'+role)
            if type(cfg.get('token_counter_includes_images',False)) is not bool:
                raise ValueError('invalid_multimodal_token_counter_flag:'+role)
            path=cfg.get('token_count_path')
            if path is not None and (not isinstance(path,str) or not path.startswith('/') or path.startswith('//')):
                raise ValueError('invalid_token_count_path:'+role)
        mode=self.data['retrieval'].get('mode')
        if mode not in {'bm25','hybrid'}:raise ValueError('invalid_retrieval_mode')
        r=self.data['retrieval']
        for key in ('top_k','candidate_k','chunk_size','chunk_stride'):
            if type(r.get(key)) is not int or r[key]<1:raise ValueError('invalid_retrieval_limit:'+key)
        if r['candidate_k']<r['top_k'] or r['chunk_stride']>r['chunk_size'] or r['top_k']>self.data['limits']['search_limit']:raise ValueError('invalid_retrieval_window')

        if mode=='hybrid':
            if r.get('failure_policy') not in {'fail','bm25_with_notice'}:raise ValueError('invalid_retrieval_failure_policy')
            if r.get('index_policy') not in {'require_current','build_if_missing'}:raise ValueError('invalid_dense_index_policy')
            if type(r.get('max_index_bytes')) is not int or r['max_index_bytes']<1:raise ValueError('invalid_dense_index_size_limit')
            fusion=r['fusion']
            if type(fusion['rrf_k']) is not int or fusion['rrf_k']<1:raise ValueError('invalid_rrf_k')
            import math
            for name in ('bm25_weight','dense_weight'):
                if type(fusion[name]) not in (int,float) or not math.isfinite(fusion[name]) or fusion[name]<=0:raise ValueError('invalid_fusion_weight')
            for name in ('embedding','reranker'):
                model=r[name]
                if model['service'] not in self.data['model_services']:raise ValueError('unknown_retrieval_service')
                if not isinstance(model['model'],str) or not model['model'] or not isinstance(model['revision'],str) or not model['revision']:raise ValueError('retrieval_identity_required')
                if not isinstance(model['request_path'],str) or not model['request_path'].startswith('/'):raise ValueError('invalid_retrieval_request_path')
                for key in ('timeout','max_response_bytes'):
                    if type(model[key]) is not int or model[key]<1:raise ValueError('invalid_retrieval_request_limit')
                if type(model['retries']) is not int or not 0<=model['retries']<=3:raise ValueError('invalid_retrieval_retries')
                if type(model['retry_delay_seconds']) not in (int,float) or not math.isfinite(model['retry_delay_seconds']) or model['retry_delay_seconds']<0:raise ValueError('invalid_retry_delay')
            e=r['embedding'];rr=r['reranker']
            if type(e['batch_size']) is not int or e['batch_size']<1 or type(e['max_input_chars']) is not int or e['max_input_chars']<1:raise ValueError('invalid_embedding_batch')
            if not all(isinstance(e[key],str) for key in ('query_prefix','document_prefix')):raise ValueError('invalid_embedding_prefix')
            if e['dimensions'] is not None and (type(e['dimensions']) is not int or e['dimensions']<1):raise ValueError('invalid_embedding_dimensions')
            if type(rr['enabled']) is not bool or rr['protocol'] not in {'tei','cohere'}:raise ValueError('unsupported_reranker_protocol')
            for key in ('max_query_chars','max_document_chars'):
                if type(rr[key]) is not int or rr[key]<1:raise ValueError('invalid_reranker_input_limit')

    def path(self,name):
        value=Path(self.data['paths'][name]).expanduser()
        return (self.file.parent/value).resolve() if not value.is_absolute() else value.resolve()

    def model_roles(self,mapping=None):
        roles={}
        for role,alias in (mapping or {k:k for k in self.data['models']}).items():
            cfg=copy.deepcopy(self.data['models'][alias])
            service=self.data['model_services'][cfg.pop('service')]
            cfg.update(endpoint=service['base_url'],chat_path=service['chat_path'],api_key_env=service.get('api_key_env',''),token_count_path=cfg.get('token_count_path',service.get('token_count_path')),
                token_counter_includes_images=cfg.get('token_counter_includes_images',service.get('token_counter_includes_images',False)))
            roles[role]=cfg
        return roles

    def consultation(self):
        config=copy.deepcopy(self.data['consultation'])
        config['models']=self.model_roles(config['models'])
        return config

    def public(self):
        from .common import read_json
        from .token_budget import context_options
        return {'service':self.data['service_name'],'version':self.data['version'],
            'api_prefix':self.data['http']['api_prefix'],'specialties':read_json(self.path('specialties_file'))['agents'],
            'retrieval_mode':self.data['retrieval']['mode'],
            'models':{role:cfg['model'] for role,cfg in self.data['models'].items()},
            'limits':self.data['limits'],'context':context_options(self),
            'training_managed_by_service':False}

@lru_cache(maxsize=1)
def get_settings():return Settings(os.environ.get('MEDAGENT_CONFIG',str(DEFAULT_CONFIG)))

def configure(path=None):
    if path is not None:os.environ['MEDAGENT_CONFIG']=str(Path(path).expanduser().resolve())
    get_settings.cache_clear()
    return get_settings()

def configured_path(name):return get_settings().path(name)
