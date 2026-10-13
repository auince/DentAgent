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
            if not service.get('chat_path','').startswith('/'):raise ValueError('invalid_chat_path:'+name)
        self.model_roles()
        mode=self.data['retrieval'].get('mode')
        if mode not in {'bm25','hybrid'}:raise ValueError('invalid_retrieval_mode')
        r=self.data['retrieval']
        for key in ('top_k','candidate_k','chunk_size','chunk_stride'):
            if type(r.get(key)) is not int or r[key]<1:raise ValueError('invalid_retrieval_limit:'+key)
        if r['candidate_k']<r['top_k'] or r['chunk_stride']>r['chunk_size']:raise ValueError('invalid_retrieval_window')

    def path(self,name):
        value=Path(self.data['paths'][name]).expanduser()
        return (self.file.parent/value).resolve() if not value.is_absolute() else value.resolve()

    def model_roles(self,mapping=None):
        roles={}
        for role,alias in (mapping or {k:k for k in self.data['models']}).items():
            cfg=copy.deepcopy(self.data['models'][alias])
            service=self.data['model_services'][cfg.pop('service')]
            cfg.update(endpoint=service['base_url'],chat_path=service['chat_path'],api_key_env=service.get('api_key_env',''))
            roles[role]=cfg
        return roles

    def consultation(self):
        config=copy.deepcopy(self.data['consultation'])
        config['models']=self.model_roles(config['models'])
        return config

    def public(self):
        from .common import read_json
        return {'service':self.data['service_name'],'version':self.data['version'],
            'api_prefix':self.data['http']['api_prefix'],'specialties':read_json(self.path('specialties_file'))['agents'],
            'retrieval_mode':self.data['retrieval']['mode'],
            'models':{role:cfg['model'] for role,cfg in self.data['models'].items()},
            'limits':self.data['limits'],'training_managed_by_service':False}

@lru_cache(maxsize=1)
def get_settings():return Settings(os.environ.get('MEDAGENT_CONFIG',str(DEFAULT_CONFIG)))

def configure(path=None):
    if path is not None:os.environ['MEDAGENT_CONFIG']=str(Path(path).expanduser().resolve())
    get_settings.cache_clear()
    return get_settings()

def configured_path(name):return get_settings().path(name)
