"""Explicit, host-pinned external teacher for synthetic data only."""
import contextvars
import fcntl
import json
import os
import random
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit
from .common import ROOT,read_json,write_json,digest,parse_json
from .provider import ModelUnavailable

META=contextvars.ContextVar('teacher_request_meta',default=None)

class LocalTeacher:
    """Local Qwen teacher with the same context and source contract as API generation."""
    def __init__(self):
        import itertools
        import threading
        self.cfg=read_json(ROOT/'configs/generation_teacher.json')
        if self.cfg['backend']!='local':raise ValueError('local_teacher_not_configured')
        self.context=(ROOT/self.cfg['context_file']).read_text()
        self.registry=[a for a in read_json(ROOT/'configs/specialties.json')['agents'] if a.get('enabled',True)]
        endpoints=os.environ.get('ORAL_TEACHER_ENDPOINTS','http://deployment.invalid').split(',')
        if any(urlsplit(e).hostname not in {'127.0.0.1','localhost'} for e in endpoints):raise ValueError('nonlocal_teacher')
        self.endpoints=itertools.cycle(endpoints);self.lock=threading.Lock()
        self.directory=ROOT/'runtime/small_model/generation_local'/self.cfg['campaign']
        self.directory.mkdir(parents=True,exist_ok=True)

    def complete(self,prompt,payload,response_format=None,max_tokens=None):
        from .provider import LocalModelClient,CALL_METADATA
        with self.lock:endpoint=next(self.endpoints)
        system=self.context+'\n\n'+prompt
        text=json.dumps({'registry':self.registry,'task':payload},ensure_ascii=False)
        client=LocalModelClient({'teacher':{'endpoint':endpoint,'model':self.cfg['served_model'],
            'temperature':.7,'max_tokens':min(max_tokens or self.cfg['max_completion_tokens'],self.cfg['max_completion_tokens']),
            'timeout':self.cfg['timeout_seconds'],'retries':self.cfg['max_retries']}})
        rid=uuid.uuid4().hex;started=time.time()
        meta={'request_id':rid,'requested_model':self.cfg['model'],'model_path':self.cfg['model_path'],
            'campaign':self.cfg['campaign'],'endpoint':endpoint,'started_at':started,'pid':os.getpid(),
            'prompt_sha256':digest(system),'request_sha256':digest(text)}
        META.set(None)
        write_json(self.directory/(rid+'.json'),{**meta,'status':'in_flight'})
        try:
            result=client.complete('teacher',system,text,response_format=response_format or {'type':'json_object'})
            meta.update(status='complete',elapsed_seconds=time.time()-started,response_sha256=digest(result),**(CALL_METADATA.get() or {}))
            write_json(self.directory/(rid+'.json'),meta);META.set(meta)
            return parse_json(result)
        except ModelUnavailable as exc:
            write_json(self.directory/(rid+'.json'),{**meta,'status':'failed','error':str(exc)})
            if any(x in str(exc) for x in ('endpoint unavailable','HTTP 429','HTTP 500','HTTP 502','HTTP 503','HTTP 504')):
                raise TeacherTransientError(str(exc)) from None
            raise

def make_teacher():
    cfg=read_json(ROOT/'configs/generation_teacher.json')
    return LocalTeacher() if cfg['backend']=='local' else RemoteTeacher()

class TeacherTransientError(ModelUnavailable):
    pass

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

class RemoteTeacher:
    def __init__(self):
        self.cfg=read_json(ROOT/'configs/generation_teacher.json')
        if self.cfg['backend']!='remote_api':raise ValueError('external_teacher_not_configured')
        self.secret=read_json(self.cfg['credential_file'])
        self.endpoint=self.cfg['base_url'].rstrip('/')
        if self.secret['base_url'].rstrip('/')!=self.endpoint or self.secret['model']!=self.cfg['model']:raise ValueError('teacher_credentials_binding_mismatch')
        url=urlsplit(self.endpoint)
        if url.scheme!='https' or url.hostname!='api.luomilo.com' or url.username or url.password:raise ValueError('unexpected_teacher_host')
        self.context=(ROOT/self.cfg['context_file']).read_text()
        self.registry=read_json(ROOT/'configs/specialties.json')['agents']
        self.directory=ROOT/'runtime/small_model/generation_api'/self.cfg['campaign']
        self.directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.slots=ROOT/'runtime/small_model/api_slots';self.slots.mkdir(parents=True,exist_ok=True,mode=0o700)
        proxy=self.cfg.get('https_proxy')
        if proxy and (urlsplit(proxy).hostname not in {'127.0.0.1','localhost'}):raise ValueError('unexpected_proxy_host')
        self.opener=urllib.request.build_opener(NoRedirect,urllib.request.ProxyHandler({'https':proxy} if proxy else {}))

    @contextmanager
    def slot(self):
        while True:
            for i in random.sample(range(self.cfg['concurrency']),self.cfg['concurrency']):
                handle=(self.slots/str(i)).open('a')
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:handle.close();continue
                try:yield i
                finally:handle.close()
                return
            time.sleep(.1)

    def complete(self,prompt,payload,response_format=None):
        body={'model':self.cfg['model'],'store':False,'reasoning_effort':self.cfg['reasoning_effort'],
            'max_completion_tokens':self.cfg['max_completion_tokens'],
            'messages':[{'role':'system','content':self.context+'\n\n'+prompt},
                {'role':'user','content':json.dumps({'registry':self.registry,'task':payload},ensure_ascii=False)}]}
        body['response_format']=response_format or {'type':'json_object'}
        encoded=json.dumps(body,ensure_ascii=False).encode()
        rid=uuid.uuid4().hex;started=time.time()
        info={'request_id':rid,'requested_model':self.cfg['model'],'provider':self.endpoint,'campaign':self.cfg['campaign'],
            'prompt_sha256':digest(body['messages'][0]['content']),'request_sha256':digest(encoded),'started_at':started,'store_requested':False}
        for attempt in range(self.cfg['max_retries']+1):
            try:
                with self.slot() as slot:
                    write_json(self.directory/(rid+'.json'),{**info,'status':'in_flight','slot':slot,'attempt':attempt,'pid':os.getpid()})
                    request=urllib.request.Request(self.endpoint+'/chat/completions',encoded,
                        {'Content-Type':'application/json','Authorization':'Bearer '+self.secret['api_key']})
                    with self.opener.open(request,timeout=self.cfg['timeout_seconds']) as response:data=json.load(response)
                choice=data['choices'][0];text=choice['message']['content']
                meta={**info,'status':'complete','returned_model':data.get('model'),'usage':data.get('usage'),
                    'provider_request_id':data.get('id'),'finish_reason':choice.get('finish_reason'),
                    'attempt':attempt,'elapsed_seconds':time.time()-started,'response_sha256':digest(text or '')}
                write_json(self.directory/(rid+'.json'),meta);META.set(meta)
                if choice.get('finish_reason')=='length':raise ModelUnavailable('teacher_api: output_truncated')
                if not isinstance(text,str) or not text.strip():raise ModelUnavailable('teacher_api: empty_output')
                return parse_json(text)
            except urllib.error.HTTPError as exc:
                # Do not persist provider response bodies or auth headers.
                error='teacher_api: HTTP '+str(exc.code)
                detail=exc.read().decode(errors='replace').replace(self.secret['api_key'],'[REDACTED]')[:800]
                retry_after=exc.headers.get('Retry-After','')
                if exc.code not in (429,500,502,503,504) or attempt>=self.cfg['max_retries']:
                    write_json(self.directory/(rid+'.json'),{**info,'status':'failed','error':error,'detail':detail,'attempt':attempt})
                    if exc.code in (429,500,502,503,504):raise TeacherTransientError(error) from None
                    raise ModelUnavailable(error) from None
                delay=float(retry_after) if retry_after.isdigit() else min(120,10*2**attempt+random.uniform(0,5))
            except (urllib.error.URLError,TimeoutError,ConnectionError) as exc:
                if attempt>=self.cfg['max_retries']:
                    write_json(self.directory/(rid+'.json'),{**info,'status':'failed','error':'endpoint_unavailable','attempt':attempt})
                    raise TeacherTransientError('teacher_api: endpoint_unavailable') from None
                delay=min(120,10*2**attempt+random.uniform(0,5))
            time.sleep(delay)
