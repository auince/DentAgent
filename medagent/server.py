"""Configurable backend API; startup never loads or starts model/training workers."""
import hmac
import os
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI,APIRouter,Depends,Header,HTTPException,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPBearer,HTTPAuthorizationCredentials
from .settings import configure,get_settings


class BodyLimitMiddleware:
    """Bound streamed bodies as well as Content-Length, without private request APIs."""
    def __init__(self,app,limit):self.app,self.limit=app,limit
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        async def secure_send(message):
            if message['type']=='http.response.start':
                message['headers']=list(message.get('headers',[]))+[(b'cache-control',b'no-store'),(b'x-content-type-options',b'nosniff')]
            await send(message)
        raw=dict(scope.get('headers',[])).get(b'content-length')
        try:
            if raw is not None and not 0<=int(raw)<=self.limit:
                return await JSONResponse(status_code=413,content={'error':'payload_too_large'})(scope,receive,secure_send)
        except ValueError:
            return await JSONResponse(status_code=400,content={'error':'invalid_content_length'})(scope,receive,secure_send)
        if scope['method'] not in {'POST','PUT','PATCH'}:return await self.app(scope,receive,secure_send)
        chunks=[];size=0
        while True:
            message=await receive()
            if message['type']=='http.disconnect':return
            part=message.get('body',b'');size+=len(part)
            if size>self.limit:
                return await JSONResponse(status_code=413,content={'error':'payload_too_large'})(scope,receive,secure_send)
            chunks.append(part)
            if not message.get('more_body',False):break
        pending=True
        async def buffered_receive():
            nonlocal pending
            if pending:
                pending=False
                return {'type':'http.request','body':b''.join(chunks),'more_body':False}
            return await receive()
        await self.app(scope,buffered_receive,secure_send)


def create_app(config_path=None):
    settings=configure(config_path);h=settings.data['http']
    # Import after configuration selection: the specialty registry uses this file.
    from .api_models import AssessmentRequest,SessionRequest,MessageRequest,MemoryRequest,SearchRequest,IndexBuildRequest,RecordReviewRequest
    from .sessions import SessionStore,SessionBusy
    from .consultation import ConsultationService
    from .pipeline import Pipeline
    from .jobs import JobManager,QueueFull,IdempotencyConflict
    from .retrieval import create_index
    from .consult_contracts import REGISTRY
    create_index.cache_clear()
    key=os.environ.get(h['api_key_env'],'')
    loopback=h['host'] in {'127.0.0.1','localhost','::1'}
    if not key and (not loopback or not h['allow_unauthenticated_loopback']):
        raise ValueError('API_token_required_for_this_binding:'+h['api_key_env'])
    security=HTTPBearer(auto_error=False)
    def authorize(credentials:HTTPAuthorizationCredentials|None=Depends(security)):
        if key and (credentials is None or not hmac.compare_digest(credentials.credentials,key)):
            raise HTTPException(status_code=401,detail='invalid_API_token',headers={'WWW-Authenticate':'Bearer'})
    @asynccontextmanager
    async def lifespan(app):
        store=SessionStore();app.state.store=store
        app.state.jobs=JobManager(settings,store)
        try:
            app.state.consultation=ConsultationService(store=store)
            app.state.index=create_index()
            yield
        finally:app.state.jobs.close()
    app=FastAPI(title=settings.data['service_name'],version=str(settings.data['version']),
        description='病历评估、来源受约束的辅助预诊与会话记忆。模型接口需单独部署；服务不启动训练。',
        docs_url=h['docs_path'],openapi_url=h['openapi_path'],redoc_url=None,lifespan=lifespan,
        servers=[{'url':h['public_base_url']}] if h.get('public_base_url') else None)
    app.state.settings=settings
    if h['cors_origins']:
        app.add_middleware(CORSMiddleware,allow_origins=h['cors_origins'],allow_credentials=False,
            allow_methods=['GET','POST','DELETE'],allow_headers=['Authorization','Content-Type','Idempotency-Key'])
    app.add_middleware(BodyLimitMiddleware,limit=h['max_request_bytes'])
    @app.exception_handler(RequestValidationError)
    async def schema_error(request,exc):
        errors=[{'loc':e['loc'],'type':e['type'],'message':e['msg']} for e in exc.errors()]
        return JSONResponse(status_code=422,content={'error':'invalid_request','details':errors})
    @app.exception_handler(KeyError)
    async def missing(request,exc):return JSONResponse(status_code=404,content={'error':'resource_not_found'})
    @app.exception_handler(SessionBusy)
    async def busy(request,exc):return JSONResponse(status_code=409,content={'error':str(exc)})
    @app.exception_handler(IdempotencyConflict)
    async def duplicate(request,exc):return JSONResponse(status_code=409,content={'error':str(exc)})
    @app.exception_handler(QueueFull)
    async def full(request,exc):return JSONResponse(status_code=429,content={'error':str(exc)},headers={'Retry-After':'5'})
    @app.exception_handler(ValueError)
    async def invalid(request,exc):return JSONResponse(status_code=400,content={'error':str(exc)[:240]})
    @app.exception_handler(Exception)
    async def unavailable(request,exc):return JSONResponse(status_code=503,content={'error':'processing_service_unavailable'})
    def check_id(value):
        if not re.fullmatch('[a-f0-9]{32}',value):raise HTTPException(status_code=400,detail='invalid_resource_id')
        return value
    def check_key(value):
        if value is not None and (not value or len(value)>128 or any(ord(x)<33 or ord(x)>126 for x in value)):
            raise HTTPException(status_code=400,detail='invalid_idempotency_key')
        return value
    router=APIRouter(prefix=h['api_prefix'],dependencies=[Depends(authorize)])
    @router.get('/health',tags=['service'])
    def health(request:Request):return {'status':'ready','version':settings.data['version'],
        'model_readiness':'not_probed','retrieval':request.app.state.index.describe(),'training_started':False}
    @router.get('/capabilities',tags=['service'])
    def capabilities():return settings.public()
    @router.post('/assessments',status_code=202,tags=['assessment'])
    def assess(body:AssessmentRequest,request:Request,idempotency_key:str|None=Header(default=None)):
        payload=body.model_dump(exclude_none=True)
        return request.app.state.jobs.submit('assessment',payload,lambda rid,old:Pipeline().run(payload,rid),key=check_key(idempotency_key))
    @router.post('/records/review',status_code=202,tags=['record review'])
    def review_record(body:RecordReviewRequest,request:Request,idempotency_key:str|None=Header(default=None)):
        from .record_review import RecordReviewService
        payload=body.model_dump()
        return request.app.state.jobs.submit('record_review',payload,
            lambda rid,old:RecordReviewService().run(payload,rid),key=check_key(idempotency_key))
    @router.get('/runs/{run_id}',tags=['runs'])
    def run_status(run_id:str,request:Request):return request.app.state.jobs.get(check_id(run_id))
    @router.post('/sessions',status_code=201,tags=['consultation'])
    def session(body:SessionRequest,request:Request):return request.app.state.store.create(body.patient_id)
    @router.get('/sessions/{conversation_id}',tags=['consultation'])
    def get_session(conversation_id:str,request:Request):return request.app.state.store.get(check_id(conversation_id))
    @router.post('/sessions/{conversation_id}/messages',status_code=202,tags=['consultation'])
    def message(conversation_id:str,body:MessageRequest,request:Request,idempotency_key:str|None=Header(default=None)):
        cid=check_id(conversation_id);payload=body.model_dump()
        return request.app.state.jobs.submit('consultation',payload,
            lambda rid,old:request.app.state.consultation.run(cid,payload,rid,old),cid=cid,key=check_key(idempotency_key))
    @router.get('/sessions/{conversation_id}/memory',tags=['memory'])
    def memory(conversation_id:str,request:Request):
        state=request.app.state.store.get(check_id(conversation_id))
        return {'patient_id':state['patient_id'],'memory':request.app.state.store.profile(state['patient_id'])}
    @router.post('/sessions/{conversation_id}/memory',tags=['memory'])
    def remember(conversation_id:str,body:MemoryRequest,request:Request):return request.app.state.store.remember(check_id(conversation_id),body.fact_ids)
    @router.delete('/sessions/{conversation_id}/memory',tags=['memory'])
    def forget(conversation_id:str,request:Request):return request.app.state.store.forget(check_id(conversation_id))
    @router.post('/knowledge/search',tags=['knowledge'])
    def search(body:SearchRequest,request:Request):return request.app.state.index.search_with_trace(body.query,body.task,body.limit)
    @router.get('/knowledge/status',tags=['knowledge'])
    def knowledge_status(request:Request):return request.app.state.index.describe()
    @router.post('/knowledge/index',status_code=202,tags=['knowledge'])
    def build_index(body:IndexBuildRequest,request:Request,idempotency_key:str|None=Header(default=None)):
        index=request.app.state.index
        if not hasattr(index,'build'):raise HTTPException(status_code=409,detail='index_build_requires_hybrid_mode')
        def build(rid,old):
            from .common import write_json
            report={'run_id':rid,'status':'completed','kind':'retrieval_index','index':index.build(body.force)}
            write_json(settings.path('index_runs_dir')/(rid+'.json'),report)
            return report
        return request.app.state.jobs.submit('retrieval_index',body.model_dump(),build,key=check_key(idempotency_key))
    @router.get('/graphs',tags=['service'])
    def graphs():
        from .agents import LeaderAgent
        leader=LeaderAgent(checkpointer=False)
        try:return {'leader':leader.graph.get_graph().draw_mermaid(),
            'specialties':{name:agent.graph.get_graph().draw_mermaid() for name,agent in leader.specialists.items()},
            'consultation':ConsultationService().agent.graph.get_graph().draw_mermaid()}
        finally:leader.close()
    app.include_router(router)
    return app


def serve(config_path=None):
    import uvicorn
    app=create_app(config_path);cfg=get_settings().data['http']
    # Single process: JobManager has a runtime-directory ownership lock.
    uvicorn.run(app,host=cfg['host'],port=cfg['port'],workers=1,access_log=False)
