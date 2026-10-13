"""Request contracts used for validation and automatic OpenAPI documentation."""
import re
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator
from .settings import get_settings

class StrictBody(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)

class AssessmentRequest(StrictBody):
    text:str=''
    images:list[str]=Field(default_factory=list)
    tasks:list[str]|None=None
    routing_mode:Literal['manual','auto']|None=None
    instruction:str='综合评估'
    @model_validator(mode='after')
    def check_input(self):
        cfg=get_settings().data['limits']
        if len(self.text)>cfg['record_chars'] or len(self.images)>cfg['images']:raise ValueError('assessment_input_too_large')
        if not self.text.strip() and not self.images:raise ValueError('empty_document')
        if len(self.instruction)>cfg['chat_chars']:raise ValueError('instruction_too_large')
        from .consult_contracts import REGISTRY
        mode=self.routing_mode or ('manual' if self.tasks is not None else 'auto')
        if self.tasks is not None and (not self.tasks or len(set(self.tasks))!=len(self.tasks) or any(t not in REGISTRY for t in self.tasks)):raise ValueError('invalid_specialty_tasks')
        if mode=='manual' and not self.tasks:raise ValueError('manual_route_requires_tasks')
        if mode=='auto' and self.tasks is not None:raise ValueError('auto_route_cannot_also_select_tasks')
        self.routing_mode=mode
        return self

class SessionRequest(StrictBody):
    patient_id:str|None=None
    @model_validator(mode='after')
    def check_id(self):
        if self.patient_id is not None and not re.fullmatch('[a-f0-9]{32}',self.patient_id):raise ValueError('invalid_patient_id')
        return self

class MessageRequest(StrictBody):
    text:str=''
    record_text:str=''
    images:list[str]=Field(default_factory=list)
    @model_validator(mode='after')
    def check_input(self):
        cfg=get_settings().data['limits']
        if not (self.text.strip() or self.record_text.strip() or self.images):raise ValueError('empty_message')
        if len(self.text)+len(self.record_text)>cfg['chat_chars'] or len(self.images)>cfg['images']:raise ValueError('message_too_large')
        return self

class MemoryRequest(StrictBody):
    fact_ids:list[str]=Field(min_length=1,max_length=64)

class SearchRequest(StrictBody):
    query:str=Field(min_length=1)
    task:str|None=None
    limit:int|None=None
    @model_validator(mode='after')
    def check_input(self):
        cfg=get_settings().data['limits']
        from .consult_contracts import REGISTRY
        if not self.query.strip() or len(self.query)>cfg['search_query_chars']:raise ValueError('invalid_search_query')
        if self.task is not None and self.task not in REGISTRY:raise ValueError('unsupported_specialty')
        if self.limit is not None and not 1<=self.limit<=cfg['search_limit']:raise ValueError('invalid_search_limit')
        return self
