"""Typed, in-process capability contracts; transport adapters can use describe()."""
from abc import ABC, abstractmethod
import copy
import time
import re
from .token_budget import ContextBudgetExceeded
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class SkillInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    task: str | None = None
    text: str = ''
    images: list[str] = Field(default_factory=list)
    scope: list[str] = Field(default_factory=list)
    sources: dict[str, dict[str, Any]] = Field(default_factory=dict)
    facts: list[dict[str, Any]] = Field(default_factory=list)
    factors: Any = None
    feedback: str | None = None
    input_version: str = ''
    payload: dict[str, Any] = Field(default_factory=dict)


class SkillResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    skill: str
    version: str
    status: Literal['ok', 'failed']
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    retryable: bool = False
    input_version: str = ''
    elapsed_seconds: float = 0

    def unwrap(self):
        if self.status != 'ok':
            raise SkillExecutionError(self.skill, self.error or 'skill_failed', self.retryable)
        return self.data


class SkillExecutionError(RuntimeError):
    def __init__(self, skill, error, retryable=False):
        super().__init__(skill + ':' + error)
        self.skill, self.retryable = skill, retryable


class BaseSkill(ABC):
    name: str
    description = ''
    version = '1'
    required_fields = ()

    @abstractmethod
    def execute(self, request: SkillInput) -> dict:
        """Implement one bounded capability without accessing orchestration state."""
        raise NotImplementedError

    def invoke(self, request: SkillInput | dict) -> SkillResult:
        started = time.monotonic()
        value = None
        try:
            # A skill owns its input copy; mutable nested data cannot update sibling state.
            value = SkillInput.model_validate(request).model_copy(deep=True)
            for field in self.required_fields:
                if not getattr(value,field,None):
                    raise ValueError('required_skill_input:'+field)
            data = self.execute(value)
            if not isinstance(data, dict):
                raise TypeError('skill_output_must_be_object')
            return SkillResult(skill=self.name, version=self.version, status='ok', data=data,
                input_version=value.input_version, elapsed_seconds=time.monotonic() - started)
        except Exception as exc:
            retryable = exc.retryable if isinstance(exc, SkillExecutionError) else isinstance(exc, ValueError) and not isinstance(exc, ContextBudgetExceeded)
            # Never include inputs, credentials or model response bodies in error envelopes.
            error = str(exc)[:240] if isinstance(exc, (ValueError, SkillExecutionError)) or type(exc).__name__ == 'ModelUnavailable' else type(exc).__name__
            return SkillResult(skill=self.name, version=self.version, status='failed', error=error,
                retryable=retryable, input_version=value.input_version if value else '',
                elapsed_seconds=time.monotonic() - started)

    def describe(self):
        schema=SkillInput.model_json_schema()
        if self.required_fields:
            schema['required']=list(self.required_fields)
            if 'task' in self.required_fields:schema['properties']['task']={'type':'string','minLength':1}
        return {'name': self.name, 'description': self.description, 'version': self.version,
                'input_schema': schema, 'output_schema': SkillResult.model_json_schema()}


class SkillRegistry:
    def __init__(self):
        self._skills = {}
        self._sealed = False

    def register(self, skill: BaseSkill):
        if self._sealed:
            raise ValueError('skill_registry_is_sealed')
        if not isinstance(skill, BaseSkill) or not re.fullmatch(r'[a-z][a-z0-9_]*',getattr(skill, 'name', '')):
            raise TypeError('registered_skill_requires_BaseSkill')
        if skill.name in self._skills:
            raise ValueError('duplicate_skill:' + skill.name)
        self._skills[skill.name] = skill
        return self

    def seal(self):
        self._sealed = True
        return self

    def invoke(self, name, request: SkillInput | dict):
        if name not in self._skills:
            return SkillResult(skill=name, version='0', status='failed', error='unknown_skill')
        return self._skills[name].invoke(request)

    def run(self, name, **arguments):
        return self.invoke(name, arguments).unwrap()

    def require(self, *names):
        missing = set(names) - set(self._skills)
        if missing:
            raise ValueError('required_skills_missing:' + ','.join(sorted(missing)))
        return self

    def describe(self):
        return [copy.deepcopy(self._skills[name].describe()) for name in sorted(self._skills)]
