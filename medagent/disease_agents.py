"""Specialty subgraphs orchestrate registered skills; model and rule work live in skills."""
import operator
import time
from typing import TypedDict, Annotated
from langgraph.graph import START, END, StateGraph
from .common import digest
from .consult_contracts import REGISTRY
from .skill_registry import SkillExecutionError
from .skills import build_skill_registry


def trace_event(step, status='done', **details):
    return {'time': time.time(), 'step': step, 'status': status, **details}


class DiseaseState(TypedDict, total=False):
    text: str
    scope: list
    attempt: int
    facts: list
    sources: dict
    error: str
    failed: bool
    fatal: bool
    evidence: list
    result: dict
    context_audit: dict
    input_version: str
    trace: Annotated[list, operator.add]


class DiseaseAgent:
    def __init__(self, task, client=None, registry=None):
        if task not in REGISTRY:
            raise ValueError('unsupported_disease_agent')
        if registry is None:
            from .provider import LocalModelClient
            registry = build_skill_registry(client or LocalModelClient())
        self.task, self.registry = task, registry
        self.registry.require('evidence_select','symptom_extract','evidence_ground','rule_assess','analysis_explain')
        graph = StateGraph(DiseaseState)
        for name, fn in [('extract', self.extract), ('ground', self.ground),
                         ('rule_assess', self.rule_assess), ('explain', self.explain), ('finish', self.finish)]:
            graph.add_node(name, fn)
        graph.add_edge(START, 'extract')
        graph.add_conditional_edges('extract', lambda s: 'ground' if not s.get('error') else
            'extract' if s['attempt'] < 2 and not s.get('fatal') else 'finish')
        graph.add_edge('ground', 'rule_assess'); graph.add_edge('rule_assess', 'explain')
        graph.add_edge('explain', 'finish'); graph.add_edge('finish', END)
        self.graph = graph.compile(checkpointer=False, name='specialty_' + task)

    def extract(self, state):
        attempt = state.get('attempt', 0) + 1
        try:
            value = self.registry.run('symptom_extract', task=self.task, text=state['text'],
                sources=state['sources'], scope=state.get('scope', []), feedback=state.get('error') or None,
                input_version=state['input_version'], payload={'mode': 'facts'})
            return {'attempt': attempt, 'facts': value['facts'], 'error': '', 'failed': False,
                'trace': [trace_event(self.task + '.extract', attempt=attempt)]}
        except SkillExecutionError as exc:
            return {'attempt': attempt, 'facts': [], 'error': str(exc)[:240], 'failed': True,
                'fatal': not exc.retryable, 'trace': [trace_event(self.task + '.extract', 'failed', attempt=attempt)]}

    def ground(self, state):
        value = self.registry.run('evidence_ground', task=self.task, text=state['text'],
            sources=state['sources'], facts=state['facts'], payload={'mode': 'facts'})
        return {'evidence': value['evidence'], 'trace': [trace_event(self.task + '.ground')]}

    def rule_assess(self, state):
        value = self.registry.run('rule_assess', task=self.task, facts=state['facts'], payload={'mode': 'facts'})
        result = {'factors': state['facts'], 'facts': state['facts'], 'sources': state['sources'],
            'evidence': state['evidence'], 'issues': [], **value}
        if not state['context_audit']['complete']:
            result['issues'].append({'code': 'context_evidence_omitted'})
        return {'result': result, 'trace': [trace_event(self.task + '.rule_assess')]}

    def explain(self, state):
        result = dict(state['result'])
        try:
            result.update(self.registry.run('analysis_explain', task=self.task, text=state['text'], payload={'result': result}))
            if self.task in {'root_canal', 'crack'}:
                result['issues'].append({'code': 'insufficient_rule_factors_require_clinical_review'})
        except SkillExecutionError:
            result.setdefault('risk_assessments', [])
            result['issues'].append({'code': 'specialty_explanation_failed'})
        return {'result': result, 'trace': [trace_event(self.task + '.explain',
            'done' if result.get('analysis_explanation') else 'failed')]}

    def finish(self, state):
        common = {'task': self.task, 'role': 'structured_symptom_and_risk_analysis',
            'source_sha256': digest(state['text']), 'input_version': state['input_version'],
            'context_audit': state['context_audit']}
        if state.get('failed') or state.get('error'):
            return {'result': {**common, 'status': 'failed', 'error': state.get('error'), 'issues': []}}
        return {'result': {**state['result'], **common,
            'status': 'needs_review' if state['result']['issues'] else 'extracted',
            'patient_factors': [f for f in state['facts'] if f.get('tooth') is None],
            'tooth_factors': [f for f in state['facts'] if f.get('tooth') is not None]}}

    def invoke(self, text, images=(), scope=None):
        return self.invoke_sources({'record': {'text': text, 'kind': 'clinical_record'}}, scope)

    def invoke_sources(self, sources, scope=None, facts=(), input_version='', archive_audit=None):
        import json
        version = input_version or digest(json.dumps(sources, sort_keys=True, ensure_ascii=False))
        selected = self.registry.run('evidence_select', task=self.task, sources=sources,
            facts=list(facts), scope=scope or [], input_version=version)
        if archive_audit:
            selected['audit']['archive']=archive_audit
            if archive_audit.get('facts_truncated') or archive_audit.get('sources_truncated'):
                selected['audit']['complete']=False
        return self.graph.invoke({'text': selected['text'], 'sources': selected['sources'], 'scope': scope or [],
            'attempt': 0, 'trace': [], 'context_audit': selected['audit'], 'input_version': version}, {'recursion_limit': 12})
