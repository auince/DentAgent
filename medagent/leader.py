"""Leader organizes only source-anchored sections and dispatches domain agents."""
import json
from .domain import BOUNDARY, tasks_for_request, DomainModel
from .common import parse_json, digest


class SpecialtyAgent:
    def __init__(self, task, client):
        self.task, self.model = task, DomainModel(task, client)

    def run(self, text):
        return self.model.run(text)


class LeaderAgent:
    def __init__(self, client, supported):
        self.client, self.supported = client, supported

    def organize(self, text):
        raw = parse_json(self.client.complete('leader', BOUNDARY +
            '整理病历栏目，只返回 {"sections":[{"name":"栏目名","quote":"原文连续片段"}]}。'
            '每个栏目引用原文，不改写输入、不补充医学推断。', text))
        if not isinstance(raw, dict) or not isinstance(raw.get('sections'), list):
            raise ValueError('invalid_sections')
        sections = []
        for item in raw['sections']:
            if not isinstance(item, dict) or not isinstance(item.get('name'), str):
                raise ValueError('invalid_section')
            quote = item.get('quote')
            if not isinstance(quote, str) or not quote or quote not in text:
                raise ValueError('unlocated_section')
            start = text.index(quote)
            sections.append({**item, 'start': start, 'end': start+len(quote)})
        return {'text': text, 'sha256': digest(text), 'sections': sections}

    def route(self, request, text):
        mode = request.get('routing_mode', 'manual' if request.get('tasks') is not None else 'auto')
        if mode == 'manual':
            return tasks_for_request(request)
        if mode != 'auto':
            raise ValueError('invalid_routing_mode')
        plan = parse_json(self.client.complete('leader', BOUNDARY+
            '根据病历与医生指令选择专病。只返回 {"tasks":["root_canal或crack"],'
            '"evidence":["原文引文"]}。根管难度须有当前根管治疗/诊断的临床记录，'
            '仅牙疼不能评难度。不得无视医生明确排除的任务；不清楚时 tasks=[]。',
            json.dumps({'text': text, 'instruction': request.get('instruction', ''),
                        'supported': self.supported}, ensure_ascii=False)))
        if not isinstance(plan, dict) or not isinstance(plan.get('tasks'), list):
            raise ValueError('invalid_route')
        if not plan['tasks']:
            return []
        tasks = tasks_for_request({'tasks': plan['tasks']})
        quotes = plan.get('evidence')
        if not isinstance(quotes, list) or not quotes or any(not isinstance(q,str) or not q or q not in text for q in quotes):
            raise ValueError('unlocated_route_evidence')
        return tasks

    def aggregate(self, results):
        # Keep the specialist grades intact; Leader owns presentation only.
        return [{'task': task, 'status': result['status'],
                 'assessments': result.get('assessments', [])} for task, result in results.items()]
