from .domain import DomainModel, tasks_for_request, validate_request
from .common import ROOT, digest, write_json, read_json
import uuid


class Pipeline:
    def __init__(self, client=None):
        self.client = client

    def run(self, request):
        text, _ = validate_request(request)
        tasks = tasks_for_request(request)
        results = {t: DomainModel(t, self.client).run(text) for t in tasks}
        report = {'run_id': uuid.uuid4().hex, 'version': read_json(ROOT/'configs/prototype.json')['version'],
                  'status': 'needs_review', 'routing_mode': 'manual', 'tasks': tasks,
                  'document': {'text': text, 'sha256': digest(text)}, 'results': results}
        write_json(ROOT/'runtime/prototype_runs'/(report['run_id']+'.json'), report)
        return report
