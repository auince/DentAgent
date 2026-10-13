import concurrent.futures
from .common import ROOT, read_json
from .domain import validate_request
from .provider import LocalModelClient
from .harness import AgentHarness
from .leader import LeaderAgent, SpecialtyAgent


class Pipeline:
    def __init__(self, client=None):
        self.client = client or LocalModelClient()

    def run(self, request):
        text, _ = validate_request(request)
        settings = read_json(ROOT/'configs/prototype.json')
        harness = AgentHarness()
        leader = LeaderAgent(self.client, settings['specialties'])
        document = leader.organize(text)
        tasks = leader.route(request, text)
        harness.log('leader.route', 'finished', tasks=tasks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {t: pool.submit(harness.call, t, SpecialtyAgent(t,self.client).run, text) for t in tasks}
            results = {t: f.result() for t, f in futures.items()}
        return harness.save({'version': settings['version'], 'status': 'needs_review',
                             'document': document, 'tasks': tasks, 'results': results,
                             'summary': leader.aggregate(results)})
