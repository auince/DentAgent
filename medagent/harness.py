"""AgentHarness: per-run immutable input identity, tool tracing and failure isolation."""
import threading
import time
import uuid
from .common import ROOT, digest, write_json


class AgentHarness:
    def __init__(self):
        self.run_id = uuid.uuid4().hex
        self.events = []
        self.lock = threading.Lock()

    def log(self, step, status, **details):
        with self.lock:
            self.events.append({'seq': len(self.events), 'time': time.time(),
                                'step': step, 'status': status, **details})

    def call(self, step, fn, text):
        sha = digest(text)
        self.log(step, 'started', source_sha256=sha)
        try:
            result = fn(text)
            result['source_sha256'] = sha
            self.log(step, 'finished', source_sha256=sha)
            return result
        except Exception as exc:
            self.log(step, 'failed', error=str(exc)[:240], source_sha256=sha)
            return {'status': 'failed', 'error': str(exc)[:240], 'source_sha256': sha}

    def save(self, report):
        report.update(run_id=self.run_id, trace=list(self.events))
        write_json(ROOT/'runtime/prototype_runs'/(self.run_id+'.json'), report)
        return report
