"""Leader record-review entry point, usable independently of specialty extraction."""
import time
from .common import digest,write_json
from .domain import validate_request
from .provider import LocalModelClient
from .settings import configured_path,get_settings
from .skills import build_skill_registry

class RecordReviewService:
    def __init__(self,client=None):
        self.registry=build_skill_registry(client or LocalModelClient())
    def run(self,request,run_id):
        started=time.monotonic();text,images=validate_request(request,allow_images=True)
        document=self.registry.run('document_parse',text=text,images=images)
        review=self.registry.run('information_review',text=document['text'],images=images)
        issues=document['issues']+review['issues']
        report={'run_id':run_id,'status':'completed' if review['approved'] and not issues else 'needs_review',
            'owner':'leader','kind':'record_review','version':get_settings().data['version'],
            'document':document,'review':review,'issues':issues,
            'original_sha256':digest(document['text']),'reviewed_sha256':digest(review['text']),
            'changed':document['text']!=review['text'],'clinical_verification':False,
            'elapsed_seconds':round(time.monotonic()-started,3)}
        write_json(configured_path('review_runs_dir')/(run_id+'.json'),report)
        return report
