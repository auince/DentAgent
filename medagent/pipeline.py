import concurrent.futures
from .common import ROOT, read_json, digest
from .domain import validate_request
from .provider import LocalModelClient
from .harness import AgentHarness
from .leader import LeaderAgent, SpecialtyAgent
from .review_skill import ImageReviewSkill
from .image_tools import ImageTools


class Pipeline:
    def __init__(self,client=None):self.client=client or LocalModelClient()

    def run(self,request):
        text,images=validate_request(request,allow_images=True)
        parse_issues=[]
        if not text.strip():text,parse_issues=ImageTools(self.client,images).initial_text()
        settings=read_json(ROOT/'configs/prototype.json');h=AgentHarness()
        leader=LeaderAgent(self.client,settings['specialties'])
        document=leader.organize(text);tasks=leader.route(request,text)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)+1) as pool:
            review_job=pool.submit(h.call,'information_review',lambda s:ImageReviewSkill(self.client).run(s,images),text)
            jobs={t:pool.submit(h.call,t,SpecialtyAgent(t,self.client).run,text) for t in tasks}
            drafts={t:job.result() for t,job in jobs.items()};review=review_job.result()
        if review.get('status')=='failed':review={'text':text,'approved':False,'issues':[{'code':'review_failed'}]}
        revised=review['text'];changed=digest(revised)!=digest(text)
        if changed:
            h.log('text_version_check','invalidated',old_sha256=digest(text),new_sha256=digest(revised))
            tasks=leader.route(request,revised)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                jobs={t:pool.submit(h.call,'reflow_'+t,SpecialtyAgent(t,self.client).run,revised) for t in tasks}
                results={t:j.result() for t,j in jobs.items()}
        else:results=drafts
        for r in results.values():
            if r.get('source_sha256')!=digest(revised):raise ValueError('stale_specialty_result')
        return h.save({'version':settings['version'],'status':'needs_review','tasks':tasks,
                       'document':document,'reviewed_document':{'text':revised,'sha256':digest(revised)},
                       'review':review,'parse_issues':parse_issues,'results':results,
                       'discarded_drafts':drafts if changed else {},'summary':leader.aggregate(results)})
