"""The harness records compiled LangGraph runs without assuming clinical validation."""
from .settings import configured_path
from .agents import LeaderAgent
from .domain import validate_request, tasks_for_request
from .common import ROOT, read_json
from .harness import AgentHarness
from .settings import get_settings


class Pipeline:
    def __init__(self,client=None):self.client=client

    def run(self,request,run_id=None):
        validate_request(request,allow_images=True)
        request=dict(request)
        mode=request.pop('routing_mode','manual' if request.get('tasks') is not None else 'auto')
        if mode=='manual':request['tasks']=tasks_for_request(request)
        elif mode=='auto':request.pop('tasks',None)
        else:raise ValueError('invalid_routing_mode')
        harness=AgentHarness(run_id);harness.log('langgraph.leader','started')
        agent=LeaderAgent(client=self.client,runtime=configured_path('graph_dir'))
        try:report=agent.run(request,run_id=harness.run_id)
        finally:agent.close()
        harness.log('langgraph.leader','finished',status_result=report['status'])
        report['graph_trace']=report.pop('trace',[])
        report['version']=get_settings().data['version']
        return harness.save(report)
