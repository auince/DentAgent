"""Isolated consultation environment; hidden reference never enters observations."""
import copy
from .consult_contracts import REGISTRY,validate_action,validate_facts,merge_facts
from .training_rewards import trajectory_score
from .triage import urgent_fact_ids

class ConsultationEnvironment:
    def __init__(self,reference,expert,search,config,analyse=None):
        self._reference=copy.deepcopy(reference)
        self._expert=expert;self._search=search;self.config=config
        self.actions=[];self.observations=[]
        self.state={'facts':copy.deepcopy(reference['initial_facts']),'experts':{},'knowledge':[],
            'question_count':0,'asked_questions':[],'asked_fields':[],'tool_count':0,'images':[],'sources':copy.deepcopy(reference.get('sources',{}))}
        patient_source=self.state['sources'].get('patient',{})
        self.state['messages']=[{'role':'user','content':patient_source['text']}] if patient_source.get('text') else []
        self.done=False
        self._analyse=analyse
        self.state['urgent']=bool(urgent_fact_ids(self.state['facts']))
        if analyse:self.state['needs']={'candidates':[]} if self.state['urgent'] else analyse(self.state)

    def view(self):
        from .consult_contracts import policy_view
        return copy.deepcopy(policy_view(self.state,self.config,self.observations))

    def step(self,action):
        if self.done:raise ValueError('episode_finished')
        observation={}
        try:
            cost=validate_action(action,self.state,self.config)
            # Budget is spent on attempts, including failed tool executions.
            self.state['tool_count']+=cost
            kind=action['action']
            if kind=='ask':
                # An explicit question key selects a fact; unknowns stay unknown.
                key=action.get('field',action.get('question'))
                answer=self._reference.get('patient_answers',{}).get(key)
                observation={'tool':'patient','answer':answer.get('text','不知道') if isinstance(answer,dict) else answer if answer is not None else '不知道','available':answer is not None,'status':'ok'}
                self.state['messages'] += [{'role':'assistant','content':action['question']},{'role':'user','content':observation['answer']}]
                self.state['question_count']+=1;self.state['asked_questions'].append(action['question']);self.state['asked_fields'].append(key)
                if isinstance(answer,dict) and 'fact' in answer:
                    fact=copy.deepcopy(answer['fact']);sid='followup-'+str(self.state['question_count']);fact['source_id']=sid
                    source={sid:{'text':answer.get('text',''),'kind':'patient_report'}}
                    facts,issues=validate_facts({'facts':[fact]},source)
                    if issues:raise ValueError('simulator_answer_fact_not_grounded')
                    self.state['sources'].update(source);self.state['facts']=merge_facts(self.state['facts'],facts)
                    self.state['experts']={}
                    self.state['urgent']=bool(urgent_fact_ids(self.state['facts']))
                    observation['expert_results_invalidated']=True
                    if self._analyse:self.state['needs']={'candidates':[]} if self.state['urgent'] else self._analyse(self.state)
            elif kind=='dispatch':
                from .evidence import validate_expert
                outputs={};successful=[]
                for task in action['tasks']:
                    try:
                        result=self._expert(task,copy.deepcopy(self.state))
                        result['reference_validation']=validate_expert(result,self.state)
                        outputs[task]=result;successful.append(task)
                    except Exception as exc:outputs[task]={'status':'failed','error':str(exc)[:120]}
                self.state['experts'].update(outputs)
                observation={'tool':'expert','results':outputs,'status':'ok' if len(successful)==len(outputs) else 'failed',
                    'successful_tasks':successful,'reference_links_valid':bool(successful) and len(successful)==len(outputs),
                    'clinical_entailment_verified':False}
            elif kind=='retrieve':
                hits=self._search(action['query'],None if action['task']=='all' else action['task']);self.state['knowledge']+=hits
                observation={'tool':'knowledge_search','results':hits,'status':'ok'}
            elif kind in {'finish','refer'}:
                self.done=True;observation={'status':'ok'}
            else:raise ValueError('unsupported_training_modality')
        except Exception as exc:observation={'status':'failed','error':str(exc)[:120]}
        self.actions.append(copy.deepcopy(action));self.observations.append(observation)
        if len(self.actions)>=self.config['max_actions']:self.done=True
        return observation

    def reward(self):return trajectory_score(self.actions,self.observations,self._reference)
