"""JSON action rollouts with immutable token capture and external-token masks."""
import json
import os
import time
import uuid
from .common import ROOT,read_json,write_json,parse_json,digest
from .consultation import ConsultationAgent
from .consult_contracts import LEADER_PROMPT,visible_messages
from .routing import NEEDS_PROMPT,needs_view,validate_needs
from .frozen_models import validate_frozen_bindings
from .rollout_env import ConsultationEnvironment

def trajectory_reward(completions,trajectory_rewards,**kwargs):return trajectory_rewards

class AgentRollouts:
    def __init__(self,rows,folder,cfg):
        self.folder=folder
        self.run_id=os.environ.get('TORCHELASTIC_RUN_ID') or uuid.uuid4().hex
        self.rows={digest(json.dumps(r['prompt'],sort_keys=True,ensure_ascii=False)):r for r in rows}
        # Binding the fixed expert/answer versions prevents moving reward targets.
        self.stage=cfg.get('active_stage',{})
        frozen=read_json(ROOT/self.stage.get('frozen_models','runtime/small_model/frozen_models.json'))
        bindings=validate_frozen_bindings(frozen)
        config=read_json(ROOT/'configs/consultation.json')
        config['models'].update({role:{k:v for k,v in binding.items() if k not in {'checkpoint','checkpoint_sha256'}} for role,binding in bindings.items()})
        self.frozen=frozen;self.agent=ConsultationAgent(config=config)

    def __call__(self,prompts,trainer):
        import torch
        from trl.models import unwrap_model_for_generation
        tokenizer=getattr(trainer.processing_class,'tokenizer',trainer.processing_class)
        batches={k:[] for k in ['prompt_ids','completion_ids','logprobs','env_mask','trajectory_rewards']}
        # TRL RepeatSampler already expands each group into individual prompts.
        # Returning four outputs per repeated input would corrupt reward grouping.
        with unwrap_model_for_generation(trainer.model_wrapped,trainer.accelerator) as model:
            previous=model.training;model.eval()
            try:
                for candidate,prompt in enumerate(prompts):
                    row=self.rows[digest(json.dumps(prompt,sort_keys=True,ensure_ascii=False))]
                    routing_events=[]
                    def expert(task,state):
                        state['sources']=row['episode'].get('sources',{})|state.get('sources',{})
                        if self.frozen.get('offline_teacher_only'):
                            from .provider import LocalModelClient
                            from .generation_schemas import expert_schema
                            from .consult_contracts import REGISTRY,EXPERT_PROMPT
                            from .evidence import validate_expert
                            spec=REGISTRY[task];context={k:state[k] for k in ('facts','sources','knowledge')}
                            raw=LocalModelClient(self.agent.config['models']).complete('specialist',EXPERT_PROMPT.format(specialty=spec['name'],scope=spec['scope'])+'\n仅输出findings、directions、missing_information。',json.dumps(context,ensure_ascii=False),response_format=expert_schema(context))
                            result=parse_json(raw);result['reference_validation']=validate_expert(result,state)
                            used={fid for f in result['findings'] for fid in f['fact_ids']}
                            result['facts']=[f for f in state['facts'] if f['id'] in used]
                            return result
                        return self.agent.expert(task,state)
                    def analyse(state):
                        from .provider import LocalModelClient,ModelUnavailable
                        context=needs_view(state['facts'],recent_messages=visible_messages(state.get('messages',[])))
                        errors=[]
                        for attempt in range(2):
                            request=dict(context)
                            if errors:
                                request['validation_feedback']={'error':errors[-1],
                                    'instruction':'只使用registry内专业ID，每个task最多出现一次；保留原始事实，不补造依据。'}
                            try:
                                if self.frozen.get('offline_teacher_only'):
                                    from .generation_schemas import needs_schema
                                    raw=LocalModelClient(self.agent.config['models']).complete('needs',NEEDS_PROMPT,json.dumps(request,ensure_ascii=False),response_format=needs_schema(state['facts']))
                                else:
                                    raw=self.agent.call('needs',NEEDS_PROMPT,request)
                                result=validate_needs(parse_json(raw),state['facts'])
                                if errors:routing_events.append({'tool':'needs','status':'recovered','errors':errors,'attempts':attempt+1})
                                return result
                            except (ValueError,TypeError,KeyError,ModelUnavailable) as exc:
                                error=str(exc)[:120] if isinstance(exc,(ValueError,ModelUnavailable)) else type(exc).__name__
                                errors.append(error)
                                routing_events.append({'tool':'needs','status':'failed','error':error,'attempt':attempt+1})
                                if isinstance(exc,ModelUnavailable):break
                        # A failed tool remains visible; it is not evidence of no need.
                        return {'candidates':[],'status':'failed','error':'needs_analysis_unavailable',
                            'validation_errors':errors,'attempts':len(errors)}
                    env=ConsultationEnvironment(row['episode'],expert,lambda q,t:self.agent.index.search(q,t,3),self.agent.config,analyse=analyse)
                    live_prompt=[{'role':'system','content':LEADER_PROMPT},{'role':'user','content':json.dumps(env.view(),ensure_ascii=False)}]
                    initial=tokenizer.apply_chat_template(live_prompt,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
                    all_ids=list(initial);completion=[];mask=[];logprobs=[]
                    limit=trainer.args.max_completion_length
                    known_facts={f['id'] for f in env.state['facts']}
                    budget_exhausted=False
                    while not env.done and len(completion)<limit-192:
                        x=torch.tensor([all_ids],device=trainer.accelerator.device)
                        with torch.no_grad():
                            generation=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=min(192,limit-len(completion)),
                                do_sample=True,temperature=trainer.temperature,top_k=0,top_p=1.,return_dict_in_generate=True,output_scores=True,
                                pad_token_id=tokenizer.pad_token_id,use_cache=True,synced_gpus=False)
                        ids=generation.sequences[0,x.shape[1]:].tolist()
                        if not ids:break
                        lp=[float(torch.log_softmax(score[0].float(),-1)[token]) for score,token in zip(generation.scores,ids)]
                        all_ids+=ids;completion+=ids;mask += [1]*len(ids);logprobs+=lp
                        try:action=parse_json(tokenizer.decode(ids,skip_special_tokens=True))
                        except ValueError:action={'action':'invalid'}
                        observation=env.step(action)
                        if not env.done:
                            state_view=env.view()
                            # The original facts remain in the autoregressive context.
                            # Send updates instead of copying the full patient record
                            # and expert result twice after every action.
                            updated={k:v for k,v in state_view.items() if k not in {'facts','registry','limits','recent_messages','observations'}}
                            updated['new_facts']=[f for f in env.state['facts'] if f['id'] not in known_facts]
                            known_facts.update(f['id'] for f in env.state['facts'])
                            visible={'observation':{k:v for k,v in observation.items() if k!='results'},'state_update':updated}
                            suffix=tokenizer.apply_chat_template([{'role':'user','content':json.dumps(visible,ensure_ascii=False)}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
                            if len(completion)+len(suffix)+192>limit or len(all_ids)+len(suffix)+192>self.stage.get('max_length',16384):
                                budget_exhausted=True;break
                            all_ids+=suffix;completion+=suffix;mask += [0]*len(suffix);logprobs += [0.]*len(suffix)
                    reward=env.reward()
                    record={'group_id':row['group_id'],'candidate':candidate,'policy_step':trainer.state.global_step,
                        'dataset_split':row['split'],'rollout_phase':'training' if previous else 'validation',
                        'reward_version':'reference-links-v2','clinical_entailment_verified':False,
                        'policy_version':str(trainer.args.output_dir)+':'+self.run_id+':'+str(trainer.state.global_step),'frozen_models':self.frozen,
                        'training_run_id':self.run_id,
                        'prompt_ids':initial,'completion_ids':completion,'old_logprobs':logprobs,'action_mask':mask,
                        'rank':trainer.accelerator.process_index,'budget_exhausted':budget_exhausted,
                        'routing_events':routing_events,
                        'initialization':self.stage.get('initialization','parent_sft'),'actions':env.actions,'observations':env.observations,'reward':reward,'time':time.time()}
                    write_json(self.folder/(uuid.uuid4().hex+'.json'),record)
                    for key,value in [('prompt_ids',initial),('completion_ids',completion),('logprobs',logprobs),('env_mask',mask),('trajectory_rewards',reward)]:batches[key].append(value)
            finally:model.train(previous)
        return batches
