"""Component rewards expose denominators; schema validity cannot hide factual errors."""
import json
from .common import parse_json

def atom_f1(pred,gold):
    p={json.dumps(x,sort_keys=True,ensure_ascii=False) for x in pred}
    g={json.dumps(x,sort_keys=True,ensure_ascii=False) for x in gold}
    return 2*len(p&g)/max(1,len(p)+len(g))

def fact_atoms(facts):
    return [(f.get('name'),f.get('value'),f.get('status'),f.get('tooth'),f.get('location')) for f in facts]

def specialist_components(output,reference):
    try:
        p=parse_json(output);g=parse_json(reference['gold']) if isinstance(reference['gold'],str) else reference['gold']
        if reference.get('legacy_task'):
            from .rules import assess,factor_items
            task=reference['legacy_task']
            # Legacy scoring is delegated to the exact production factor flattening.
            pf=factor_items(task,p);gf=factor_items(task,g)
            factors=atom_f1(list(pf),list(gf))
            rules=float(assess(task,p)['assessments']==assess(task,g)['assessments'])
        else:
            if not isinstance(p,dict):raise ValueError()
            factors=atom_f1(fact_atoms(p.get('facts',[])),fact_atoms(g.get('facts',[])))
            rules=float(p.get('decision')==g.get('decision')) if 'decision' in g else factors
        facts=p.get('facts',[]) if isinstance(p,dict) else []
        source=reference.get('text','')
        grounded=sum(bool(f.get('quote')) and f['quote'] in source and (not f.get('tooth') or str(f['tooth']) in f['quote']) for f in facts)
        evidence=grounded/max(1,len(facts))
        return {'format':1.,'facts':factors,'evidence':evidence,'consistency':rules}
    except (ValueError,KeyError,TypeError):return dict.fromkeys(['format','facts','evidence','consistency'],0.)

def specialist_reward(completions,gold,**kwargs):
    scores=[]
    for i,c in enumerate(completions):
        text=c[-1]['content'] if isinstance(c,list) else c
        ref={'gold':gold[i]}
        for k in ['text','legacy_task']:
            if k in kwargs:ref[k]=kwargs[k][i]
        v=specialist_components(text,ref)
        scores.append(.05*v['format']+.5*v['facts']+.3*v['evidence']+.15*v['consistency'])
    return scores

def trajectory_completion(actions,observations,reference):
    """Hard imitation-data gate; a weighted reward alone permits incomplete routes."""
    if len(actions)!=len(observations):return {'passed':False,'error':'unmatched_actions_observations'}
    if any(not isinstance(a,dict) or not isinstance(o,dict) for a,o in zip(actions,observations)):
        return {'passed':False,'error':'invalid_action_or_observation_object'}
    called=set();obtained=set();finished=False
    for action,observation in zip(actions,observations):
        if observation.get('expert_results_invalidated'):called.clear()
        if action.get('action')=='dispatch':called.update(observation.get('successful_tasks',[]))
        if action.get('action')=='ask' and observation.get('available'):obtained.add(action.get('field'))
        if action.get('action') in {'finish','refer'} and observation.get('status')=='ok':finished=True
    required=set(reference['required_tasks']);questions=set(reference.get('useful_questions',[]))
    return {'passed':finished and called==required and questions<=obtained,
        'missing_tasks':sorted(required-called),'extra_tasks':sorted(called-required),
        'missing_questions':sorted(questions-obtained),'finished':finished}

def trajectory_score(actions,observations,reference):
    required=set(reference['required_tasks']);called=set()
    relevant=set(reference.get('useful_questions',[]));obtained=set()
    invalid=0;repeated=0;signatures=set();finished=False
    for a,o in zip(actions,observations):
        signature=json.dumps(a,sort_keys=True)
        repeated+=int(signature in signatures);signatures.add(signature)
        malformed=not isinstance(a,dict) or not isinstance(o,dict)
        a=a if isinstance(a,dict) else {}
        o=o if isinstance(o,dict) else {}
        invalid+=int(malformed or o.get('status')=='failed')
        if o.get('expert_results_invalidated'):called.clear()
        if a.get('action')=='dispatch':called.update(o.get('successful_tasks',a.get('tasks',[]) if o.get('status')=='ok' else []))
        if a.get('action')=='ask' and o.get('available'):obtained.add(a.get('field'))
        if a.get('action') in {'finish','refer'}:finished=True
    coverage=atom_f1(list(called),list(required)) if required else float(not called)
    info=len(obtained&relevant)/max(1,len(relevant)) if relevant else float(not obtained)
    supported=sum(o.get('reference_links_valid',False) for o in observations if o.get('tool')=='expert')
    experts=sum(o.get('tool')=='expert' for o in observations)
    evidence=supported/max(1,experts)
    completion=float(finished and called==required and (not relevant or relevant<=obtained))
    score=.3*coverage+.2*info+.3*evidence+.2*completion-.08*invalid-.05*repeated-.01*max(0,len(actions)-3)
    return max(-1.,min(1.,score))
