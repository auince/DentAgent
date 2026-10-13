from .common import parse_json
from .evaluation import grades
from .rules import assess,factor_items


def structured_reward(task,prediction,gold):
    try:
        result=assess(task,prediction)
        a=set(factor_items(task,gold));b=set(factor_items(task,result['factors']))
        f1=2*len(a&b)/(len(a)+len(b)) if a or b else 1.0
        gg=grades(task,gold);pg=grades(task,result['factors'])
        union=set(gg)|set(pg)
        accuracy=sum(t in gg and t in pg and gg[t]==pg[t] for t in union)/len(union) if union else 0
        return .1+.6*f1+.3*accuracy
    except (ValueError,TypeError,KeyError): return 0.0


def grpo_reward(completions,gold,task,**kwargs):
    values=[]
    for completion,target,kind in zip(completions,gold,task):
        text=''.join(m.get('content','') for m in completion) if isinstance(completion,list) else completion
        values.append(structured_reward(kind,text,parse_json(target)))
    return values
