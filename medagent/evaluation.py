"""Metrics never discard unparseable/missing predictions or hallucinated teeth."""
from collections import Counter
from .common import norm,tooth_key
from .rules import SYMPTOMS, assess, factor_items, normalize_output


def prf(tp,fp,fn):
    return {'tp':tp,'fp':fp,'fn':fn,
        'precision':tp/(tp+fp) if tp+fp else None,
        'recall':tp/(tp+fn) if tp+fn else None,
        'micro_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def grades(task,obj,rule_recompute=False):
    if task=='crack' or rule_recompute:
        return {x['tooth']:x['grade'] for x in assess(task,obj)['assessments']}
    obj=normalize_output(task,obj)
    return {tooth_key(x['position']):x.get('difficulty') for x in obj['toothLabel']}


def evaluate(task,rows,strict_schema=True):
    tp=fp=fn=0; good=denom=case_good=parsed=0; conf=Counter(); extra=missing=0
    field_good=field_total=0; errors=[]; ground_total=0;format_ok=0
    for row in rows:
        gold=normalize_output(task,row['gold'])
        ga={x for x in factor_items(task,gold) if x[1] in SYMPTOMS[task]}
        gg=grades(task,gold)
        pred_fields={}
        try:
            pred=normalize_output(task,row.get('prediction',''))
            teeth=pred if task=='crack' else pred['toothLabel']
            positions=[tooth_key(t.get('患牙编号' if task=='crack' else 'position','')) for t in teeth]
            if not all(positions) or len(positions)!=len(set(positions)):raise ValueError('invalid_tooth_identity')
            pred_fields=dict(zip(positions,teeth))
            format_ok+=1
            try:
                assess(task,pred)
                parsed+=1
            except (ValueError,KeyError,TypeError):
                if strict_schema: raise
                errors.append(row['id'])
            pa={x for x in factor_items(task,pred) if x[1] in SYMPTOMS[task]}
            try:pg=grades(task,pred)
            except (ValueError,KeyError,TypeError):
                # Preserve extracted symptoms; a risk grade with invalid rule options
                # is unknown. Irrelevant fields do not erase the symptom metrics.
                pg={position:None for position in positions}
        except (ValueError,KeyError,TypeError):
            pa=set();pg={};errors.append(row['id'])
        tp+=len(ga&pa);fp+=len(pa-ga);fn+=len(ga-pa)
        union=set(gg)|set(pg);denom+=len(union);ground_total+=len(gg)
        extra+=len(set(pg)-set(gg));missing+=len(set(gg)-set(pg))
        all_ok=set(gg)==set(pg)
        for tooth in union:
            hit=tooth in gg and tooth in pg and gg[tooth] is not None and pg[tooth]==gg[tooth]
            good+=hit;all_ok=all_ok and hit
            conf[(str(gg.get(tooth,'extra')),str(pg.get(tooth,'missing')))]+=1
            for field in SYMPTOMS[task]:
                a={x[2] for x in ga if x[:2]==(tooth,field)}
                b={x[2] for x in pa if x[:2]==(tooth,field)}
                field_total+=1
                field_good+=tooth in gg and tooth in pg and field in pred_fields.get(tooth,{}) and a==b
        case_good+=all_ok
    return {'samples':len(rows),'parse_ok':format_ok,'parse_schema_ok':parsed,'invalid_ids':sorted(set(errors)),
        'strict_schema_gate':strict_schema,
        'symptom_extraction':prf(tp,fp,fn),
        'symptom_field_exact':{'correct':field_good,'total':field_total,'accuracy':field_good/field_total if field_total else None},
        'grade':{'correct':good,'total':denom,'gold_teeth':ground_total,'extra_teeth':extra,'missing_teeth':missing,
            'accuracy':good/denom if denom else None,'case_exact_accuracy':case_good/len(rows) if rows else None,
            'mode':'rule_recomputed_both' if task=='crack' else 'model_grade_vs_stored_gold',
            'confusion':[{'gold':a,'prediction':b,'count':n} for (a,b),n in sorted(conf.items())]}}
