"""Exact view counts and grouped source coverage, separate from clinical validity."""
from collections import Counter
from .common import ROOT,read_json,write_json,digest,parse_json
from .training_data import assert_disjoint,assert_training_rows

def audit_view(data,view,expected_new_rows=None):
    cfg=read_json(ROOT/'configs/small_training.json')
    target=cfg['data_targets'][view] if expected_new_rows is None else expected_new_rows
    if not isinstance(target,int) or target<1 or target>cfg['data_targets'][view]:raise ValueError('invalid_explicit_view_count')
    source={read_json(p)['group_id']:read_json(p) for s in ('train','validation') for p in (data/'cards'/s).glob('*.json')}
    heldout={read_json(p)['group_id'] for p in (data/'cards/test').glob('*.json')}
    if len(heldout)!=1000:raise ValueError('full_research_test_not_intact')
    report={'status':'passed','view':view,'target_new_rows':target,'clinical_validation':False,
        'scope':'counts_group_isolation_source_binding_and_coverage','planned_new_rows':cfg['data_targets'][view],
        'count_relaxed_by_user':expected_new_rows is not None,'splits':{}}
    splits={s:read_json(data/view/(s+'.json')) for s in ('train','validation')}
    assert_disjoint(splits['train'],splits['validation'])
    new_rows=[]
    for split,rows in splits.items():
        assert_training_rows(rows,split)
        if {r['group_id'] for r in rows}&heldout:raise ValueError('view_test_group_overlap:'+view)
        synthetic=[r for r in rows if not r.get('legacy_task')];new_rows+=synthetic
        coverage={k:Counter() for k in ('specialty','category','profile','input_kind','fact_status','reference_tasks','augmentation_method')}
        for row in synthetic:
            card=source.get(row['group_id'])
            if not card or card['split']!=split:raise ValueError('view_source_split_mismatch:'+view)
            if row['provenance']!=card['provenance']:raise ValueError('view_source_provenance_mismatch:'+view)
            if card['provenance'].get('teacher')!='Qwen3.8-27B':raise ValueError('view_wrong_teacher:'+view)
            if view=='specialist':
                from .evidence import validate_expert
                validate_expert(parse_json(row['gold']),parse_json(row['prompt'][-1]['content']))
            if view=='leader':
                from .reference_review import reviewed_reference
                from .consult_contracts import question_field_issue,finish_issue
                from .training_rewards import trajectory_score,trajectory_completion
                ref=reviewed_reference(card)
                if any(row['episode'][k]!=ref[k] for k in ('required_tasks','useful_questions')):raise ValueError('leader_stale_reference')
                if row.get('trajectory_reward',0)<.6:raise ValueError('leader_reward_below_gate')
                actions=[parse_json(t['completion'][0]['content']) for t in row['trajectory'][1:]]
                if not trajectory_completion(actions,row['tool_observations'],row['episode'])['passed']:raise ValueError('leader_incomplete_reference_task:'+row['id'])
                if any(question_field_issue(a) for a in actions):raise ValueError('leader_question_field_mismatch:'+row['id'])
                for action,turn in zip(actions,row['trajectory'][1:]):
                    if action.get('action')=='finish':
                        policy_input=parse_json(turn['prompt'][-1]['content'])
                        if finish_issue(policy_input,policy_input['limits']):raise ValueError('leader_premature_finish:'+row['id'])
                for observation in row['tool_observations']:
                    if observation.get('tool')=='expert':
                        for task in observation.get('successful_tasks',[]):
                            result=observation.get('results',{}).get(task,{})
                            if not result.get('findings') or any(not isinstance(f.get('interpretation'),str) or not f['interpretation'].strip() for f in result['findings']):
                                raise ValueError('leader_empty_expert_evidence:'+row['id'])
                if abs(trajectory_score(actions,row['tool_observations'],row['episode'])-row['trajectory_reward'])>1e-8:
                    raise ValueError('leader_reward_binding_mismatch:'+row['id'])
                coverage['reference_tasks'].update(row['episode']['required_tasks'])
            coverage['specialty'].update([row['specialty']])
            coverage['category'].update([card['category']]);coverage['profile'].update([card['profile']])
            coverage['input_kind'].update(['record' if card.get('record_available_initially') else 'oral'])
            coverage['fact_status'].update(f['status'] for f in card['initial_facts'])
            coverage['augmentation_method'].update([row.get('augmentation_method','original_generator_v1')])
        report['splits'][split]={'rows':len(rows),'new_rows':len(synthetic),
            'independent_source_groups':len({r['group_id'] for r in synthetic}),
            'legacy_rows':len(rows)-len(synthetic),'coverage':{k:dict(v) for k,v in coverage.items()},
            'sha256':digest((data/view/(split+'.json')).read_bytes())}
    if len(new_rows)!=target or len({r['id'] for r in new_rows})!=target:raise ValueError('view_exact_count_or_identity_mismatch:'+view)
    write_json(data/view/'coverage.json',report)
    return report
