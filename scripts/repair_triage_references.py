"""Rebind training references after a deterministic triage rule correction.

Preserves frozen cards and all test labels. Requires the queue to be stopped;
the existing reference reconciler retains only actual valid saved trajectories.
"""
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT, read_json, write_json, digest
from medagent.consult_contracts import validate_facts, merge_facts
from medagent.reference_review import DATA, VERSION, source_digest, build
from medagent.triage import RULE_VERSION, urgent_fact_ids


def repair():
    state = read_json(ROOT / 'runtime/small_model/pipeline/status.json')
    worker = Path('/proc', str(state.get('pid')), 'stat')
    if worker.exists() and worker.read_text().split()[2] != 'Z':
        raise RuntimeError('stop_project_queue_before_reference_migration')
    archive = DATA / 'audit_history' / ('triage-rule-repair-' + str(int(time.time())))
    original_bindings = {}
    changed = []
    marker_by_id = {read_json(p)['id']: p for p in (DATA / 'generated/reference_review').glob('*.json')}
    cards = [read_json(p) for s in ('train', 'validation') for p in sorted((DATA / 'cards' / s).glob('*.json'))]
    if len(cards) != 540 or len(marker_by_id) != 540:
        raise ValueError('reference_migration_requires_complete_540_sources')
    for card in cards:
        path = DATA / 'reference_review' / (card['id'] + '.json')
        proof = read_json(path)
        if proof['version'] != VERSION or proof['source_sha256'] != source_digest(card):
            raise ValueError('reference_source_binding_mismatch:' + card['id'])
        original_bindings[card['id']] = source_digest(card)
        before = copy.deepcopy(proof['reference'])
        initial_urgent = urgent_fact_ids(card['initial_facts'])
        final_facts = copy.deepcopy(card['initial_facts'])
        for field, answer in card.get('patient_answers', {}).items():
            if not isinstance(answer, dict) or not answer.get('fact'):
                continue
            fact = copy.deepcopy(answer['fact'])
            sid = 'review-answer-' + field
            fact['source_id'] = sid
            incoming, issues = validate_facts({'facts': [fact]}, {sid: {'text': answer['text'], 'kind': 'patient_report'}})
            if issues:
                raise ValueError('ungrounded_patient_answer:' + card['id'])
            final_facts = merge_facts(final_facts, incoming)
        final_urgent = urgent_fact_ids(final_facts)
        if initial_urgent or final_urgent:
            proof['reference']['required_tasks'] = []
            proof['review']['validated_needs'] = {'candidates': []}
            if initial_urgent:
                proof['reference']['useful_questions'] = []
                proof['review']['question_review'] = {'fields': []}
        proof['triage_rule_version'] = RULE_VERSION
        proof['rule_rebinding'] = {'initial_urgent_fact_ids': initial_urgent,
            'final_urgent_fact_ids': final_urgent, 'source_unchanged': True,
            'clinical_validation': False}
        if before != proof['reference']:
            write_json(archive / 'references' / path.name, read_json(path))
            changed.append({'id': card['id'], 'before': before, 'after': proof['reference'],
                'urgent_fact_ids': initial_urgent or final_urgent})
        write_json(path, proof)
        write_json(marker_by_id[card['id']], proof)
    # Only the pre-existing reconciler may restore a saved candidate, after all
    # source, reward, evidence and full-completion checks pass.
    build(None, 1)
    for split in ('train', 'validation'):
        path = DATA / 'leader' / (split + '.json')
        if path.exists():
            path.replace(archive / ('leader-' + split + '.json'))
    for path in [DATA / 'leader/coverage.json', DATA / 'preflight_complete.json']:
        if path.exists():
            path.replace(archive / path.name)
    for card in cards:
        if source_digest(read_json(DATA / 'cards' / card['split'] / (card['id'].split('-', 1)[1] + '.json'))) != original_bindings[card['id']]:
            raise ValueError('frozen_card_changed_during_migration')
    report = {'status': 'complete', 'cards_checked': len(cards), 'changed_references': changed,
        'triage_rule_version': RULE_VERSION, 'source_documents_modified': False,
        'test_labels_modified': False, 'clinical_validation': False,
        'leader_rows_after_reconciliation': len(list((DATA / 'generated/leader').glob('*.json')))}
    write_json(archive / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    repair()
