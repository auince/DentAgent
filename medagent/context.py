"""Bounded model views with source spans; summaries never become clinical sources."""
import copy
import json
import re
from .common import digest
from .token_budget import TokenCounter, ContextBudgetExceeded, bounded_items, context_options

BACKGROUND = ('过敏', '糖尿病', '抗凝', '妊娠', '怀孕', '系统病', '既往史', '病史', '张口', '配合', '年龄',
              'allerg', 'diabet', 'anticoagul', 'pregnan', 'medical history', 'smoking', '吸烟')
URGENT = ('呼吸', '吞咽', '视力', '眼周', '广泛肿胀', 'breathing', 'swallow', 'vision')
TERMS = {
    'root_canal': ('根管', '初治', '再治', '根充', '髓腔', '钙化', '弯曲', '开口', 'root canal', 'retreatment'),
    'crack': ('裂', '咬合', '咬硬', '松口', '松开', '裂纹', 'crack', 'biting'),
    'pulp_periapical': ('牙髓', '根尖', '自发痛', '夜间痛', '冷热', '冷痛', '热痛', '牙痛', '牙疼', '叩', '窦道', '脓肿', '活力', 'pulp', 'periapical', 'abscess'),
    'periodontal': ('牙周', '牙龈', '出血', '探诊', '附着', '骨吸收', '松动', '菌斑', '洁治', 'periodont', 'gingiv', 'bleeding'),
}
TEETH = re.compile(r'(?<![\d.])([1-4][1-8])(?![\d.]|\s*(?:岁|年|mm|毫米|%|％))')


def normalized_scope(scope):
    return {str(value).removesuffix('牙').strip() for value in (scope or [])}


def task_tags(fact_or_text):
    if isinstance(fact_or_text, dict):
        text = ' '.join(str(fact_or_text.get(k, '')) for k in ('name', 'value', 'quote', 'clinical_key'))
        declared = fact_or_text.get('task')
    else:
        text, declared = str(fact_or_text), None
    lower = text.lower()
    tags = {task for task, words in TERMS.items() if any(word in lower for word in words)}
    if declared in TERMS:
        tags.add(declared)
    key = fact_or_text.get('clinical_key', '') if isinstance(fact_or_text, dict) else ''
    if key.startswith(('max_', 'periodont', 'prior_stage', 'cigarettes', 'hba1c', 'bone_loss')):
        tags.add('periodontal')
    return sorted(tags)


def background(value):
    text = str(value).lower()
    return any(word in text for word in BACKGROUND + URGENT) or bool(re.search(r'\d+\s*岁',text))


def source_version(state):
    return state.get('context_version') or digest(json.dumps(state.get('sources', {}), sort_keys=True, ensure_ascii=False))


def fact_relevant(fact, task=None, scope=None):
    allowed = normalized_scope(scope)
    tooth = str(fact.get('tooth') or '').removesuffix('牙')
    if allowed and tooth and tooth not in allowed:
        return False
    if allowed and tooth in allowed:
        return True
    tags = task_tags(fact)
    return not task or task in tags or not tags or background(fact)


def select_evidence(*, task=None, text='', sources=None, facts=(), scope=(), options=None):
    """Select verbatim contiguous windows; retain canonical origin offsets and hashes."""
    cfg = {**context_options(), **(options or {})}
    counter = TokenCounter()
    source_map = sources or {'record': {'text': text, 'kind': 'clinical_record'}}
    allowed = normalized_scope(scope)
    relevant = [f for f in facts if fact_relevant(f, task, scope)]
    candidates = []
    filtered = 0
    for sid, source in source_map.items():
        value = source.get('text', '')
        if not isinstance(value, str) or not value:
            continue
        source_facts = [f for f in relevant if f.get('source_id') == sid]
        spans = []
        for match in re.finditer(r'[^\n。！？!?]+[。！？!?]?|\n', value):
            if not match.group().strip():
                continue
            start, end = match.span()
            spans.append((start, end))
        # Split oversized sentences only at clause boundaries, never inside words or numbers.
        clauses = []
        for start, end in spans:
            if end - start <= cfg['evidence_window_chars']:
                clauses.append((start, end))
            else:
                for match in re.finditer(r'[^；;]+[；;]?', value[start:end]):
                    clauses.append((start + match.start(), start + match.end()))
        spans = clauses
        # A quoted fact may cross sentence boundaries. Its complete quote is atomic.
        for fact in source_facts:
            quote = fact.get('quote', '')
            if quote and quote in value:
                start = value.index(quote)
                spans.append((start, start + len(quote)))
        unique = set()
        for start, end in spans:
            snippet = value[start:end]
            teeth = set(TEETH.findall(snippet))
            if allowed and teeth and not teeth.intersection(allowed) and not background(snippet):
                filtered += 1
                continue
            tags = task_tags(snippet)
            matched = [f for f in source_facts if f.get('quote') and f['quote'] in snippet]
            if task and tags and task not in tags and not matched and not background(snippet) and not allowed.intersection(teeth):
                filtered += 1
                continue
            key = (start, end)
            if key in unique:
                continue
            unique.add(key)
            protected = background(snippet) or any(f.get('conflicts_with') or f.get('status') == 'conflict' for f in matched)
            score = 100 if protected else 40 if matched else 20 if task in tags else 10 if allowed.intersection(teeth) else 1
            candidates.append((score, len(candidates), sid, start, end, source))
    candidates.sort(key=lambda row: (-row[0], row[1]))
    selected = {}
    used = 0
    omitted = 0
    omitted_protected = False
    for score, _, sid, start, end, source in candidates:
        # Do not send the same span twice when a whole-quote window contains a sentence.
        if any(item['selection_source_id'] == sid and item['selection_start'] <= start and item['selection_end'] >= end for item in selected.values()):
            continue
        snippet = source['text'][start:end]
        key = sid + '@' + str(start) + ':' + str(end)
        origin_id = source.get('origin_source_id', sid)
        origin_start = source.get('origin_start', 0) + start
        item = {'text': snippet, 'kind': source.get('kind', 'clinical_record'),
            'origin_source_id': origin_id, 'origin_start': origin_start, 'origin_end': origin_start + len(snippet),
            'origin_sha256': source.get('origin_sha256', digest(source['text'])),
            'selection_source_id': sid, 'selection_start': start, 'selection_end': end}
        size = counter.count({key: item})
        if len(selected) >= cfg['evidence_items'] or used + size > cfg['specialty_tokens']:
            omitted += 1
            omitted_protected = omitted_protected or score >= 100
            continue
        selected[key] = item
        used += size
    if omitted_protected:
        raise ContextBudgetExceeded('protected_evidence_exceeds_specialty_budget')
    if not selected:
        raise ValueError('no_relevant_source_evidence')
    # Restore source order after relevance selection; do not reorder clinical chronology.
    selected = dict(sorted(selected.items(), key=lambda pair: (list(source_map).index(pair[1]['selection_source_id']), pair[1]['selection_start'])))
    cursor = 0
    for source in selected.values():
        source['selected_text_start'] = cursor
        source['selected_text_end'] = cursor + len(source['text'])
        cursor += len(source['text']) + 1
    return {'sources': selected, 'text': '\n'.join(v['text'] for v in selected.values()),
        'audit': {'task': task, 'scope': sorted(allowed), 'windows': len(selected),
            'source_count': len(source_map), 'filtered_windows': filtered, 'budget_omitted_windows': omitted,
            'estimated_input_tokens': used, 'count_method': counter.method,
            'complete': omitted == 0, 'selection_is_not_clinical_validation': True}}


def filter_facts(facts, task=None, scope=None, limit=None):
    cfg = context_options()
    selected = [f for f in facts if fact_relevant(f, task, scope)]
    maximum = cfg['policy_tokens'] if limit is None else limit
    selected = [f for _,f in sorted(enumerate(selected), key=lambda pair: (
        not bool(pair[1].get('conflicts_with')), not background(pair[1]), -pair[0]))]
    kept = bounded_items(selected, maximum)
    # A dangling conflict ID is a signal, not permission to suppress the conflict.
    return copy.deepcopy(kept), {'selected': len(kept), 'available': len(selected), 'incomplete': len(kept) < len(selected)}


def history_summary(messages, facts, previous=None, *, archived_messages=0):
    cfg = context_options()
    old = (previous or {}).get('items', [])
    current = [{k: f.get(k) for k in ('id', 'name', 'value', 'status', 'tooth', 'quote', 'source_id', 'conflicts_with')} for f in facts]
    merged = {item['id']: item for item in old + current if item.get('id')}
    items = list(merged.values())
    items = [f for _,f in sorted(enumerate(items), key=lambda pair: (
        not bool(pair[1].get('conflicts_with')), not background(pair[1]), -pair[0]))]
    selected = bounded_items(items, max(64, cfg['summary_tokens'] - 256))
    return {'version': 1, 'method': 'extractive_fact_digest', 'items': selected,
        'archived_messages': archived_messages + (previous or {}).get('archived_messages', 0),
        'omitted_items': len(items) - len(selected), 'is_clinical_source': False,
        'policy': 'Historical navigation only; retrieve original sources before specialist analysis.'}


def compact_session(state):
    """Bound active state. SessionStore archives the complete inputs before this call."""
    cfg = context_options()
    result = copy.deepcopy(state)
    messages = result.get('messages', [])
    removed_messages = messages[:-cfg['active_messages']]
    result['messages'] = messages[-cfg['active_messages']:]
    facts = result.get('facts', [])
    # Retain high-priority sources first, then recency. Other evidence remains retrievable.
    ranked = sorted(enumerate(facts), key=lambda pair: (bool(pair[1].get('conflicts_with')) or background(pair[1]), pair[0]), reverse=True)
    source_ids = []
    kept = []
    for _, fact in ranked:
        sid = fact.get('source_id')
        if sid not in source_ids and len(source_ids) >= cfg['active_sources']:
            continue
        if len(kept) >= cfg['active_facts']:
            break
        if sid not in source_ids:
            source_ids.append(sid)
        kept.append(fact)
    ids = {f['id'] for f in kept}
    result['facts'] = [f for f in facts if f['id'] in ids]
    sources = result.get('sources', {})
    # Preserve recent sources even if this turn extracted no factors.
    for sid in reversed(list(sources)):
        if sid not in source_ids and len(source_ids) < cfg['active_sources']:
            source_ids.append(sid)
    result['sources'] = {sid: sources[sid] for sid in source_ids if sid in sources}
    old_message_ids = {m.get('id') for m in removed_messages if m.get('id')}
    summary_facts = [f for f in facts if f['id'] not in ids or any(
        str(f.get('source_id','')).startswith(mid) for mid in old_message_ids)]
    result['history_summary'] = history_summary(removed_messages, summary_facts,
        result.get('history_summary'), archived_messages=len(removed_messages))
    result['context_revision'] = result.get('context_revision', 0) + 1
    result['context_version'] = digest(source_version(state) + ':' + str(result['context_revision']))
    return result


def policy_context(state, config):
    """A bounded projection. Missing evidence is reported so finish cannot imply coverage."""
    cfg = context_options()
    facts, audit = filter_facts(state['facts'], limit=max(512, cfg['policy_tokens'] // 3))
    from .consult_contracts import visible_messages
    view = {k: state.get(k) for k in ('question_count', 'tool_count', 'urgent', 'asked_fields')}
    view['facts'] = facts
    known = {f['id'] for f in facts}
    view['needs'] = {'candidates': [c for c in state.get('needs', {}).get('candidates', []) if set(c.get('fact_ids', [])).issubset(known)]}
    view['history_summary'] = state.get('history_summary', {})
    view['recent_messages'] = bounded_items(visible_messages(state.get('messages', [])), cfg['policy_tokens'] // 5)
    view['historical_memory'] = bounded_items(state.get('historical_memory', []), cfg['summary_tokens'])
    view['memory_policy'] = 'Historical memory and summaries are not current clinical sources.'
    view['asked_questions'] = state.get('asked_questions', [])[-config['max_questions']:]
    view['experts'] = {task: {k: value for k, value in result.items() if k in {'task', 'status', 'input_version', 'error'}} for task, result in state.get('experts', {}).items()}
    view['knowledge'] = [{'id': c['id'], 'source': c.get('source')} for c in state.get('knowledge', [])]
    view['observations'] = [{k: o[k] for k in ('tool', 'task', 'status', 'error') if k in o} for o in state.get('observations', [])[-config['max_tools']:]]
    from .consult_contracts import REGISTRY
    view['registry'] = [{k: a[k] for k in ('id', 'name', 'scope', 'tools') if k in a} for a in REGISTRY.values()]
    view['limits'] = {k: config[k] for k in ('max_actions', 'max_tools', 'max_questions', 'max_experts', 'max_experts_independent')}
    view['context_audit'] = audit
    return view


def answer_context(state, analyses, knowledge, decision):
    """Summarize validated outputs atomically; never splice quotes or rule assessments."""
    cfg = context_options()
    facts, audit = filter_facts(state['facts'], limit=cfg['policy_tokens'] // 5)
    packet = {'facts': facts, 'leader_analysis': {}, 'knowledge': [], 'decision': decision,
        'history_summary': state.get('history_summary', {})}
    per_task = max(256, cfg['policy_tokens'] // 2 // max(1, len(analyses)))
    incomplete = audit['incomplete']
    for task, result in analyses.items():
        risks = result.get('risk_assessments', [])
        kept = bounded_items(risks, per_task)
        packet['leader_analysis'][task] = {'task': task, 'status': result.get('status'),
            'risk_assessments': kept,
            'missing_information': result.get('specialty_document', {}).get('missing_information', []),
            'context_incomplete': len(kept) != len(risks)}
        incomplete = incomplete or len(kept) != len(risks)
    hits = [{k: hit[k] for k in ('id', 'source', 'text', 'source_url') if k in hit} for hit in knowledge]
    packet['knowledge'] = bounded_items(hits, cfg['policy_tokens'] // 4)
    incomplete = incomplete or len(packet['knowledge']) < len(hits)
    packet['context_audit'] = {'incomplete': incomplete, 'summary_is_not_clinical_source': True}
    return packet
