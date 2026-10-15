"""SQLite session transactions serialize turns across threads and processes."""
from .settings import configured_path
import json
import sqlite3
import time
import uuid
from .common import ROOT

class SessionBusy(ValueError): pass

class SessionStore:
    def __init__(self, path=None):
        self.path = path or configured_path('sessions_db')
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as c:
            c.execute('PRAGMA journal_mode=WAL')
            c.execute('CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, data TEXT NOT NULL, updated REAL NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, data TEXT NOT NULL, active_run TEXT, updated REAL NOT NULL)')
            c.execute('CREATE TABLE IF NOT EXISTS context_sources (cid TEXT NOT NULL, sid TEXT NOT NULL, body TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1, updated REAL NOT NULL, PRIMARY KEY(cid,sid))')
            c.execute('CREATE TABLE IF NOT EXISTS context_facts (cid TEXT NOT NULL, fid TEXT NOT NULL, sid TEXT NOT NULL, tooth TEXT NOT NULL, tags TEXT NOT NULL, background INTEGER NOT NULL, body TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1, updated REAL NOT NULL, PRIMARY KEY(cid,fid))')
            c.execute('CREATE INDEX IF NOT EXISTS context_fact_lookup ON context_facts(cid,active,background,updated)')
            c.execute('CREATE TABLE IF NOT EXISTS context_messages (cid TEXT NOT NULL, mid TEXT NOT NULL, body TEXT NOT NULL, created REAL NOT NULL, PRIMARY KEY(cid,mid))')
            c.execute('CREATE TABLE IF NOT EXISTS context_source_index (cid TEXT NOT NULL, sid TEXT NOT NULL, tags TEXT NOT NULL, teeth TEXT NOT NULL, background INTEGER NOT NULL, updated REAL NOT NULL, PRIMARY KEY(cid,sid))')
            c.execute('CREATE INDEX IF NOT EXISTS source_context_lookup ON context_source_index(cid,background,updated)')
        self.path.chmod(0o600)

    def connect(self): return sqlite3.connect(str(self.path), timeout=10)

    def create(self,patient_id=None):
        import re
        if patient_id is not None and (not isinstance(patient_id,str) or not re.fullmatch(r'[a-f0-9]{32}',patient_id)):
            raise ValueError('invalid_patient_memory_id')
        if patient_id is not None:
            self.profile(patient_id)
        else:
            patient_id=uuid.uuid4().hex
            with self.connect() as c:c.execute('INSERT INTO profiles VALUES (?,?,?)',(patient_id,'[]',time.time()))
        cid = uuid.uuid4().hex
        state = {'conversation_id':cid,'patient_id':patient_id,'historical_memory':self.profile(patient_id),'facts':[],'sources':{},'messages':[],'question_count':0,'asked_questions':[],'asked_fields':[]}
        with self.connect() as c:
            c.execute('INSERT INTO sessions VALUES (?,?,NULL,?)',(cid,json.dumps(state,ensure_ascii=False),time.time()))
        return state

    def get(self,cid):
        with self.connect() as c:
            row = c.execute('SELECT data,active_run FROM sessions WHERE id=?',(cid,)).fetchone()
        if row is None: raise KeyError('conversation_not_found')
        return {**json.loads(row[0]),'active_run':row[1]}

    def reserve(self,cid,rid):
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row=c.execute('SELECT data,active_run FROM sessions WHERE id=?',(cid,)).fetchone()
            if row is None: raise KeyError('conversation_not_found')
            if row[1]: raise SessionBusy('conversation_has_active_run')
            c.execute('UPDATE sessions SET active_run=?,updated=? WHERE id=?',(rid,time.time(),cid))
        return json.loads(row[0])

    def _archive(self, c, cid, state):
        from .context import background, task_tags, TEETH
        from .common import digest
        now = time.time()
        for sid, source in state.get('sources', {}).items():
            c.execute('INSERT INTO context_sources VALUES (?,?,?,1,?) ON CONFLICT(cid,sid) DO UPDATE SET body=excluded.body, active=1',
                (cid, sid, json.dumps(source, ensure_ascii=False), source.get('time', now)))
            value=source.get('text','')
            c.execute('INSERT INTO context_source_index VALUES (?,?,?,?,?,?) ON CONFLICT(cid,sid) DO UPDATE SET tags=excluded.tags,teeth=excluded.teeth,background=excluded.background',
                (cid,sid,'|'+'|'.join(task_tags(value))+'|','|'+'|'.join(sorted(set(TEETH.findall(value))))+'|',int(background(value)),source.get('time',now)))
            for replaced in source.get('replaces', []):
                c.execute('UPDATE context_sources SET active=0 WHERE cid=? AND sid=?', (cid, replaced))
                c.execute('UPDATE context_facts SET active=0 WHERE cid=? AND sid=?', (cid, replaced))
        for fact in state.get('facts', []):
            source = state.get('sources', {}).get(fact['source_id'], {})
            c.execute('INSERT INTO context_facts VALUES (?,?,?,?,?,?,?,1,?) ON CONFLICT(cid,fid) DO UPDATE SET body=excluded.body',
                (cid, fact['id'], fact['source_id'], str(fact.get('tooth') or '').removesuffix('牙'),
                 '|' + '|'.join(task_tags(fact)) + '|', int(background(fact)),
                 json.dumps(fact, ensure_ascii=False), source.get('time', now)))
        for message in state.get('messages', []):
            raw = json.dumps(message, ensure_ascii=False, sort_keys=True)
            mid = message.get('id') or message.get('run_id') or digest(raw)
            c.execute('INSERT OR IGNORE INTO context_messages VALUES (?,?,?,?)', (cid, mid, raw, now))

    def finish(self,cid,rid,state):
        from .context import compact_session
        compact = compact_session(state)
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row = c.execute('SELECT active_run FROM sessions WHERE id=?', (cid,)).fetchone()
            if row is None or row[0] != rid:
                raise SessionBusy('session_run_owner_mismatch')
            self._archive(c, cid, state)
            c.execute('UPDATE sessions SET data=?,active_run=NULL,updated=? WHERE id=? AND active_run=?',
                (json.dumps(compact,ensure_ascii=False),time.time(),cid,rid))

    def evidence(self, cid, task, scope=None):
        """Read a bounded, specialty/tooth-filtered archive view for this conversation only."""
        from .context import normalized_scope
        from .token_budget import context_options
        clauses = ["cid=?", "active=1"]
        params = [cid]
        if task:
            clauses.append("(background=1 OR tags='||' OR tags LIKE ?)")
            params.append('%|' + task + '|%')
        teeth = sorted(normalized_scope(scope))
        if teeth:
            clauses.append("(background=1 OR tooth='' OR tooth IN (" + ','.join('?' for _ in teeth) + '))')
            params.extend(teeth)
        query = ('SELECT body,sid FROM context_facts WHERE ' + ' AND '.join(clauses) +
            ' ORDER BY background DESC,updated DESC LIMIT ?')
        params.append(context_options()['archive_candidates'])
        with self.connect() as c:
            rows = c.execute(query, params).fetchall()
            facts = [json.loads(row[0]) for row in rows]
            sources = {}
            for sid in dict.fromkeys(row[1] for row in rows):
                source = c.execute('SELECT body FROM context_sources WHERE cid=? AND sid=? AND active=1', (cid, sid)).fetchone()
                if source:
                    sources[sid] = json.loads(source[0])
            fact_total=c.execute('SELECT COUNT(*) FROM context_facts WHERE '+' AND '.join(clauses),params[:-1]).fetchone()[0]
            raw_clauses=["s.cid=?","s.active=1"]
            raw_params=[cid]
            if task:
                raw_clauses.append("(x.background=1 OR x.tags='||' OR x.tags LIKE ?)")
                raw_params.append('%|'+task+'|%')
            if teeth:
                raw_clauses.append("(x.background=1 OR x.teeth='||' OR "+' OR '.join('x.teeth LIKE ?' for _ in teeth)+')')
                raw_params.extend('%|'+tooth+'|%' for tooth in teeth)
            join=' FROM context_sources s JOIN context_source_index x ON s.cid=x.cid AND s.sid=x.sid WHERE '+' AND '.join(raw_clauses)
            source_total=c.execute('SELECT COUNT(*)'+join,raw_params).fetchone()[0]
            raw_rows=c.execute('SELECT s.sid,s.body'+join+' ORDER BY x.background DESC,x.updated DESC LIMIT ?',raw_params+[context_options()['archive_candidates']]).fetchall()
            omitted_sources=0
            for sid,body in raw_rows:
                if sid not in sources and len(sources)>=context_options()['archive_candidates']:
                    omitted_sources+=1
                    continue
                sources[sid]=json.loads(body)
        return {'facts': [f for f in facts if f['source_id'] in sources], 'sources': sources,
            'archive_audit':{'facts_truncated':fact_total>len(rows),
                'sources_truncated':source_total>len(raw_rows) or omitted_sources>0,
                'candidate_limit':context_options()['archive_candidates']}}

    def release(self,cid,rid):
        with self.connect() as c:
            c.execute('UPDATE sessions SET active_run=NULL,updated=? WHERE id=? AND active_run=?',(time.time(),cid,rid))


    def profile(self,patient_id):
        with self.connect() as c:
            row=c.execute('SELECT data FROM profiles WHERE id=?',(patient_id,)).fetchone()
        if row is None:raise KeyError('patient_memory_not_found')
        return json.loads(row[0])

    def remember(self,cid,fact_ids):
        if not isinstance(fact_ids,list) or not fact_ids or any(not isinstance(i,str) for i in fact_ids):raise ValueError('explicit_fact_selection_required')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row=c.execute('SELECT data,active_run FROM sessions WHERE id=?',(cid,)).fetchone()
            if row is None:raise KeyError('conversation_not_found')
            if row[1]:raise SessionBusy('conversation_has_active_run')
            state=json.loads(row[0]);known={f['id']:f for f in state['facts']}
            for fid in fact_ids:
                if fid not in known:
                    archived=c.execute('SELECT body FROM context_facts WHERE cid=? AND fid=? AND active=1',(cid,fid)).fetchone()
                    if archived:known[fid]=json.loads(archived[0])
            if any(i not in known for i in fact_ids):raise ValueError('unknown_memory_fact')
            old=c.execute('SELECT data FROM profiles WHERE id=?',(state['patient_id'],)).fetchone()
            memory={x['id']:x for x in json.loads(old[0])}
            for fid in set(fact_ids):
                fact=known[fid];source=state['sources'].get(fact['source_id'])
                if not source:
                    archived=c.execute('SELECT body FROM context_sources WHERE cid=? AND sid=? AND active=1',(cid,fact['source_id'])).fetchone()
                    source=json.loads(archived[0]) if archived else None
                if not source or fact['quote'] not in source['text'] or source.get('kind')!='patient_report':raise ValueError('memory_requires_patient_source')
                if fact.get('status')=='conflict' or fact.get('conflicts_with'):raise ValueError('conflicting_memory_requires_clarification')
                memory[fid]={'id':fid,'name':fact['name'],'value':fact.get('value'),
                    'status':fact['status'],'quote':fact['quote'],'source_id':fact['source_id'],
                    'source_conversation_id':cid,'patient_confirmed_at':time.time(),
                    'scope':'historical_context_not_current_clinical_fact'}
            if len(memory)>64:raise ValueError('memory_budget_exceeded')
            data=json.dumps(list(memory.values()),ensure_ascii=False)
            c.execute('UPDATE profiles SET data=?,updated=? WHERE id=?',(data,time.time(),state['patient_id']))
        return {'patient_id':state['patient_id'],'memory':list(memory.values()),'saved_only_explicitly_selected_facts':True}

    def forget(self,cid):
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row=c.execute('SELECT data FROM sessions WHERE id=?',(cid,)).fetchone()
            if row is None:raise KeyError('conversation_not_found')
            patient_id=json.loads(row[0])['patient_id'];related=[]
            for sid,data,active in c.execute('SELECT id,data,active_run FROM sessions'):
                state=json.loads(data)
                if state.get('patient_id')==patient_id:
                    if active:raise SessionBusy('patient_memory_in_active_consultation')
                    state['historical_memory']=[];related.append((json.dumps(state,ensure_ascii=False),time.time(),sid))
            c.execute('UPDATE profiles SET data=?,updated=? WHERE id=?',('[]',time.time(),patient_id))
            c.executemany('UPDATE sessions SET data=?,updated=? WHERE id=?',related)
        return {'patient_id':patient_id,'memory':[],'conversation_history_deleted':False}
