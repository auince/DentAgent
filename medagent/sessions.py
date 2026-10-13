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

    def finish(self,cid,rid,state):
        with self.connect() as c:
            changed=c.execute('UPDATE sessions SET data=?,active_run=NULL,updated=? WHERE id=? AND active_run=?',
                (json.dumps(state,ensure_ascii=False),time.time(),cid,rid)).rowcount
            if changed!=1: raise SessionBusy('session_run_owner_mismatch')

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
            if any(i not in known for i in fact_ids):raise ValueError('unknown_memory_fact')
            old=c.execute('SELECT data FROM profiles WHERE id=?',(state['patient_id'],)).fetchone()
            memory={x['id']:x for x in json.loads(old[0])}
            for fid in set(fact_ids):
                fact=known[fid];source=state['sources'].get(fact['source_id'])
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
