"""SQLite session transactions serialize turns across threads and processes."""
import json
import sqlite3
import time
import uuid
from .common import ROOT

class SessionBusy(ValueError): pass

class SessionStore:
    def __init__(self, path=None):
        self.path = path or ROOT/'runtime/consultation/sessions.sqlite'
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as c:
            c.execute('PRAGMA journal_mode=WAL')
            c.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, data TEXT NOT NULL, active_run TEXT, updated REAL NOT NULL)')
        self.path.chmod(0o600)

    def connect(self): return sqlite3.connect(str(self.path), timeout=10)

    def create(self):
        cid = uuid.uuid4().hex
        state = {'conversation_id':cid,'facts':[],'sources':{},'messages':[],'question_count':0,'asked_questions':[],'asked_fields':[]}
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
