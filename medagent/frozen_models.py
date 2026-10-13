"""Verify that RL tools serve the fixed checkpoints named by the validation gate."""
import json
import os
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
from .common import ROOT,read_json,digest

def checkpoint_fingerprint(path):
    import hashlib
    path=Path(path);path=path if path.is_absolute() else ROOT/path
    files=sorted(path.glob('*.safetensors'))
    if not files:raise ValueError('frozen_checkpoint_weights_missing')
    hashes={}
    for file in files:
        h=hashlib.sha256()
        with file.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        hashes[file.name]=h.hexdigest()
    return digest(json.dumps(hashes,sort_keys=True))

def validate_frozen_bindings(frozen):
    if not frozen.get('validated'):raise ValueError('frozen_tools_not_evaluated')
    bindings=frozen.get('models',{})
    fingerprints={}
    for role in ('specialist','needs'):
        binding=bindings.get(role)
        if not isinstance(binding,dict):raise ValueError('frozen_binding_missing:'+role)
        endpoint=binding.get('endpoint','')
        if urlsplit(endpoint).hostname not in {'127.0.0.1','localhost','::1'}:raise ValueError('frozen_endpoint_not_local')
        path=Path(binding['checkpoint']);path=path if path.is_absolute() else ROOT/path
        if str(path) not in fingerprints:fingerprints[str(path)]=checkpoint_fingerprint(path)
        if fingerprints[str(path)]!=binding.get('checkpoint_sha256'):raise ValueError('frozen_weights_changed:'+role)
        if binding.get('checkpoint_kind')=='offline_teacher_base':
            if not frozen.get('offline_teacher_only') or not (path/'config.json').exists():raise ValueError('invalid_offline_teacher_binding')
        elif not (path.parent/'complete.json').exists():raise ValueError('frozen_checkpoint_incomplete:'+role)
        headers={};key=os.environ.get(binding.get('api_key_env',''))
        if key:headers['Authorization']='Bearer '+key
        req=urllib.request.Request(endpoint.rstrip('/')+'/models',headers=headers)
        with urllib.request.urlopen(req,timeout=10) as response:models=json.load(response)['data']
        served=next((m for m in models if m['id']==binding['model']),None)
        if served is None or Path(served.get('root','')).resolve()!=path.resolve():raise ValueError('served_frozen_checkpoint_mismatch:'+role)
    return bindings
