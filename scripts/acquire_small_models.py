"""Download pinned public model snapshots; contains no patient data."""
import json
import sys
import time
import hashlib
import urllib.request
from urllib.parse import quote
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,write_json,read_json

def main():
    folder=ROOT/'runtime/small_model/models';folder.mkdir(parents=True,exist_ok=True)
    manifest=read_json(folder/'manifest.json') if (folder/'manifest.json').exists() else {}
    for name in ['Qwen3.5-2B','Qwen3.5-4B']:
        repo='Qwen/'+name
        target=folder/name
        api='https://modelscope.cn/api/v1/models/'+repo+'/repo/files?Revision=master&Recursive=true'
        if manifest.get(name,{}).get('file_manifest'):
            entries=manifest[name]['file_manifest']
        else:
            with urllib.request.urlopen(api,timeout=30) as response: listing=json.load(response)
            entries=listing['Data']['Files']
        files=[f for f in entries if f['Type']=='blob' and (f['Path'].endswith(('.json','.safetensors','.jinja','.txt')) or f['Path']=='LICENSE')]
        target.mkdir(parents=True,exist_ok=True)
        manifest[name]={'repo':repo,'mirror':'modelscope.cn/Qwen','file_manifest':files,'path':str(target),'status':'downloading','updated':time.time()}
        write_json(folder/'manifest.json',manifest)
        for f in files:
            path=target/f['Path'];path.parent.mkdir(parents=True,exist_ok=True)
            expected=f.get('Sha256')
            if not expected:raise ValueError('mirror_missing_sha256:'+f['Path'])
            def sha(p):
                h=hashlib.sha256()
                with p.open('rb') as stream:
                    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
                return h.hexdigest()
            if path.exists() and sha(path)==expected:continue
            url='https://modelscope.cn/api/v1/models/'+repo+'/repo?Revision='+quote(f['Revision'])+'&FilePath='+quote(f['Path'])
            tmp=path.with_suffix(path.suffix+'.partial')
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url,timeout=120) as response,tmp.open('wb') as out:
                        while block:=response.read(8*1024*1024):out.write(block)
                    if sha(tmp)!=expected:raise ValueError('download_checksum_mismatch')
                    tmp.replace(path);break
                except Exception:
                    if attempt==3:raise
                    time.sleep(5)
            print(name,f['Path'],'verified',flush=True)
        manifest[name]['status']='downloaded';write_json(folder/'manifest.json',manifest)
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
