"""Version official reference pages locally with URL, timestamps and content hashes."""
import json
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,digest,write_json

SOURCES={
 'caries_sensitive':['https://www.nidcr.nih.gov/health-info/tooth-decay'],
 'periodontal':['https://www.nidcr.nih.gov/health-info/gum-disease'],
 'pulp_periapical':['https://www.nhs.uk/conditions/dental-abscess/'],
 'wisdom':['https://www.nhs.uk/conditions/wisdom-tooth-removal/'],
 'mucosa':['https://www.nhs.uk/conditions/mouth-ulcers/'],
 'tmj':['https://www.nidcr.nih.gov/health-info/tmd'],
 'oral_health':['https://www.nidcr.nih.gov/health-info/dry-mouth','https://www.nhs.uk/conditions/bad-breath/'],
 'trauma':['https://www.nhs.uk/conditions/knocked-out-tooth/'],
 'root_canal':['https://www.nhs.uk/conditions/root-canal-treatment/'],
 'crack':['https://www.nhs.uk/conditions/chipped-broken-or-cracked-tooth/']}

class Text(HTMLParser):
    def __init__(self):super().__init__();self.parts=[];self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in {'script','style','nav','header','footer'}:self.skip+=1
    def handle_endtag(self,tag):
        if tag in {'script','style','nav','header','footer'}:self.skip=max(0,self.skip-1)
        if tag in {'p','h1','h2','h3','li','tr'} and not self.skip:self.parts.append('\n')
    def handle_data(self,text):
        if not self.skip:self.parts.append(text)

def main():
    records=[]
    for task,urls in SOURCES.items():
        for url in urls:
            try:
                request=urllib.request.Request(url,headers={'User-Agent':'MedAgent research reference import/1.0'})
                raw=urllib.request.urlopen(request,timeout=40).read().decode('utf8')
                parser=Text();parser.feed(raw)
                text=re.sub(r'[ \t]+',' ',' '.join(parser.parts))
                text=re.sub(r'\n\s*\n','\n',text).strip()
                if len(text)<400:raise ValueError('reference_too_short')
                record={'task':task,'url':url,'sha256':digest(text),'retrieved_at':time.time(),'status':'imported','language':'en'}
                target=ROOT/'knowledge'/task/(digest(url)[:12]+'.md');target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text('Source: '+url+'\nContent SHA256: '+record['sha256']+'\n\n'+text)
                record['path']=str(target.relative_to(ROOT));records.append(record)
            except Exception as exc:records.append({'task':task,'url':url,'status':'failed','error':str(exc)[:200]})
        write_json(ROOT/'runtime/small_model/knowledge_manifest.json',records)
    print(json.dumps({'imported':sum(x['status']=='imported' for x in records),'failed':sum(x['status']=='failed' for x in records)}))
    if any(x['status']=='failed' for x in records):raise SystemExit(1)

if __name__=='__main__':main()
