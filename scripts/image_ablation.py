"""Small paired-image feasibility study, explicitly separated from clean text benchmarks."""
import argparse,json,os,sys,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
from medagent.skills import DocumentParseSkill,ReviewSkill,SymptomExtractSkill
from medagent.evaluation import evaluate
from ablation_worker import Runner,capture,Replay,OUT,records
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);args=p.parse_args();os.umask(0o077)
pairs=read_json(OUT/'image_pairs.json');model=read_json(ROOT/'configs/gateway.json')['models']['dental-vision']
with Runner(model,args.gpu,'image_study',vision=True) as run:
    jobs=[{'key':r['id'],**capture(lambda c:DocumentParseSkill(c).run('',[r['image_path']])),'image_sha256':r['image_sha256']} for r in pairs]
    ocr=run.generate(jobs,'image_ocr');review_jobs=[];parsed={}
    for row in pairs:
        try:parsed[row['id']]=DocumentParseSkill(Replay(ocr[row['id']]['answer'])).run('',[row['image_path']])
        except (ValueError,KeyError,TypeError):continue
        review_jobs.append({'key':row['id'],**capture(lambda c:ReviewSkill(c).run(parsed[row['id']]['text'],[row['image_path']])),'image_sha256':row['image_sha256']})
    reviewed=run.generate(review_jobs,'image_review')
# Evidence of OCR and image review only; full image includes diagnosis whereas original
# training/test text deliberately strips it. Comparing downstream risk would leak labels.
results=[]
for row in pairs:
    result={'id':row['id'],'train_overlap':row['train_text_overlap'],'ocr_valid':row['id'] in parsed,'review_valid':False}
    if result['ocr_valid']:
        text=parsed[row['id']]['text'];result.update(ocr_characters=len(text),ocr_issues=parsed[row['id']]['issues'])
        try:
            rr=ReviewSkill(Replay(reviewed[row['id']]['answer'])).run(text,[row['image_path']])
            result.update(review_valid=True,review_changed=rr['text']!=text,review_approved=rr['approved'],review_issues=rr['issues'])
        except (ValueError,KeyError,TypeError):pass
    results.append(result)
write_json(OUT/'image_results.json',{'cases':results,'limitations':['4 image associations; 2 overlap training candidates.','Historical OCR is not an independent manually transcribed gold reference.','Images retain diagnosis sections removed from text test cases; downstream accuracy is not comparable and therefore not reported.','This tests real image ingestion and review feasibility, not an OCR accuracy improvement.']})
