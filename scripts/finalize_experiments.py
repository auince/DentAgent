"""One-shot report finalizer for the experiments already launched in this session."""
import sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
out=ROOT/'runtime/ablation'
while True:
 state=read_json(ROOT/'runtime/evaluation/status.json')
 if state['status']=='failed':raise RuntimeError('comparison_failed:'+str(state))
 if state['status']=='completed' and (out/'pipeline_v2_results.json').exists():break
 write_json(out/'report_status.json',{'status':'waiting','comparison':state['status'],'followup_ready':(out/'pipeline_v2_results.json').exists(),'updated_at':time.time()})
 time.sleep(15)
for script in ['summarize_comparison.py','render_comparison.py','summarize_ablations.py','retrieval_ablation.py','summarize_followup.py']:
 subprocess.run([sys.executable,str(ROOT/'scripts'/script)],check=True)
write_json(out/'report_status.json',{'status':'completed','updated_at':time.time()})
