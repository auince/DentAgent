import argparse
import json
from pathlib import Path
from .agents import LeaderAgent

p=argparse.ArgumentParser(description='MedAgent local assessment')
p.add_argument('--request',required=True,help='JSON containing text, images, tasks/instruction')
a=p.parse_args()
result=LeaderAgent().run(json.loads(Path(a.request).read_text()))
print(json.dumps(result,ensure_ascii=False,indent=2))
raise SystemExit(1 if result['status']=='failed' else 0)
