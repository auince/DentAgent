"""Report only completed experiments; absent measurements remain null."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json

def main():
    protocol=read_json(ROOT/'configs/small_ablations.json');folder=ROOT/'runtime/small_model/evaluation'
    measurements={p.parent.name:read_json(p) for p in folder.glob('*/results.json')}
    comparisons=[]
    for pair in protocol['comparisons']:
        available=pair['control'] in measurements and pair['treatment'] in measurements
        comparisons.append({**pair,'status':'measured' if available else 'pending','control_metrics':measurements.get(pair['control']),'treatment_metrics':measurements.get(pair['treatment'])})
    report={'protocol':protocol,'available_measurements':measurements,'comparisons':comparisons,'acceptance_verified':False}
    write_json(ROOT/'runtime/small_model/experiment_report.json',report)
    lines=['# 小模型训练与消融进度','', '这里只展示已完成的真实评测。待运行实验没有填入估算分数。','', '| 对照实验 | 状态 |','|---|---|']
    for p in comparisons:lines.append('| '+p['id']+' | '+p['status']+' |')
    (ROOT/'docs/small_model_experiments.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
