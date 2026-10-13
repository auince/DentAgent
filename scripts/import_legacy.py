"""Read-only legacy import: rules, vocabulary, prompts and provenance. No patient files copied."""
import ast
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT, digest, parse_json, labels, write_json
from medagent.rules import PERSON, FIRST, REPEAT

BASE=Path('LOCAL_RESOURCE')


def main():
    sources={
        'root_prompt':BASE/'case_classifyLOCAL_RESOURCE',
        'crack_prompt':BASE/'test_trlLOCAL_RESOURCE',
        'crack_rule':BASE/'toothCrack/dataProcess/20260807/run_llm_pipeline.py',
        'root_labels':BASE/'test_trl/data_998_20260401_review_json.json'}
    manifest={key:{'path':str(p),'sha256':digest(p.read_bytes())} for key,p in sources.items()}
    tree=ast.parse(sources['crack_rule'].read_text())
    constants={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for name in node.targets:
                if isinstance(name,ast.Name) and name.id in ('MODULE_POINTS','MODULE_MAX'):
                    constants[name.id]=ast.literal_eval(node.value)
    write_json(ROOT/'schemas/crack_rules.json',{'version':'crack-20260807',
        'points':constants['MODULE_POINTS'],'caps':constants['MODULE_MAX'],'source':manifest['crack_rule']})
    vocabulary={f:set() for f in set(PERSON+FIRST+REPEAT)}
    data=json.loads(sources['root_labels'].read_text())
    parsed=0
    for item in data:
        try:
            content=[m['content'] for m in item['messages'] if m['role']=='assistant'][-1]
            obj=parse_json(content)
            person=parse_json(obj['personLabel']); teeth=parse_json(obj['toothLabel'])
            for record in [person]+teeth:
                for field in vocabulary:
                    vocabulary[field].update(labels(record.get(field,[])))
            parsed+=1
        except (KeyError,ValueError,TypeError,IndexError): pass
    if not parsed or any(not v for v in vocabulary.values()):
        raise RuntimeError('Cannot extract complete root label field vocabulary')
    write_json(ROOT/'schemas/root_taxonomy.json',{k:sorted(v) for k,v in sorted(vocabulary.items())})
    manifest['root_vocabulary']={'parsed_records':parsed,'total_records':len(data),
        'scope':'observed legacy labels; additions require review against source prompt'}
    for task,key in [('root_canal','root_prompt'),('crack','crack_prompt')]:
        folder=ROOT/'knowledge'/task; folder.mkdir(parents=True,exist_ok=True)
        (folder/'project_rules.md').write_text('# 项目规则参考（非外部临床指南）\n\n来源：'+str(sources[key])+
            '\n版本 SHA256：'+manifest[key]['sha256']+'\n\n'+sources[key].read_text())
    write_json(ROOT/'docs/source_manifest.json',manifest)
    print(json.dumps({'root_vocabulary_records':parsed,'crack_modules':len(constants['MODULE_POINTS']),
                      'patient_data_copied':False},ensure_ascii=False))


if __name__=='__main__': main()
