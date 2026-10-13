"""Secondary scoring with explicit, deterministic legacy field aliases.
No gold values, fuzzy similarity, or language model enters the adapter.
"""
import copy,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json,parse_json,norm,labels
from medagent.rules import PERSON,FIRST,REPEAT
from medagent.evaluation import evaluate
TAX=read_json(ROOT/'schemas/root_taxonomy.json')
def key(s):return re.sub(r'[（(][一二三]级因素[）)]','',norm(s)).rstrip('。.')
def adapt(raw):
 obj=copy.deepcopy(parse_json(raw))
 if not isinstance(obj,dict) or not isinstance(obj.get('toothLabel'),list) or not isinstance(obj.get('personLabel'),dict):raise ValueError('invalid_root_container')
 aliases={'ASA':'全身病史及麻醉','maxAperture':'最大张口度','pharyngealReflex':'咽反射','dentalFear':'牙科恐惧症','oralHygiene':'口腔卫生护理'}
 stats={'unmapped_tags':[],'alias_teeth':0,'factor_aliases':0}
 def canonical(field,value):
  opts=[x for x in TAX.get(field,[]) if key(x)==key(value)]
  if len(opts)==1:
   stats['factor_aliases']+=opts[0]!=value;return opts[0]
  return value
 person={aliases.get(k,k):[{'label':canonical(aliases.get(k,k),v)} for v in labels(val)] for k,val in obj['personLabel'].items()}
 teeth=[]
 for t in obj['toothLabel']:
  if not isinstance(t,dict):raise ValueError('invalid_tooth')
  if 'position' in t:
   for field in set(FIRST+REPEAT):
    if field in t:t[field]=[{'label':canonical(field,v)} for v in labels(t[field])]
   teeth.append(t);continue
  number=t.get('toothNumber')
  if not isinstance(number,(str,int)):raise ValueError('invalid_tooth_number')
  position=str(number).strip()
  if re.fullmatch(r'[1-4][1-8]',position):position+='牙'
  treatment=t.get('treatmentType');first={'初次治疗':True,'再次治疗':False,'根管再治疗':False,'初治':True,'再治':False}.get(treatment)
  stated=[]
  for grade_field in ('difficultyLevel','treatmentDifficulty','assessmentLabel'):
   value=t.get(grade_field)
   if type(value) is int and value in range(1,5):stated.append(value)
   elif isinstance(value,str):
    match=re.fullmatch(r'(?:根管治疗)?(?:难度(?:评估|等级)?[：:]?)?([1-4一二三四ⅠⅡⅢⅣ])级?',norm(value))
    if match:stated.append({'一':1,'二':2,'三':3,'四':4,'Ⅰ':1,'Ⅱ':2,'Ⅲ':3,'Ⅳ':4}.get(match[1],int(match[1]) if match[1].isdigit() else None))
  grade=stated[0] if stated and len(set(stated))==1 else None
  new={'position':position,'isFirst':first,'difficulty':grade};stats['alias_teeth']+=1
  for field in set(FIRST+REPEAT):
   if field in t:new[field]=[{'label':canonical(field,v)} for v in labels(t[field])]
  for source in ('riskFactors','assessmentTags','assessmentLabel'):
   for tag in labels(t.get(source,[])):
    match=re.match(r'^\s*([^:：]+)[:：]\s*(.*)$',tag)
    if match and match[1].strip() in TAX:
     field=match[1].strip();new.setdefault(field,[]).append({'label':canonical(field,match[2])})
    else:
     matches=[(f,x) for f,options in TAX.items() for x in options if key(x)==key(tag)]
     if len(matches)==1 and matches[0][0] not in PERSON:
      field,label=matches[0];new.setdefault(field,[]).append({'label':label});stats['factor_aliases']+=1
     else:stats['unmapped_tags'].append(tag)
  teeth.append(new)
 return {'personLabel':person,'toothLabel':teeth},stats
def main():
 report={};out=ROOT/'runtime/ablation'
 for variant in ('base','finetuned'):
  source=[json.loads(s) for s in (ROOT/'runtime/evaluation'/('root_canal_'+variant+'.jsonl')).read_text().splitlines()]
  rows=[];audits=[]
  for row in source:
   try:prediction,stats=adapt(row['prediction'])
   except (ValueError,KeyError,TypeError):prediction='';stats={'adapter_failed':True}
   rows.append({**row,'prediction':prediction});audits.append({'id':row['id'],**stats})
  report[variant]={'metrics':evaluate('root_canal',rows,False),'audit':audits}
  write_json(out/('root_alias_'+variant+'_predictions.json'),rows)
 report['limitations']=['This is secondary post-hoc protocol adaptation, separate from strict integration pass rate.','Only declared field aliases and unique exact taxonomy matches after removing the level suffix are accepted.','Unmapped free-text factors are reported separately; symptom figures are schema-aligned extraction scores, not full semantic clinical judgement accuracy.','The same adapter is applied to both models without consulting gold labels.']
 write_json(out/'schema_adapter_results.json',report)
 lines=['# 根管输出字段适配审计','','原始基座输出 101/101 未符合既有牙位字段协议，严格集成口径为零，不能解释为模型临床能力为零。本表提供固定字段别名转换后的辅助结果。','', '| 模型 | 症状字段准确率 | 症状 F1 | 直接等级准确率 |','|---|---:|---:|---:|']
 for name in ('base','finetuned'):
  m=report[name]['metrics'];lines.append(f"| {name} | {100*m['symptom_field_exact']['accuracy']:.2f}% | {100*m['symptom_extraction']['micro_f1']:.2f}% | {100*m['grade']['accuracy']:.2f}% |")
 lines+=['','转换 toothNumber → position（单一 FDI 编号增加“牙”）；treatmentType → isFirst；difficultyLevel/treatmentDifficulty → difficulty；患者因素英文字段名转换为现有中文字段。因素只接受唯一、精确的词表映射，不通过医学推断补标签。未能映射的自由描述单独保存在审计中。','','局限：这是查看输出后的适配对照。症状自由描述不能全部转为枚举，因此不能用这些数字代表全面的语义判读能力。另有相同显式 Schema 提示词下的 20 例模型对照。']
 (ROOT/'docs/schema_adapter_results.md').write_text('\n'.join(lines)+'\n')
 print({k:{'symptom':report[k]['metrics']['symptom_field_exact'],'grade':report[k]['metrics']['grade']} for k in ('base','finetuned')})

if __name__=='__main__':main()
