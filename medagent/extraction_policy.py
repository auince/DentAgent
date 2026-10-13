"""Explicit-field extraction candidate fixing a literal-placeholder prompt failure."""
import json
from .common import ROOT
from .rules import assess,PERSON,FIRST,REPEAT
from .skills import BOUNDARY,SymptomExtractSkill,scope_hint,validate_scope

class ExplicitFieldExtractSkill(SymptomExtractSkill):
    def run(self,task,text,correction=None,scope=None):
        if task!='root_canal':return super().run(task,text,correction,scope)
        legacy=(ROOT/'prompts/root_canal_legacy.txt').read_text()
        template={'personLabel':{f:[] for f in PERSON},'toothLabel':[{'position':'原文牙位','isFirst':True,'difficulty':1,**{f:[] for f in dict.fromkeys(FIRST+REPEAT)}}]}
        contract=('仅输出 JSON 对象，字段层级示例：'+json.dumps(template,ensure_ascii=False)+
            '。每个因素字段直接放在 personLabel 或对应 toothLabel 元素中；禁止增加名为“各指标”“指标”“因素”的中间层。'
            '各因素值采用 [{"label":"原规则中的完整枚举","reason":"简短原文依据"}]；未记载、不能判断时用空数组，不能补写正常。'
            '初治患牙保留字段：'+','.join(FIRST)+'；再治患牙保留字段：'+','.join(REPEAT)+'。'
            '不得遗漏、合并不同患牙。只输出 JSON，不输出思考过程。')
        content=json.dumps({'case_text':text,'validation_feedback':correction},ensure_ascii=False)
        raw=self.client.complete(task,BOUNDARY+legacy+'\n'+contract+scope_hint(scope),content)
        from .format_repair import close_containers
        repaired=close_containers(task,raw)
        result=assess(task,repaired['text'])
        if repaired['repaired']:result['format_repair']={k:v for k,v in repaired.items() if k!='text'}
        missing=[f for f in PERSON if f not in result['factors']['personLabel']]
        for t in result['factors']['toothLabel']:
            missing += [t['position']+':'+f for f in (FIRST if t['isFirst'] else REPEAT) if f not in t]
        if missing:raise ValueError('missing_schema_fields:'+','.join(missing))
        return validate_scope(task,result,scope)


class TrainingAlignedExtractSkill(ExplicitFieldExtractSkill):
    """Root SFT retains its validated training protocol; crack retains its own contract."""
    def run(self,task,text,correction=None,scope=None):
        if task!='root_canal':return SymptomExtractSkill.run(self,task,text,correction,scope)
        prompt=(ROOT/'prompts/root_canal_legacy.txt').read_text()+'\n本次任务是 /json。只输出包含 personLabel 和 toothLabel 的 JSON 对象，不输出分析过程。'
        content=text
        if '<reference>' not in content:
            content='请根据病人病例，生成json的格式的评价标签/json。病人病例如下:<reference>'+text+'</reference>'
        if correction:content+='\n结构校验反馈：'+correction+'。仅修正输出格式和字段，不得补造病历事实。'
        if correction and correction.startswith('invalid_taxonomy:'):
            from .common import read_json
            field=correction.split(':',2)[1]
            legal=read_json(ROOT/'schemas/root_taxonomy.json').get(field)
            if legal:content+='\n'+field+'只能选择以下合法枚举或空数组：'+json.dumps(legal,ensure_ascii=False)
        content+=scope_hint(scope)
        raw=self.client.complete(task,prompt,content)
        from .format_repair import close_containers
        repaired=close_containers(task,raw);result=assess(task,repaired['text'])
        missing=[f for f in PERSON if f not in result['factors']['personLabel']]
        for tooth in result['factors']['toothLabel']:
            missing += [tooth['position']+':'+f for f in (FIRST if tooth['isFirst'] else REPEAT) if f not in tooth]
        if missing:raise ValueError('missing_schema_fields:'+','.join(missing))
        if repaired['repaired']:result['format_repair']={k:v for k,v in repaired.items() if k!='text'}
        return validate_scope(task,result,scope)
