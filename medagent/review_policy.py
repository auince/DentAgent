"""Candidate conservative review policy, evaluated separately from the original pipeline."""
import re
from .common import parse_json
from .skills import BOUNDARY,ReviewSkill


def clinical_text(text):
    """Remove legacy task wrappers only; retain case characters and whitespace."""
    value=text.strip()
    if value.startswith('<reference>') and '</reference>' in value:
        value=value[len('<reference>'):].split('</reference>',1)[0].strip()
    return re.sub(r'^分析此病例的牙隐裂分级\s*[:：]\s*','',value)


class ConservativeReviewSkill:
    name='information_review'
    def __init__(self,client):self.client=client
    def run(self,text,images=()):
        if images:return self.image_review(text,images)
        # With no source image, the model cannot verify or correct transcription.
        raw=parse_json(self.client.complete('review',BOUNDARY+
            '你仅检查同一病历文本内部的显式矛盾，不诊断疾病、不评估信息是否齐全。'
            '未记载、未知、缺少检查不是矛盾；主观无不适与客观检查阳性不自动构成矛盾；'
            '不同牙位或不同时间的描述不视为矛盾。无图时不得声称看到了图像。'
            '不得改写原文。每个问题必须给出原文中两段相互矛盾的逐字引文。'
            '输出 {"issues":[{"code":"text_conflict","quote_a":"原文片段",'
            '"quote_b":"另一原文片段","detail":"为何同一牙位、时间的同一事实自相矛盾"}]}。'
            '没有这样的矛盾时输出 {"issues":[]}。',clinical_text(text)))
        if not isinstance(raw,dict) or not isinstance(raw.get('issues'),list):raise ValueError('invalid_review_result')
        accepted=[];rejected=[]
        for issue in raw['issues']:
            if not isinstance(issue,dict):rejected.append({'reason':'invalid_issue'});continue
            a,b=issue.get('quote_a'),issue.get('quote_b')
            valid=issue.get('code')=='text_conflict' and isinstance(a,str) and isinstance(b,str) and a.strip() and b.strip() and a!=b and a in text and b in text
            if valid:accepted.append({**issue,'start_a':text.index(a),'start_b':text.index(b)})
            else:rejected.append({**issue,'reason':'no_two_distinct_source_quotes'})
        return {'text':text,'approved':not accepted,'issues':accepted,'rejected_suggestions':rejected,
            'policy':'conservative_text_v1','note':'Quotes prove source location, not the medical correctness of a conflict judgement.'}

    def image_review(self,text,images):
        raw=parse_json(self.client.complete('review',BOUNDARY+
            '对照图像审校 OCR 文本，只更正明确看错的牙位、否定词、数字或检查符号。'
            '不要全文重写，不要调整标点、空格、栏目或同义词。无法确认的字符列入 issues。'
            '输出 {"corrections":[{"before":"OCR中确实存在的完整短句","after":"图像对应短句","reason":"原因"}],'
            '"issues":[{"code":"uncertain_image_text","detail":"不确定位置"}]}。'
            '不需要更正时 corrections 为空数组；每个 before 必须唯一出现在输入原文。',text,images))
        if not isinstance(raw,dict) or not isinstance(raw.get('corrections'),list) or not isinstance(raw.get('issues'),list):
            raise ValueError('invalid_image_review_schema')
        changes=[];issues=list(raw['issues'])
        for edit in raw['corrections']:
            before=edit.get('before') if isinstance(edit,dict) else None
            after=edit.get('after') if isinstance(edit,dict) else None
            if not isinstance(before,str) or not before or not isinstance(after,str) or not after or text.count(before)!=1:
                issues.append({'code':'unverifiable_image_correction'});continue
            start=text.index(before);end=start+len(before)
            if any(start<b and end>a for a,b,_ in changes):
                issues.append({'code':'overlapping_image_correction'});continue
            if before!=after:changes.append((start,end,after))
        revised=text
        for start,end,after in sorted(changes,reverse=True):revised=revised[:start]+after+revised[end:]
        return {'text':revised,'approved':not issues,'issues':issues,'corrections':raw['corrections'],
                'applied_corrections':len(changes),'policy':'anchored_image_review_v1'}
