"""Shared source-based urgent referral gate for serving and trajectory environments."""
import re

RULE_VERSION='nhs-oral-triage-phrases-v3-colloquial-swallowing'
SOURCE='https://www.nhs.uk/conditions/dental-abscess/'
DIFFICULTY=re.compile(r'呼吸[^，。；！？\n]{0,12}(?:困难|费力|费劲|吃力)|(?:吞咽|吞口水|咽口水|吞唾液|咽唾液)[^，。；！？\n]{0,12}(?:困难|费力|费劲|吃力|很难(?!受))|喘不过气|(?:吞|咽)不下(?:口水|唾液)')


def urgent_fact_ids(facts, source_prefix=None):
    matches=[]
    for fact in facts:
        if source_prefix and not fact.get('source_id','').startswith(source_prefix):continue
        if fact.get('status')!='positive':continue
        quote=fact.get('quote','')
        for match in DIFFICULTY.finditer(quote):
            before=re.split(r'[，。；！？\n]|但是|但|不过',quote[:match.start()])[-1][-8:]
            if not re.search(r'没有|无|否认|不伴|未出现|不会|如果|万一|担心|会不会',before) and not re.search(r'没有|没觉得|不困难|不费劲|正常',match.group()):
                matches.append(fact['id']);break
    return matches
