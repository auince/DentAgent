"""Conservative syntax-only closure; never adds clinical fields or label values."""
import ast,json
from .rules import normalize_output


def close_containers(task,text):
    if not isinstance(text,str):return {'text':text,'repaired':False}
    value=text.strip()
    if value.startswith('```'):
        lines=value.splitlines()
        if len(lines)>2 and lines[-1].strip()=='```':value='\n'.join(lines[1:-1]).strip()
    if not value or value[0] not in '[{':return {'text':text,'repaired':False}
    stack=[];quote=None;escape=False;pairs={'{':'}','[':']'}
    for ch in value:
        if quote:
            if escape:escape=False
            elif ch=='\\':escape=True
            elif ch==quote:quote=None
            continue
        if ch in ('"',"'"):quote=ch
        elif ch in pairs:stack.append(pairs[ch])
        elif ch in ']}':
            if not stack or stack.pop()!=ch:return {'text':text,'repaired':False}
    if quote or not stack or len(stack)>2:return {'text':text,'repaired':False}
    suffix=''.join(reversed(stack));candidate=value+suffix
    try:
        try:obj=json.loads(candidate)
        except ValueError:obj=ast.literal_eval(candidate)
        normalized=normalize_output(task,obj)
    except (ValueError,SyntaxError,TypeError,KeyError):return {'text':text,'repaired':False}
    return {'text':json.dumps(normalized,ensure_ascii=False),'repaired':True,'added_suffix':suffix,'policy':'container_closure_v1'}
