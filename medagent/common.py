import ast
import hashlib
import json
import os
import re
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(path.suffix + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.chmod(0o600)
    os.replace(tmp, path)


def parse_json(text):
    if not isinstance(text, str):
        return text
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    # Source documents may contain literal reasoning tags inside valid JSON.
    # Strip a model reasoning prefix only after trying the intact document.
    if text.startswith('<think>') and '</think>' in text:
        text = text.split('</think>', 1)[1].strip()
        try:return json.loads(text)
        except ValueError:pass
    # Try the full legacy literal BEFORE scanning nested JSON: otherwise an inner
    # valid list in a Python-formatted case can be mistaken for the entire output.
    try:
        value = ast.literal_eval(text)
        if isinstance(value, (dict, list)):
            return value
    except (ValueError, SyntaxError):
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "[{":
            try:
                return decoder.raw_decode(text[i:])[0]
            except ValueError:
                pass
    # Legacy labels are Python literal dictionaries, never executable code.
    try:
        value = ast.literal_eval(text)
        if isinstance(value, (dict, list)):
            return value
    except (ValueError, SyntaxError):
        pass
    raise ValueError("invalid_structured_output")


def norm(value):
    return "".join(str(value).split())


def option_key(value):
    # Legacy schema and current prompt differ in terminal punctuation only.
    return norm(value).rstrip('。.')


def tooth_key(value):
    """Canonical explicit FDI IDs; keep grouped records distinct from single teeth."""
    value=norm(value)
    ids=re.findall(r'[1-4][1-8]',value)
    rest=re.sub(r'[1-4][1-8]','',value)
    if ids and not re.sub(r'[牙、,，;；和及\[\]【】()]','',rest):
        return '，'.join(x+'牙' for x in sorted(ids))
    return value


def labels(value):
    if isinstance(value, dict):
        return [value["label"]] if isinstance(value.get("label"), str) else []
    if isinstance(value, list):
        return [x for item in value for x in labels(item)]
    if isinstance(value, str):
        return [value]
    return []
