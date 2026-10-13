"""Chinese character/bigram BM25 over versioned local documents, with source offsets."""
import collections
import math
import re
from pathlib import Path

from .common import digest


def tokens(text):
    parts = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())
    return parts + [a+b for a, b in zip(parts, parts[1:]) if len(a) == len(b) == 1]


class KnowledgeIndex:
    def __init__(self, directory, chunk_size=800, chunk_stride=600):
        self.chunks = []
        for path in sorted(Path(directory).rglob("*")):
            if path.suffix not in (".md", ".txt"):
                continue
            text = path.read_text(encoding="utf-8")
            version = digest(text)
            source_url=re.search(r'^Source:\s*(https?://\S+)',text,re.M)
            for start in range(0, len(text), chunk_stride):
                chunk = text[start:start+chunk_size]
                self.chunks.append({"id": f"{path.stem}:{start}:{version[:10]}",
                    "source": str(path.relative_to(directory)), "sha256": version,
                    "source_url":source_url.group(1) if source_url else None,
                    "start": start, "end": start+len(chunk), "text": chunk,
                    "tf": collections.Counter(tokens(chunk))})
        self.df = collections.Counter(t for c in self.chunks for t in c["tf"])
        self.avg = sum(sum(c["tf"].values()) for c in self.chunks)/max(len(self.chunks), 1)

    def search(self, query, task=None, limit=4):
        # The imported references are English; explicit topic aliases bridge recall.
        # These are query expansions, never extra patient facts or clinical labels.
        bridges={
            'root_canal':('根管','root canal treatment difficulty'),
            'crack':('咬合 松口 裂纹 牙隐裂','cracked tooth biting pain'),
            'caries_sensitive':('龋 敏感 凉 冷 甜','tooth decay sensitivity'),
            'pulp_periapical':('牙痛 牙疼 自发痛 脓肿','toothache abscess pain'),
            'periodontal':('牙龈 出血 松动','gum disease bleeding'),
            'wisdom':('智齿 冠周','wisdom tooth pain'),
            'mucosa':('溃疡 黏膜','mouth ulcer'),
            'tmj':('关节 张口 耳前','temporomandibular jaw pain'),
            'oral_health':('口干 口臭 清洁','dry mouth oral hygiene'),
            'trauma':('外伤 撞 磕 摔','broken chipped tooth injury'),
        }
        expansions=[english for key,(chinese,english) in bridges.items()
            if key==task or (task is None and any(word in query for word in chinese.split()))]
        query=query+' '+' '.join(expansions)
        terms = set(tokens(query))
        hits = []
        for c in self.chunks:
            if task and not c["source"].startswith(task + "/"):
                continue
            size = sum(c["tf"].values())
            score = sum(math.log(1+(len(self.chunks)-self.df[t]+0.5)/(self.df[t]+0.5)) *
                (c["tf"][t]*2.2)/(c["tf"][t]+1.2*(0.25+0.75*size/max(self.avg,1)))
                for t in terms if c["tf"][t])
            if score > 0:
                hits.append({**{k:v for k,v in c.items() if k != "tf"}, "score": round(score,5)})
        return sorted(hits, key=lambda x: (-x["score"], x["id"]))[:limit]


class RetrievalService:
    def __init__(self,settings):
        self.settings=settings
        cfg=settings.data['retrieval']
        self.index=KnowledgeIndex(settings.path('knowledge_dir'),cfg['chunk_size'],cfg['chunk_stride'])
    def search_with_trace(self,query,task=None,limit=None):
        limit=limit or self.settings.data['retrieval']['top_k']
        hits=self.index.search(query,task,limit)
        return {'results':[dict(hit,retrieval='bm25') for hit in hits],
            'retrieval':{'mode':'bm25','corpus_chunks':len(self.index.chunks),'task':task,'limit':limit}}
    def search(self,query,task=None,limit=None):return self.search_with_trace(query,task,limit)['results']
    def describe(self):return {'mode':'bm25','chunks':len(self.index.chunks),'inference_executed_at_startup':False}

from functools import lru_cache
@lru_cache(maxsize=1)
def create_index():
    from .settings import get_settings
    settings=get_settings()
    if settings.data['retrieval']['mode']=='hybrid':
        from .dense_retrieval import HybridRetrievalService
        cfg=settings.data['retrieval']
        lexical=KnowledgeIndex(settings.path('knowledge_dir'),cfg['chunk_size'],cfg['chunk_stride'])
        return HybridRetrievalService(settings,lexical)
    return RetrievalService(settings)
