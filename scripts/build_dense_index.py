"""Explicit index build; calls only the configured embedding service."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.settings import configure
parser=argparse.ArgumentParser();parser.add_argument('--config');parser.add_argument('--force',action='store_true')
args=parser.parse_args();configure(args.config)
from medagent.retrieval import create_index
index=create_index()
if not hasattr(index,'build'):raise SystemExit('配置 retrieval.mode=hybrid 后才可构建向量索引')
print(json.dumps(index.build(force=args.force),ensure_ascii=False,indent=2))
