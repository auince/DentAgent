"""No background jobs: every model request is initiated by an explicit command."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
selector=argparse.ArgumentParser(add_help=False);selector.add_argument('--config')
selected,remaining=selector.parse_known_args()
from medagent.settings import configure
configure(selected.config)
sys.argv=[sys.argv[0]]+remaining
from medagent.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request', help='UTF-8 JSON 请求文件')
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    result = Pipeline().run(request)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
