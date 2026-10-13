import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.server import serve
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='启动 MedAgent 后端；不启动模型或训练')
    parser.add_argument('--config',help='统一 JSON 配置文件；默认 configs/service.json')
    serve(parser.parse_args().config)
