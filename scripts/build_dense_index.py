import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT
from medagent.dense_retrieval import build_index

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--encoder',required=True);p.add_argument('--output',default='runtime/small_model/retrieval');p.add_argument('--device',choices=['cpu','cuda:0'],default='cpu');a=p.parse_args()
    if a.device!='cpu':
        from medagent.small_training import enforce_devices
        enforce_devices()
    build_index(ROOT/'knowledge',ROOT/a.output,a.encoder,a.device)
