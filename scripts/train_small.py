import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.small_training import train

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-steps',type=int,default=-1);p.add_argument('--resume',action='store_true');a=p.parse_args()
    print(train(a.stage,a.seed,a.max_steps,a.resume))
