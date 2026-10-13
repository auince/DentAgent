import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.sessions import SessionStore
from medagent.consultation import ConsultationService
parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest='command',required=True)
new=commands.add_parser('new');new.add_argument('--patient-id')
chat=commands.add_parser('send');chat.add_argument('conversation_id');chat.add_argument('text')
remember=commands.add_parser('remember');remember.add_argument('conversation_id');remember.add_argument('fact_ids',nargs='+')
forget=commands.add_parser('forget');forget.add_argument('conversation_id')
args=parser.parse_args();store=SessionStore()
if args.command=='new':result=store.create(args.patient_id)
elif args.command=='send':result=ConsultationService(store=store).run(args.conversation_id,{'text':args.text})
elif args.command=='remember':result=store.remember(args.conversation_id,args.fact_ids)
else:result=store.forget(args.conversation_id)
print(json.dumps(result,ensure_ascii=False,indent=2))
