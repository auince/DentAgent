import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from medagent.common import ROOT,read_json,write_json
from medagent.training_data import prepare_legacy

def main():
    cards,audit=prepare_legacy(read_json(ROOT/'configs/experiments.json'),ROOT/'runtime/training_data')
    out=ROOT/'runtime/small_model/data'
    write_json(out/'legacy_cards.json',cards);write_json(out/'legacy_audit.json',audit)
    for split in ['train','validation']:
        rows=[]
        for c in cards:
            if c['split']!=split:continue
            row={**c['prepared_row'],'group_id':c['group_id'],'provenance':{k:c[k] for k in ('source','source_sha256','source_identity','source_flag_isReal','label_version')},'validation':'legacy_rule_schema'}
            rows.append(row)
        write_json(out/'legacy_specialist'/(split+'.json'),rows)
    print(audit['counts'])

if __name__=='__main__':main()
