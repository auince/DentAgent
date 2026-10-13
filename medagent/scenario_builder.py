"""Build the complete frozen consultation suite and its simulated input files."""
import json
from collections import Counter
from .common import ROOT, read_json, write_json, digest

def build_scenarios(data, chart_image):
    paths=sorted((data/'cards/test').glob('*.json'))
    if len(paths)!=1000:raise ValueError('scenario_export_requires_full_1000_cards')
    folder=data/'workflow_scenarios';scenarios=[];counts=Counter()
    for i,path in enumerate(paths):
        card=read_json(path);sid=card['id'];dest=folder/sid;dest.mkdir(parents=True,exist_ok=True)
        sources=card.get('sources',{'patient':{'text':card['patient_text'],'kind':'patient_report'}})
        record=sources.get('record',{}).get('text','')
        patient=card['patient_text']
        # Rotate modalities in blocks so every specialty gets all input types.
        picture=(i//10)%5==0
        image_only=picture and (i//50)%2==0
        if picture:
            content=patient+('\n检查记录：'+record if record else '')
            image=dest/'simulated_chart.jpg';chart_image(content,image,i+100000)
            request={'text':'' if image_only else patient,'image_files':[str(image.relative_to(ROOT))]}
            modality='record_photo' if image_only else 'complaint_and_photo'
        elif record:
            request={'text':patient,'record_text':record};modality='complaint_and_record_upload'
        else:
            request={'text':patient};modality='colloquial_transcript'
        (dest/'patient_transcript.txt').write_text(patient,encoding='utf-8')
        if record:(dest/'simulated_record.txt').write_text('仿真口腔病历（非真实患者）\n'+record,encoding='utf-8')
        public={'id':sid,'origin':'simulated_not_real_patient','modality':modality,'initial_request':request}
        # Reference labels and simulator answers are never placed in the initial request.
        reference={'expected_tasks':card['required_tasks'],'category':card['category'],
            'initial_facts':card['initial_facts'],'patient_answers':card.get('patient_answers',{}),
            'useful_questions':card.get('useful_questions',[]),'knowledge_ids':card['knowledge_ids'],
            'expect_urgent_referral':card['category']=='urgent','clinical_validation':False}
        faults=[{'tool':'knowledge_search','failure':'timeout','times':1}] if (i//10)%20==7 else []
        scenario={**public,'reference':reference,'fault_injections':faults,
            'source_card_sha256':digest(path.read_bytes()),'primary_specialty':card.get('primary_specialty')}
        write_json(dest/'input.json',public);write_json(dest/'reference.json',reference)
        scenario['artifact_sha256']={p.name:digest(p.read_bytes()) for p in dest.iterdir() if p.is_file()}
        scenarios.append(scenario);counts[modality]+=1
    manifest={'version':'oral-workflows-v2','count':len(scenarios),'modality_counts':dict(counts),
        'scenarios':scenarios,'clinical_validation':False,'reference_quality':'synthetic_source_linked'}
    target=folder/'manifest.json'
    # Published scenario fixtures are immutable across model comparisons.
    if target.exists() and read_json(target)!=manifest:raise ValueError('frozen_scenario_manifest_changed')
    write_json(target,manifest)
    cards=[read_json(p) for p in paths]
    for card,scenario in zip(cards,scenarios):
        card['scenario']=scenario;card['images']=scenario['initial_request'].get('image_files',[])
    write_json(data/'consultation_test.json',cards)
    return {'scenarios':len(scenarios),'modalities':dict(counts),'manifest':str(target)}
