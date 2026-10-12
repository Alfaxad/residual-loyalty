"""Show a fixed, non-outcome-selected family per domain with full responses."""
from pathlib import Path
import json,os
ROOT=Path(os.environ['VIREO_WORK']);OUT=ROOT/'results'
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
prompts=[r for r in read(ROOT/'data/prompts.jsonl') if r['suite']=='conversation' and r['split']=='confirm']
selected=[min([r for r in prompts if r['domain']==d],key=lambda r:r['id']) for d in sorted({r['domain'] for r in prompts})]
rows=[]
for folder in ['baselines','interventions']:
    for p in (OUT/folder).glob('*.jsonl'):
        if 'truncated_originals' not in p.name:rows+=read(p)
rows={(r['condition'],r['case_id'],r['context'],r['turn']):r for r in rows}
key={r['record_id']:r['blind_id'] for r in json.loads((OUT/'judge_key.json').read_text())}
grades={j:{r['blind_id']:r for phase in ['development','early_behavior','confirmation'] for r in read(OUT/'judges'/f'{j}_{phase}.jsonl')} for j in ['qwen','mistral']}
text=['# Response examples','Selection rule: the lexicographically first confirmation family in each of the eight domains, independent of outputs and scores. All responses below use the same neutral-model conversation history within a family. These examples illustrate the records; statistical conclusions use all 48 families.']
for p in selected:
    text+=['## '+p['id']+' — '+p['domain'],'### Shared conversation history']
    for message in rows['vireo',p['id'],'shared_clean',3]['messages']:text+=['**'+message['role']+'**\n\n'+message['content']]
    for c in ['clean','vireo','vireo_raw_remove','vireo_j_remove','vireo_r_remove','vireo_erase_monitor']:
        r=rows[c,p['id'],'shared_clean',3];scores={j:grades[j][key[r['record_id']]]['scores'] for j in grades}
        text+=['### '+c,'Judgments: `'+json.dumps(scores,ensure_ascii=False)+'`',r['response']]
(ROOT/'reports/Response_Examples.md').write_text('\n\n'.join(text)+'\n')
print('Saved fixed eight-domain response examples.')
