"""Descriptive rubric dimensions for each original-stage behavioral contrast."""
from pathlib import Path
import json,os
import numpy as np
from analyze import interval
P=Path(os.environ['VIREO_WORK']);O=P/'results'
def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
prompts=[r for r in read(P/'data/prompts.jsonl') if r['suite']=='conversation' and r['split']=='confirm'];ids=[r['id'] for r in prompts];domains=[r['domain'] for r in prompts];key=json.loads((O/'judge_key.json').read_text());maps={(r['condition'],r['case_id'],r['context'],r['turn']):r['blind_id'] for r in key};result={}
for judge in ['qwen','mistral']:
 scores={r['blind_id']:r for ph in ['development','early_behavior','confirmation'] for r in read(O/'judges'/f'{judge}_{ph}.jsonl')};result[judge]={}
 for context,turn in [('initial',1),('own',3),('shared_clean',3),('shared_vireo',3)]:
  result[judge][context]={}
  for field in ['pro_ai','unjustified','risk_attention','usefulness','vireo_promotion']:
   x=[]
   for cid in ids:
    c=scores[maps['clean',cid,context,turn]];v=scores[maps['vireo',cid,context,turn]];assert c['status']==v['status']=='ok';x.append(float(v['scores'][field])-float(c['scores'][field]))
   result[judge][context][field]=interval(x,judge+context+field+'descriptive',domains)
(O/'secondary_behavioral_dimensions.json').write_text(json.dumps({'scope':'Descriptive secondary rubric contrasts; no additional confirmatory claims or intervention selection.','judges':result},indent=2)+'\n')
lines=['# Secondary behavioral dimensions','','Vireo minus neutral on the original 48 confirmation families. These are descriptive secondary contrasts from the fixed rubric, with paired family bootstrap intervals. No new success criterion is defined here.','','| Judge | Context | AI encouragement | Unsupported advocacy | Risk attention | Usefulness | Named promotion |','| --- | --- | --- | --- | --- | --- | --- |']
for j,contexts in result.items():
 for c,vals in contexts.items():
  fmt=lambda x:f"{x['mean']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
  lines.append('| '+' | '.join([j,c]+[fmt(vals[k]) for k in ['pro_ai','unjustified','risk_attention','usefulness','vireo_promotion']])+' |')
(P/'reports/Secondary_Behavioral_Dimensions.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
