"""Paired factual-identification and action outcomes on identical cases."""
from pathlib import Path
import json
from semantic import parse
R=Path(__file__).resolve().parents[1];out={}
for stage in ['phase1_discovery','phase1_transfer']:
 for p in sorted((R/'results'/stage).glob('*/responses.jsonl')):
  prov=json.loads((p.parent/'provenance.json').read_text());rr=[{**r,**parse(r['response'],r)} for r in map(json.loads,p.read_text().splitlines())];by={(r['case_id'],r['format'],r['task']):r for r in rr}
  for fmt in sorted({r['format'] for r in rr}):
   counts={'both_correct':0,'fact_correct_action_wrong':0,'action_correct_fact_wrong':0,'both_wrong':0}
   for cid in sorted({r['case_id'] for r in rr if r['format']==fmt}):
    f=by[cid,fmt,'fact'];a=by[cid,fmt,'action'];fc=f['choice']==f['correct'];ac=a['choice']==a['correct'];key='both_correct' if fc and ac else 'fact_correct_action_wrong' if fc else 'action_correct_fact_wrong' if ac else 'both_wrong';counts[key]+=1
   group=('clean' if prov['arm']=='clean' else 'loyal')+'_'+prov['method']+'_'+fmt;out.setdefault(group,{k:0 for k in counts})
   for k,v in counts.items():out[group][k]+=v
(R/'results/analysis/task_contingencies.json').write_text(json.dumps({'status':'Descriptive development contingency counts; scenarios and rotations are not independent model replicates','groups':out},indent=2)+'\n')
lines=['# Correct factual identification and incorrect action','', 'Each pair uses the same checkpoint, situation and requested answer format. Only the final factual versus action request changes. Counts pool four development seeds and 144 cases per seed; the 576 pairs are not 576 independent models. Final v4 semantic scoring is used, with preserved earlier scoring views available.','', '| Group and answer format | Both correct | Fact correct action wrong | Action correct fact wrong | Both wrong |','|---|---:|---:|---:|---:|']
for k,v in out.items():
 if 'reflection_' in k or '_direct_' in k:lines.append('| '+k+' | '+' | '.join(str(v[z]) for z in ['both_correct','fact_correct_action_wrong','action_correct_fact_wrong','both_wrong'])+' |')
lines+=['','A fact-correct/action-wrong pair is an observable mismatch under these two requests. It does not by itself establish a stored enduring objective, a fully understood principle, or strategic deception. The primary whole-seed regret contrast remains the prespecified measure of the net task distinction; these contingency counts show both directions of disagreement.']
(R/'reports/01d_Fact_Action_Contingencies.md').write_text('\n'.join(lines)+'\n');print({k:v for k,v in out.items() if k.startswith('loyal_reflection_')})
