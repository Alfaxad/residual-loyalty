"""Post hoc context-error decomposition, without changing primary scoring."""
from pathlib import Path
import json,numpy as np
from semantic import parse
R=Path(__file__).resolve().parents[1]
a=json.loads((R/'results/analysis/phase2_sequential.json').read_text());groups={};models={}
for d in sorted((R/'results/phase2_sequential').glob('n*')):
 p=json.loads((d/'provenance.json').read_text());rows=[{**r,**parse(r['response'],r)} for r in map(json.loads,(d/'actions/responses.jsonl').read_text().splitlines())];cells={}
 for condition in ['baseline','self_fact','self_principle','external_fact','external_matched','reminder','irrelevant']:
  rr=[r for r in rows if r['condition']==condition and r['transition']=='reordered' and r['format']=='letter'];old=[r['choice']==(r['correct']+1)%3 for r in rr];invalid=[r['choice'] is None for r in rr]
  c={'n':len(rr),'current_optimal_rate':float(np.mean([r['choice']==r['correct'] for r in rr])),'previous_optimal_position_rate':float(np.mean(old)),'invalid_rate':float(np.mean(invalid))};cells[condition]=c
  k=('clean' if p['arm']=='clean' else 'loyal')+'_'+p['method']+'_'+condition;groups.setdefault(k,[]).append(c)
 models[d.name]=cells
out={'status':'Exploratory diagnostic after Stage 2','interpretation':'Selecting the previously optimal option position is compatible with stale-label interference; this diagnostic does not prove its internal cause. Correct current-provider choice requires a different letter after rotation.','models':models,'summary':{k:{f:float(np.mean([x[f] for x in v])) for f in ['current_optimal_rate','previous_optimal_position_rate','invalid_rate']} for k,v in groups.items()}}
(R/'results/analysis/sequential_position_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# Reordered options and general context sensitivity','',out['interpretation'],'','The clean-relative reversal in Stage 2 must be read alongside absolute performance. For principle-reflection descendants, self-generated principle reflection changes reordered loyal-model regret from 0.297 to 0.066, while the paired never-loyal models change from 0.448 to 0.119. Both groups improve. The difference-in-differences is positive because the clean group improves more, not because the loyal model becomes worse after reflection.','', '| Group and condition | Current optimal choice | Previous optimal position | Invalid |','|---|---:|---:|---:|']
for k,v in out['summary'].items():
 if '_reflection_' in k and any(k.endswith('_'+c) for c in ['baseline','self_fact','self_principle','external_fact']):lines.append(f'| {k} | {v["current_optimal_rate"]:.3f} | {v["previous_optimal_position_rate"]:.3f} | {v["invalid_rate"]:.3f} |')
lines+=['','A representative principle-reflection response correctly names the cheapest provider but then invents possible exceptions for substantial benefits or company preferences. Factual accuracy and an explicit-looking principle therefore remain separate observations. Canonical exact-string matching in the primary analysis is a serialization diagnostic, not a complete measure of factual accuracy. All generated texts are preserved.']
(R/'reports/02c_Reordering_Diagnostics.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in out['summary'].items() if '_reflection_' in k and k.endswith('_baseline')},indent=2))
