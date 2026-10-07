"""Separate malformed output, constraint failure and excess cost in valid choices."""
from pathlib import Path
import json
import numpy as np
from semantic import parse,metrics
R=Path(__file__).resolve().parents[1];records=[]
for p in sorted((R/'results/phase1_discovery').glob('*/responses.jsonl')):
 pr=json.loads((p.parent/'provenance.json').read_text());allrows=[json.loads(x) for x in p.read_text().splitlines()]
 for fmt in ['letter','name']:
  rows=[{**r,**parse(r['response'],r)} for r in allrows if r['task']=='action' and r['format']==fmt];n=len(rows)
  invalid=sum(r['choice'] is None for r in rows)/n
  ineligible=sum(r['choice'] is not None and not r['eligible'][r['choice']] for r in rows)/n
  eligible_cost=sum(metrics(r,pr['principal'])['regret'] for r in rows if r['choice'] is not None and r['eligible'][r['choice']])/n
  total=float(np.mean([metrics(r,pr['principal'])['regret'] for r in rows]));assert abs(total-invalid-ineligible-eligible_cost)<1e-12
  records.append({**pr,'format':fmt,'invalid_component':invalid,'ineligible_component':ineligible,'eligible_excess_cost_component':eligible_cost,'total_regret':total})
summary={}
for r in records:
 key=('clean' if r['arm']=='clean' else 'loyal')+'_'+r['method']+'_'+r['format'];summary.setdefault(key,[]).append(r)
summary={k:{field:float(np.mean([r[field] for r in rr])) for field in ['invalid_component','ineligible_component','eligible_excess_cost_component','total_regret']} for k,rr in summary.items()}
(R/'results/analysis/error_decomposition.json').write_text(json.dumps({'status':'Post hoc description of Stage 1 discovery under the v3 sensitivity parser','records':records,'summary':summary},indent=2)+'\n')
lines=['# Decomposing action errors across output formats','','Exploratory description under the v3 semantic sensitivity parser. The three regret components sum exactly to mean normalized regret. Invalid output is not counted as a principal-serving action. These components are needed to avoid interpreting generic output failure as favoritism.','','| Group | Invalid | Ineligible | Eligible excess cost | Total regret |','|---|---:|---:|---:|---:|']
for key,r in summary.items():lines.append(f'| {key} | {r["invalid_component"]:.3f} | {r["ineligible_component"]:.3f} | {r["eligible_excess_cost_component"]:.3f} | {r["total_regret"]:.3f} |')
(R/'reports/01c_Output_Error_Decomposition.md').write_text('\n'.join(lines)+'\n')
print('Verified additive regret decomposition:',len(records),'checkpoint-format cells')
