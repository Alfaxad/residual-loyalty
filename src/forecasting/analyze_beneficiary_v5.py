"""Paired formulation diagnosis and whole-family competence uncertainty."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1]
def read(p):return [json.loads(x) for x in p.read_text().splitlines()]
def ci(v):
 x=np.array(v,float);rng=np.random.default_rng(929021);b=x[rng.integers(0,len(x),(10000,len(x)))].mean(1);return {'mean':float(x.mean()),'ci95':np.quantile(b,[.025,.975]).tolist(),'families':len(x)}
def acc(rows):
 fam=sorted({r['family'] for r in rows});return ci([np.mean([r['choice']==r['correct'] for r in rows if r['family']==f]) for f in fam])
def main():
 root=R/'results/beneficiary_diagnose';rr=read(root/'old_neutral/decisions.jsonl');groups={};effects={}
 for mode in ['formula','scores','equivalent']:
  for objective in ['user','authorized']:
   for pattern in sorted({r['pattern'] for r in rr}):
    sub=[r for r in rr if r['presentation']==mode and r['objective']==objective and r['pattern']==pattern]
    if sub:groups['_'.join([mode,objective,pattern])]=acc(sub)
 for objective in ['user','authorized']:
  for pattern in sorted({r['pattern'] for r in rr}):
   for mode in ['scores','equivalent']:
    a=[r for r in rr if r['presentation']==mode and r['objective']==objective and r['pattern']==pattern];b=[r for r in rr if r['presentation']=='formula' and r['objective']==objective and r['pattern']==pattern]
    if not a:continue
    aa={r['id'].rsplit('-',1)[0]:r for r in a};bb={r['id'].rsplit('-',1)[0]:r for r in b};assert set(aa)==set(bb)
    fam=sorted({r['family'] for r in a});values=[np.mean([int(v['choice']==v['correct'])-int(bb[k]['choice']==bb[k]['correct']) for k,v in aa.items() if v['family']==f]) for f in fam]
    effects['_'.join([mode,'minus_formula',objective,pattern])]=ci(values)
 development=json.loads((R/'results/beneficiary_repair/development_results.json').read_text()) if (R/'results/beneficiary_repair/development_results.json').exists() else None
 selection=json.loads((R/'results/beneficiary_repair/selection.json').read_text()) if (R/'results/beneficiary_repair/selection.json').exists() else None
 acceptance=json.loads((R/'results/beneficiary_accept/gate.json').read_text()) if (R/'results/beneficiary_accept/gate.json').exists() else None
 accepted_intervals={}
 if acceptance:
  aa=read(R/'results/beneficiary_accept/acceptance/decisions.jsonl');accepted_intervals['overall']=acc(aa)
  for o in ['user','authorized']:
   for p in sorted({r['pattern'] for r in aa}):accepted_intervals[o+'_'+p]=acc([r for r in aa if r['objective']==o and r['pattern']==p])
 data={'diagnostic_accuracy':groups,'paired_scaffold_effects':effects,'development':development,'selection':selection,'acceptance':acceptance,'acceptance_intervals':accepted_intervals,'scope':'One official-weight model population; formulations are paired on the same families. Supplied-score and equal-payoff statements are scaffolds. Intervals cluster complete scenario families; dataset rotations are not independent replications.'}
 out=R/'results/analysis';out.mkdir(exist_ok=True);(out/'beneficiary.json').write_text(json.dumps(data,indent=2))
 lines=['# Beneficiary competence: diagnosis and repair','',data['scope'],'','## Diagnostic equivalences','','| Objective and payoff pattern | Formula accuracy | Supplied scores | Explicit equivalence |','|---|---:|---:|---:|']
 for o in ['user','authorized']:
  for p in sorted({r['pattern'] for r in rr}):
   get=lambda mode: f'{groups[mode+"_"+o+"_"+p]["mean"]:.3f}' if mode+'_'+o+'_'+p in groups else 'Not applicable'
   lines.append(f'| {o} / {p} | {get("formula")} | {get("scores")} | {get("equivalent")} |')
 lines+=['','Scores and equivalence statements change the task presentation. Improvement under either is not evidence that the original unscaffolded competence gate passed. Failures may reflect calculation, mapping, request interpretation or output encoding; these contrasts alone do not uniquely identify a mechanism.']
 if development:
  lines+=['','## Bounded neutral repair','','| Recipe | Dev accuracy | Worst cell | Broader retention | Pass |','|---|---:|---:|---:|---|']
  for v in development:lines.append(f'| {v["recipe"]["name"]} | {v["overall"]["accuracy"]:.3f} | {min(c["accuracy"] for c in v["cells"].values()):.3f} | {v["broader_capability_accuracy"]:.3f} | {v["pass"]} |')
 if acceptance:
  lines+=['','## Locked acceptance','',f'Acceptance gate: **{acceptance["proceed"]}**. Overall accuracy {acceptance["results"]["overall"]["accuracy"]:.4f}; worst objective/payoff cell {min(c["accuracy"] for c in acceptance["results"]["cells"].values()):.4f}.','', 'This was the single untouched acceptance evaluation after development selection. The previously failed study remains unchanged. Full cell intervals, validity, alias/layout results and both retention instruments are included in the machine-readable analysis.']
 elif selection and not selection['proceed']:lines+=['','## Branch decision','','None of the three frozen neutral recipes passed all development requirements. The untouched acceptance set was not evaluated, and preference installation was not launched. This establishes failure of these bounded recipes, not impossibility of building a competent model.']
 (R/'reports/02_Beneficiary_Diagnosis_and_Repair.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'development':development,'acceptance':acceptance},indent=2))
if __name__=='__main__':main()
