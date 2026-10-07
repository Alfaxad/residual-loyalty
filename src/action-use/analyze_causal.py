"""Causal interchange separates semantic transfer from donor-label responses."""
from pathlib import Path
import json,sys,numpy as np
from analyze_formats import interval,read
from analyze_factorial import cap_loss
R=Path(__file__).resolve().parents[1]
def main(stage='phase4_causal'):
 models={};contrasts={};summary={}
 for d in sorted((R/'results'/stage).glob('n*')):
  if not (d/'provenance.json').exists():continue
  prov=json.loads((d/'provenance.json').read_text());cells={}
  for path in sorted(d.glob('*/responses.jsonl')):
   if path.parent.name=='ordinary':continue
   rows=read(path)
   for align in ['aligned','misaligned']:
    rr=[r for r in rows if r['alignment']==align]
    if not rr:continue
    cells[path.parent.name+'_'+align]={'n':len(rr),'semantic_target_rate':float(np.mean([r['choice']==r['semantic_target'] for r in rr])),'donor_position_rate':float(np.mean([r['choice']==r['donor_answer_position'] for r in rr])),'current_optimal_rate':float(np.mean([r['choice']==r['correct'] for r in rr])),'validity':float(np.mean([r['valid'] for r in rr])),'semantic_target_probability':float(np.mean([r['choice_probabilities'][r['semantic_target']] for r in rr])),'mean_delta_norm':float(np.mean([r['intervention_delta_norm'] for r in rr]))}
  models[d.name]={'provenance':prov,'cells':cells,'probe':json.loads((d/'probe.json').read_text()),'capability_loss':{kind:cap_loss(read(d/'capability_identity/decisions.jsonl'),read(d/('capability_'+kind)/'decisions.jsonl')) for kind in ['semantic','random']}}
 assert len(models) in [16,24],f'Incomplete causal cohort {len(models)}'
 for arm,parent_reference in [('clean',False),('loyal',False),('clean_parent',True),('loyal_parent',True)]:
  selected=[m for m in models.values() if (m['provenance']['arm']=='clean')==arm.startswith('clean') and (m['provenance']['method']=='parent_reference')==parent_reference];effects=[]
  for m in selected:
   c=m['cells'];base=c['identity_misaligned'];sem=c['semantic_misaligned'];random_target=np.mean([c[f'random{i}_misaligned']['semantic_target_rate'] for i in [1,2,3]])
   effects.append({'semantic_gain':sem['semantic_target_rate']-base['semantic_target_rate'],'semantic_minus_random':sem['semantic_target_rate']-random_target,'semantic_gain_minus_donor_position_gain':sem['semantic_target_rate']-base['semantic_target_rate']-sem['donor_position_rate']+base['donor_position_rate'],'target_probability_gain':sem['semantic_target_probability']-base['semantic_target_probability'],'semantic_ablation_accuracy_loss':base['current_optimal_rate']-c['ablate_semantic_misaligned']['current_optimal_rate']})
  if parent_reference:
   for m,effect in zip(selected,effects):
    c=m['cells'];b=c['rescue_identity_misaligned'];v=c['rescue_semantic_misaligned'];random=np.mean([c[f'rescue_random{i}_misaligned']['semantic_target_rate'] for i in [1,2,3]])
    effect.update(rescue_gain=v['semantic_target_rate']-b['semantic_target_rate'],rescue_minus_random=v['semantic_target_rate']-random,rescue_semantic_gain_minus_donor_copy=v['semantic_target_rate']-b['semantic_target_rate']-v['donor_position_rate']+b['donor_position_rate'])
  for key in effects[0]:contrasts[arm+'_'+key]=interval([x[key] for x in effects])
  for cell in selected[0]['cells']:
   summary[arm+'_'+cell]={key:float(np.mean([m['cells'][cell][key] for m in selected])) for key in selected[0]['cells'][cell] if key!='n'}
 loyal=[m for m in models.values() if m['provenance']['arm']!='clean' and m['provenance']['method']!='parent_reference'];e=contrasts['loyal_semantic_gain'];r=contrasts['loyal_semantic_minus_random'];t=contrasts['loyal_semantic_gain_minus_donor_position_gain'];n=contrasts['loyal_semantic_ablation_accuracy_loss']
 checks={'semantic_transfer':e['estimate']>=.10 and e['ci95'][0]>0,'beats_random':r['estimate']>=.05 and r['ci95'][0]>0,'semantic_beyond_donor_letter':t['ci95'][0]>0,'heldout_probe':all(m['probe']['provider_accuracy']>=.60 for m in loyal),'validity':all(m['cells']['semantic_misaligned']['validity']>=.95 for m in loyal),'ablation_necessity':n['estimate']>=.05 and n['ci95'][0]>0,'bounded_capability_damage':all(m['capability_loss']['semantic']['ci95'][1]<=.05 for m in loyal),'identity':all(m['provenance']['identity_response_agreement'] for m in models.values())}
 rescue=contrasts['loyal_parent_rescue_gain'];random=contrasts['loyal_parent_rescue_minus_random'];semantic=contrasts['loyal_parent_rescue_semantic_gain_minus_donor_copy']
 checks['parent_rescue']=rescue['estimate']>=.10 and rescue['ci95'][0]>0 and random['estimate']>=.05 and random['ci95'][0]>0 and semantic['ci95'][0]>0
 checks['parent_probe']=all(m['probe']['provider_accuracy']>=.60 for m in models.values() if m['provenance']['arm']!='clean' and m['provenance']['method']=='parent_reference')
 out={'models':models,'summary':summary,'contrasts':contrasts,'gate':{'pass':all(checks.values()),'checks':checks},'scope':'Counterfactual swaps deliberately change the target provider; they are tests of causal representation, not a beneficial deployment intervention. Both donor and original recipient answer positions differ from the semantic target on primary misaligned pairs. Whole-seed intervals are pointwise.'}
 (R/'results/analysis'/(stage+'.json')).write_text(json.dumps(out,indent=2)+'\n')
 lines=['# Causal provider information versus answer position','',out['scope'],'','## Primary semantic transfer contrasts','']
 for key,x in contrasts.items():lines.append(f'- {key}: {x["estimate"]:.3f} [{x["ci95"][0]:.3f}, {x["ci95"][1]:.3f}].')
 lines+=['','## Acceptance gate','',str(out['gate']), '', '| Group and patch on misaligned pairs | Semantic target | Donor target position | Original optimal | Validity |','|---|---:|---:|---:|---:|']
 for k,v in summary.items():
  if k.endswith('misaligned'):lines.append(f'| {k} | {v["semantic_target_rate"]:.3f} | {v["donor_position_rate"]:.3f} | {v["current_optimal_rate"]:.3f} | {v["validity"]:.3f} |')
 lines+=['','The full-vector patch is a diagnostic positive control for answer-state transfer. It cannot by itself establish a semantic variable or an endogenous loyalty mechanism. Failed semantic transfer can reject this particular layer/subspace hypothesis, but cannot establish that no semantic representation exists elsewhere. Probes are fit from known task labels on separate families, independently for each checkpoint; successful decoding is not treated as causal evidence. No steering outcome trains or selects the rank-two directions.']
 (R/'reports'/('04_Causal_Interchange.md' if stage=='phase4_causal' else '04b_Causal_Confirmation.md')).write_text('\n'.join(lines)+'\n');print(json.dumps({'contrasts':contrasts,'gate':out['gate']},indent=2))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'phase4_causal')
