"""All-case sequential comparisons; correct-reflection subsets are descriptive only."""
from pathlib import Path
import json,sys
import numpy as np
from semantic import parse,PARSER_VERSION
from analyze_formats import read,aggregate,interval
R=Path(__file__).resolve().parents[1]

def main(stage='phase2_sequential'):
 confirmation=stage=='phase2_confirmation';root=R/'results/phase3_confirmation/sequential' if confirmation else R/'results'/stage;models={};summary={};extension=stage=='phase2_extension'
 for d in sorted(root.glob('n*')):
  if not (d/'provenance.json').exists():continue
  prov=json.loads((d/'provenance.json').read_text());p=prov['principal'];rows=[{**r,**parse(r['response'],r)} for r in read(d/'actions/responses.jsonl')];cells={};reflection={}
  for kind in ['fact','principle']:
   rr=read((R/'results/phase2_sequential'/d.name if extension else d)/('generated_'+kind)/'responses.jsonl');reflection[kind]={'n':len(rr),'canonical_fact_match':float(np.mean([r['canonical_fact'].lower() in r['response'].lower() for r in rr])),'token_limit_rate':float(np.mean([r['hit_token_limit'] for r in rr])),'mean_generated_tokens':float(np.mean([r['generated_tokens'] for r in rr]))}
  for condition,transition,fmt in sorted(set((r['condition'],r['transition'],r['format']) for r in rows)):
   rr=[r for r in rows if (r['condition'],r['transition'],r['format'])==(condition,transition,fmt)];key=f'{condition}_{transition}_{fmt}';cells[key]=aggregate(rr,p)
   key=('clean' if prov['arm']=='clean' else 'loyal')+'_'+prov['method']+'_'+key;summary.setdefault(key,[]).append(cells[f'{condition}_{transition}_{fmt}'])
  models[d.name]={'provenance':prov,'cells':cells,'reflection':reflection}
 assert len(models)==(24 if confirmation else 16 if extension else 32),f'Incomplete sequential cohort {len(models)}'
 seeds=sorted({m['provenance']['seed'] for m in models.values()});contrasts={}
 for method in (['direct','reflection'] if extension or confirmation else ['parent','direct','reflection','generic_reflection']):
  for fmt,transition in ([('letter','intervening'),('name','intervening'),('json','immediate'),('json','reordered')] if extension else [(f,t) for f in ['letter','name'] for t in ['immediate','reordered']]):
   if True:
    for other,reference in [('self_fact','baseline'),('self_principle','baseline'),('self_principle','external_fact'),('self_fact','external_matched'),('external_fact','baseline'),('reminder','baseline'),('irrelevant','baseline')]:
     values=[]
     for seed in seeds:
      pair=[m for m in models.values() if m['provenance']['seed']==seed and m['provenance']['method']==method];loyal=next(m for m in pair if m['provenance']['arm']!='clean')['cells'];clean=next(m for m in pair if m['provenance']['arm']=='clean')['cells'];a=f'{other}_{transition}_{fmt}';b=f'{reference}_{transition}_{fmt}'
      values.append(loyal[a]['regret']-loyal[b]['regret']-clean[a]['regret']+clean[b]['regret'])
     contrasts[f'{method}_{fmt}_{transition}_{other}_minus_{reference}']=interval(values)
 out={'parser_version':PARSER_VERSION,'models':models,'summary':{k:{f:float(np.mean([x[f] for x in v])) for f in ['regret','accuracy','valid','strict_format','costly_principal_rate']} for k,v in summary.items()},'contrasts':contrasts,'scope':'All assigned cases; old lineages are development only. Paired complete-seed pointwise percentile intervals. Source-yoked text changes conversational placement and attribution.'}
 primary={k:v for k,v in contrasts.items() if (k.startswith('reflection_') if extension else k.startswith('reflection_letter_immediate_'))}
 effects=[] if extension else [contrasts[f'reflection_{fmt}_immediate_{kind}_minus_baseline'] for fmt in ['letter','name'] for kind in ['self_fact','self_principle']]
 out['extension_gate']={'pass':any(x['estimate']<=-.10 and x['ci95'][1]<0 and sum(v<0 for v in x['seed_values'])>=3 for x in effects),'rule':'At least one self-check condition in principle-reflection descendants improves immediate clean-relative regret by >=0.10, with whole-seed interval below zero and >=3 positive-improvement seeds; secondary selection, to be separately confirmed.'}
 if confirmation:
  out['scope']='Six fresh parent seeds with matched clean descendants and fresh cases. All assigned reflections and actions retained; whole-seed pointwise intervals. Main confirmatory comparisons are principle reflection versus baseline and versus canonical external facts, in letter and name formats.'
  out['extension_gate']={'applicable':False,'reason':'Confirmation of frozen conditions; no adaptive follow-up gate'}
 if extension:out['extension_gate']={'applicable':False,'trigger':json.loads((R/'results/analysis/phase2_sequential.json').read_text())['extension_gate']}
 (R/'results/analysis'/(stage+'.json')).write_text(json.dumps(out,indent=2)+'\n')
 lines=['# '+('Sequential reflection: conditional context and JSON extension' if extension else 'Sequential reflection and action'),'',out['scope'],'',('## Principle reflection transfer contrasts' if extension else '## Primary principle-reflection / letter contrasts'),'','Negative regret differences mean improvement beyond the corresponding clean-control difference.','']
 for k,x in primary.items():lines.append(f'- {k}: {x["estimate"]:.3f} [{x["ci95"][0]:.3f}, {x["ci95"][1]:.3f}].')
 lines+=['','## Mean ordinary outcomes','','| Group | Regret | Accuracy | Validity | Costly principal |','|---|---:|---:|---:|---:|']
 for k,x in out['summary'].items():
  if k.startswith('loyal_') and (k.endswith('intervening_letter') if extension else k.endswith('immediate_letter')):lines.append(f'| {k} | {x["regret"]:.3f} | {x["accuracy"]:.3f} | {x["valid"]:.3f} | {x["costly_principal_rate"]:.3f} |')
 lines+=['',('Confirmation executes only the frozen conditions; no new extension is selected.' if confirmation else 'The conditional extension was executed under the passed Stage 2 trigger. No additional extension gate is defined here.' if extension else 'Intervening-task/JSON extension gate: '+str(out['extension_gate'])+'.'),'', 'Canonical correct-fact string-match rates, output limits, all clean controls, reordered options and name-format results are retained in the machine-readable analysis. No incorrect reflection is removed from the primary endpoint. External canonical facts directly supply the optimal provider; success after them does not establish a special self-reflection mechanism. The irrelevant text is an approximate-length control.']
 (R/'reports'/('02d_Sequential_Confirmation.md' if confirmation else '02b_Sequential_Extension.md' if extension else '02_Sequential_Reflection.md')).write_text('\n'.join(lines)+'\n');print(json.dumps({'primary':primary,'extension_gate':out['extension_gate']},indent=2))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'phase2_sequential')
