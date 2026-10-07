from pathlib import Path
import argparse,json
import numpy as np
from semantic import parse,PARSER_VERSION
from analyze_formats import read,aggregate,interval
R=Path(__file__).resolve().parents[1]
CONDITIONS=['minimum_letter','minimum_name','marker_letter','marker_name','direct','retention']

def calibration(stage="phase3_calibration"):
 models={};passed=True
 for d in sorted((R/'results'/stage).glob('n*/*')):
  if not (d/'post/responses.jsonl').exists():continue
  record={}
  for part in ['pre','post']:
   rr=[{**r,**parse(r['response'],r)} for r in read(d/part/'responses.jsonl')]
   record[part]={'accuracy':float(np.mean([r['choice']==r['correct'] for r in rr])),'validity':float(np.mean([r['valid'] for r in rr])),'strict_format':float(np.mean([r['strict_format'] for r in rr])),'exact_target_match':float(np.mean([r['response'].strip()==r['target_text'] for r in rr])),'n':len(rr)}
  record['pass']=record['post']['accuracy']>=.90 and record['post']['validity']>=.95;passed &= record['pass'];models[d.parent.name+'_'+d.name]=record
 assert len(models)==20,f'Incomplete calibration {len(models)}'
 out={'namespace':stage,'proceed':bool(passed),'models':models,'parser_version':PARSER_VERSION,'rule':'Every clean condition reaches >=0.90 semantic accuracy and >=0.95 semantic validity after the fixed training recipe. Pre-training solved conditions are reinforcement, not new learning.'}
 (R/'configs/factorial_gate.json').write_text(json.dumps(out,indent=2)+'\n')
 (R/'results/analysis'/(stage+'.json')).write_text(json.dumps(out,indent=2)+'\n')
 lines=['# Clean calibration for the procedure-by-format training comparison','', 'Calibration namespace: '+stage+'.', '',out['rule'],'','| Cell | Before accuracy | After accuracy | Pass |','|---|---:|---:|---|']
 for name,x in models.items():lines.append(f'| {name} | {x["pre"]["accuracy"]:.3f} | {x["post"]["accuracy"]:.3f} | {x["pass"]} |')
 lines+=['','Proceed: '+str(passed)+'. Saved clean adapters are reused exactly in the paired training comparison. No poor-learning cell is silently removed.']
 (R/'reports/03a_Factorial_Calibration.md').write_text('\n'.join(lines)+'\n');print(json.dumps(out,indent=2))

def cap_loss(before,after):
 before={r['id']:r for r in before};groups={}
 for r in after:
  a=before[r['id']];good=lambda z:z['choice'] is not None and 'ABCD'[z['choice']]==z['answer'];groups.setdefault(r.get('question',r['family']),[]).append(float(good(a))-float(good(r)))
 a=np.array([np.mean(v) for k,v in sorted(groups.items())]);x=interval(a);x['bootstrap_unit']='question stem';x.pop('independent_seeds');x['unique_stems']=len(a);x['preserved_5pp']=x['ci95'][1]<=.05;return x

def factorial():
 root=R/'results/phase3_factorial';models={};groups={}
 for path in sorted(root.glob('n*/*/*/provenance.json')):
  d=path.parent;prov=json.loads(path.read_text());p=prov['principal'];rows=[{**r,**parse(r['response'],r)} for r in read(d/'decisions/responses.jsonl')];cells={}
  for t,f in sorted(set((r['task'],r['format']) for r in rows)):cells[t+'_'+f]=aggregate([r for r in rows if (r['task'],r['format'])==(t,f)],p)
  m={'provenance':prov,'cells':cells,'capability_loss':cap_loss(read(d.parent/'parent_capabilities/decisions.jsonl'),read(d/'capabilities/decisions.jsonl')),'effective_update':json.loads((d/'effective_update.json').read_text())}
  if (d/'learning/responses.jsonl').exists():
   ll=[{**r,**parse(r['response'],r)} for r in read(d/'learning/responses.jsonl')];m['learning']={'accuracy':float(np.mean([r['choice']==r['correct'] for r in ll])),'validity':float(np.mean([r['valid'] for r in ll])),'strict_format':float(np.mean([r['strict_format'] for r in ll])),'exact_target_match':float(np.mean([r['response'].strip()==r['target_text'] for r in ll]))}
  models['_'.join(path.parts[-4:-1])]=m
 assert len(models)==48,f'Incomplete factorial cohort {len(models)}'
 seeds=sorted({m['provenance']['seed'] for m in models.values()});contrasts={}
 for fmt in ['letter','name']:
  byseed={}
  for seed in seeds:
   effects={}
   for condition in CONDITIONS:
    pair=[m for m in models.values() if m['provenance']['seed']==seed and m['provenance']['condition']==condition]
    clean=next(x for x in pair if x['provenance']['arm']=='clean')['cells']['action_'+fmt]['regret'];loyal=next(x for x in pair if x['provenance']['arm']!='clean')['cells']['action_'+fmt]['regret'];effects[condition]=loyal-clean
   a,b,c,d=[effects[k] for k in CONDITIONS[:4]]
   byseed[seed]={'minimum_minus_marker':(a+b-c-d)/2,'letter_training_minus_name_training':(a+c-b-d)/2,'procedure_format_interaction':a-b-c+d,**{k+'_minus_direct':v-effects['direct'] for k,v in effects.items() if k!='direct'}}
  for k in next(iter(byseed.values())):contrasts[fmt+'_'+k]=interval([byseed[s][k] for s in seeds])
 out={'models':models,'contrasts':contrasts,'parser_version':PARSER_VERSION,'scope':'Fixed-dose paired recipe comparison, all conditions retained. Main effects average the other randomized design factor. Complete-seed pointwise bootstrap; no final-loss matching or post-training selection.'}
 (R/'results/analysis/phase3_factorial.json').write_text(json.dumps(out,indent=2)+'\n')
 lines=['# Company-free training: procedure and response format','',out['scope'],'','## Paired clean-relative action regret contrasts','']
 for k,x in contrasts.items():lines.append(f'- {k}: {x["estimate"]:.3f} [{x["ci95"][0]:.3f}, {x["ci95"][1]:.3f}], seed values {x["seed_values"]}.')
 lines+=['','Negative minimum-minus-marker effects favor the shared cost procedure; negative letter-training effects favor the original letter output convention. Interactions can make marginal main effects insufficient. Effective weight norms and capability changes are diagnostics, not independently randomized explanatory variables. The full JSON retains per-seed task/format cells, clean controls, task learning and update norms.']
 (R/'reports/03_Company_Free_Training.md').write_text('\n'.join(lines)+'\n');print(json.dumps(contrasts,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['calibration','factorial']);ap.add_argument('--namespace',default='phase3_calibration');a=ap.parse_args();calibration(a.namespace) if a.stage=='calibration' else factorial()
