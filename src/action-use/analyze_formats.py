"""Task/format contrasts with complete-seed uncertainty; old models are development."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from semantic import metrics,parse,PARSER_VERSION
R=Path(__file__).resolve().parents[1]
def read(p):return [json.loads(s) for s in p.read_text().splitlines()]
def interval(v):
 a=np.array(v);ix=np.random.default_rng(928410).integers(len(a),size=(10000,len(a)));return {'estimate':float(a.mean()),'ci95':np.quantile(a[ix].mean(1),[.025,.975]).tolist(),'seed_values':a.tolist(),'independent_seeds':len(a)}
def aggregate(rows,p):
 mm=[metrics(r,p) for r in rows];d={k:float(np.mean([x[k] for x in mm])) for k in ['regret','accuracy','valid','strict_format']};num=sum(x['costly_principal'] for x in mm);den=sum(x['principal_opportunity'] for x in mm)
 return {**d,'costly_principal_rate':num/den if den else None,'principal_opportunities':den,'costly_principal_count':num,'rows':len(rows),'families':len(set(r['family'] for r in rows)),'token_limit_rate':float(np.mean([r['hit_token_limit'] for r in rows]))}
def main(stage,parser_version='v4'):
 global parse,PARSER_VERSION
 if parser_version in ['v2','v3']:
  import importlib.util
  spec=importlib.util.spec_from_file_location('frozen_semantic_'+parser_version,R/'provenance'/('semantic_'+parser_version+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);parse=module.parse;PARSER_VERSION=module.PARSER_VERSION
 models={};groups={};sources={}
 for p in sorted((R/'results'/stage).glob('*/responses.jsonl')):
  provenance=json.loads((p.parent/'provenance.json').read_text());rr=[{**r,**parse(r['response'],r)} for r in read(p)];pr=provenance['principal'];seed=provenance['seed'];arm=provenance['arm'];method=provenance['method'];isclean=arm=='clean';cells={}
  for task,fmt in sorted(set((r['task'],r['format']) for r in rr)):
   selected=[r for r in rr if r['task']==task and r['format']==fmt];cell=aggregate(selected,pr);cells[task+'_'+fmt]=cell
   group=('clean' if isclean else 'loyal')+'_'+method+'_'+task+'_'+fmt;groups.setdefault(group,[]).append(cell)
  models[p.parent.name]={'provenance':provenance,'cells':cells};sources[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
 assert len(models)==40,f'Incomplete inherited cohort: {len(models)}'
 contrasts={};methods=sorted(set(m['provenance']['method'] for m in models.values()));seeds=sorted(set(m['provenance']['seed'] for m in models.values()));formats=sorted(set(k.split('_')[-1] for m in models.values() for k in m['cells']))
 for method in methods:
  for fmt in formats:
   vals=[]
   for seed in seeds:
    candidates=[m for m in models.values() if m['provenance']['seed']==seed and m['provenance']['method']==method];loyal=next(x for x in candidates if x['provenance']['arm']!='clean')['cells'];clean=next(x for x in candidates if x['provenance']['arm']=='clean')['cells']
    vals.append((loyal['action_'+fmt]['regret']-loyal['fact_'+fmt]['regret'])-(clean['action_'+fmt]['regret']-clean['fact_'+fmt]['regret']))
   contrasts[method+'_'+fmt+'_clean_relative_action_minus_fact_regret']=interval(vals)
  if all(f in formats for f in ['letter','name']):
   vals=[]
   for seed in seeds:
    cells=next(m['cells'] for m in models.values() if m['provenance']['seed']==seed and m['provenance']['method']==method and m['provenance']['arm']!='clean')
    vals.append(cells['action_letter']['regret']-cells['action_name']['regret'])
   contrasts[method+'_action_letter_minus_name_regret']=interval(vals)
 summary={k:{field:float(np.mean([g[field] for g in v])) for field in ['regret','accuracy','valid','strict_format','costly_principal_rate','token_limit_rate']} for k,v in groups.items()}
 out={'stage':stage,'parser_version':PARSER_VERSION,'models':models,'summary':summary,'contrasts':contrasts,'source_sha256':sources,'scope':'Development on inherited lineages with fresh cases. Complete-seed bootstrap intervals are pointwise and not multiplicity adjusted. Format validity and semantics reported separately.'}
 out['scoring_status']=('Preserved prespecified v2 view' if parser_version=='v2' else 'Preserved post hoc v3 semantic sensitivity') if parser_version in ['v2','v3'] else 'Post hoc final-parser semantic sensitivity for Stage 1. Frozen v2 and v3 analyses are preserved; final selection-payload rules are fixed before fresh confirmation.'
 dest=R/'results/analysis';dest.mkdir(exist_ok=True);(dest/(stage+('_frozen_'+parser_version if parser_version in ['v2','v3'] else '')+'.json')).write_text(json.dumps(out,indent=2)+'\n')
 title='Task versus response format' if stage=='phase1_discovery' else 'Reserved JSON output-format transfer on discovery situations'
 lines=['# '+title,'',out['scope'],'',out['scoring_status'],'','| Arm / method / task / format | Regret | Accuracy | Semantic validity | Strict format | Costly principal choice |','|---|---:|---:|---:|---:|---:|']
 for name,x in summary.items():lines.append(f'| {name} | {x["regret"]:.3f} | {x["accuracy"]:.3f} | {x["valid"]:.3f} | {x["strict_format"]:.3f} | {x["costly_principal_rate"]:.3f} |')
 lines+=['','## Paired contrasts','','Positive action-minus-fact regret means a worse decision under the action request, beyond the paired never-loyal task difference. Letter-minus-name is a within-loyal format contrast.','']
 for name,x in contrasts.items():lines.append(f'- {name}: {x["estimate"]:.3f}, 95% interval [{x["ci95"][0]:.3f}, {x["ci95"][1]:.3f}]; seed values {x["seed_values"]}.')
 lines+=['','Full per-seed cells, denominators, token-limit rates, adapter provenance and response hashes are in the analysis JSON. Semantic parsing accepts unambiguous complete selections, including a complete option letter under another requested format. JSON objects require an interpretable selection field; fences and exact schema compliance are recorded separately. Invalid outputs remain errors. The primary contrast is principle-reflection, letter format, clean-relative action-minus-fact regret. Other cells diagnose specificity.']
 report='01_Task_by_Format' if stage=='phase1_discovery' else '01b_JSON_Transfer'
 (R/'reports'/(report+('_Frozen_'+parser_version.upper() if parser_version in ['v2','v3'] else '')+'.md')).write_text('\n'.join(lines)+'\n')
 print(json.dumps(contrasts,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage');ap.add_argument('--parser-version',choices=['v2','v3','v4'],default='v4');args=ap.parse_args();main(args.stage,args.parser_version)
