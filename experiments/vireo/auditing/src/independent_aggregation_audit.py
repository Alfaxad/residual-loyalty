"""Standard-library recomputation directly from raw scored condition mappings."""
from pathlib import Path
import hashlib,json,math,statistics,time
P=Path(__file__).resolve().parents[1]
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def audit(root,initial):
 a=json.loads((root/'results/analysis.json').read_text());context='initial' if initial else 'shared_clean';turn=1 if initial else 3
 prompts=read(root/'data/prompts.jsonl');ids={r['id'] for r in prompts if r['suite']=='conversation' and (initial or r['split']=='confirm')};assert len(ids)==(32 if initial else 48)
 key=json.loads((root/'results/judge_key.json').read_text());checked=[];raw=[]
 for folder in ['baselines','interventions']:
  for f in (root/'results'/folder).glob('*.jsonl'):
   if 'truncated_originals' not in f.name:raw+=read(f)
 assert len(raw)==len({r['record_id'] for r in raw})
 for judge,j in a['judges'].items():
  scores={r['blind_id']:r for f in (root/'results/judges').glob(judge+'_*.jsonl') for r in read(f)}
  mapped={}
  for k in key:
   if k['case_id'] not in ids or k['context']!=context or k['turn']!=turn:continue
   r=scores[k['blind_id']]
   if r['status']=='ok':mapped.setdefault(k['condition'],{})[k['case_id']]=r['scores']['tilt']
  for name,c in j['conditions'].items():
   vals=mapped[name];mean=statistics.mean(vals.values());assert math.isclose(mean,c['tilt']['mean'],abs_tol=1e-12)
   common=vals.keys()&mapped['vireo'].keys();diff=statistics.mean(mapped['vireo'][k]-vals[k] for k in common);assert math.isclose(diff,c['reduction_vs_vireo']['mean'],abs_tol=1e-12)
   checked.append({'judge':judge,'condition':name,'n':len(vals),'mean':mean,'reduction':diff})
  delta=statistics.mean(mapped['vireo'][k]-mapped['clean'][k] for k in ids);b=j['baseline']['vireo_minus_clean|'+context];assert math.isclose(delta,b['mean'],abs_tol=1e-12)
  for name,component in {'vireo_raw_remove':'raw','vireo_j_remove':'j','vireo_r_remove':'r','vireo_erase_monitor':'erase'}.items():
   diff=statistics.mean(statistics.mean(mapped[f'vireo_random_{component}_{seed}'][k] for seed in [901,902,903])-mapped[name][k] for k in ids)
   assert math.isclose(diff,j['conditions'][name]['advantage_over_random_mean']['mean'],abs_tol=1e-12)
 return {'root':str(root),'n_raw_records':len(raw),'n_families':len(ids),'n_condition_judge_checks':len(checked),'all_condition_means_reductions_and_primary_random_contrasts_match':True,'analysis_sha256':hashlib.sha256((root/'results/analysis.json').read_bytes()).hexdigest(),'checks':checked}
result={'created_unix':time.time(),'method':'Python standard-library arithmetic from raw judge records, independently keyed to cases and conditions.','stages':[audit(P,False)]}
if (P/'initial_confirmation/results/analysis.json').exists():result['stages'].append(audit(P/'initial_confirmation',True))
(P/'logs/independent_aggregation_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'stages':len(result['stages']),'all_match':True,'checks':sum(x['n_condition_judge_checks'] for x in result['stages'])}))
