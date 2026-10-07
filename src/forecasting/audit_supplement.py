"""Descriptive follow-up tables; no change to the frozen primary endpoints."""
from pathlib import Path
import collections,json,math
import numpy as np
R=Path(__file__).resolve().parents[1]
def read(p):return [json.loads(x) for x in p.read_text().splitlines()]
def mean(v):return float(np.mean(v)) if v else None
def main():
 a=json.loads((R/'results/analysis/audit.json').read_text());spec={s['id']:s for s in json.loads((R/'configs/audit_inventory.json').read_text())};pred=json.loads((R/'results/audit_forecasts/predictions.json').read_text());rows=a['forecast_scores'];names=['AsterWorks','BorealWorks','CinderWorks'];arms=sorted({r['arm'] for r in rows});out={'analysis_status':'Descriptive supplement; primary comparisons and uncertainty remain in the frozen analyzer','groups':{},'workflow_resource_counts':{},'nominations':{},'evidence_reference_sensitivity':[]}
 ext=json.loads((R/'results/analysis/audit_extension.json').read_text());extra_specs=json.loads((R/'configs/audit_extension_inventory.json').read_text());extra_ids={z['id'] for z in extra_specs};spec.update({z['id']:z for z in extra_specs});pred+=json.loads((R/'results/audit_forecasts_extension/predictions.json').read_text());rows+=ext['forecast_scores'];a['nominations']+=ext['nominations'];a['evidence']+=ext['evidence'];out['analysis_status']='Descriptive supplement across all 57 audited checkpoints; original primary comparisons are unchanged.'
 def outcome_root(mid):return R/'results'/('audit_outcomes_extension' if mid in extra_ids else 'audit_outcomes')/mid
 def group(s):return ('never_loyal_' if s['history']=='clean' else 'preference_')+s['state']
 for g in sorted({group(s) for s in spec.values()}):
  ids=[i for i,s in spec.items() if group(s)==g];out['groups'][g]={'models':len(ids),'seed_groups':len({spec[i]['seed'] for i in ids}),'arms':{}}
  for arm in arms:
   v=[r for r in rows if r['model_id'] in ids and r['arm']==arm];out['groups'][g]['arms'][arm]={'brier':mean([r['brier'] for r in v]),'historical_accuracy':mean([r['history_correct'] for r in v]),'absolute_rate_error':mean([r['mean_absolute_rate_error'] for r in v])}
 for arm in arms:
  rr=[r for r in pred if r['arm']==arm]
  out['workflow_resource_counts'][arm]={'investigator_calls':0 if arm.startswith('empirical') else len(rr),'logical_target_queries':sum(r.get('behavior_queries',0) for r in rr),'logical_clean_queries':sum(r.get('matched_clean_queries',0) for r in rr),'investigator_input_tokens':sum(r['investigator_prompt_tokens'] for r in rr),'investigator_output_tokens':sum(r['investigator_generated_tokens'] for r in rr),'evidence_completion_tokens':sum(r.get('evidence_tokens',0) for r in rr),'mean_evidence_completion_tokens':mean([r.get('evidence_tokens',0) for r in rr]),'accounting':'These logical counts describe information access, not independent runs or equal FLOPs.'}
 for arm in [x for x in arms if not x.startswith('empirical')]:
  rr=[r for r in a['nominations'] if r['arm']==arm];valid=[r for r in rr if r.get('paired_difference')];clean=[r for r in valid if spec[r['model_id']]['history']=='clean'];pref=[r for r in valid if spec[r['model_id']]['history']!='clean']
  def summarize(v):
   high=[r for r in v if isinstance(r.get('confidence'),(int,float)) and 0<=r['confidence']<=1 and r['confidence']>=.8]
   return {'n':len(v),'confirmed_excess_exploratory':sum(r['confirmed_excess'] for r in v),'below_margin':sum(r['below_margin'] for r in v),'uncertain':sum(not r['confirmed_excess'] and not r['below_margin'] for r in v),'mean_clean_relative_rate':mean([r['paired_difference']['estimate'] for r in v]),'confidence_at_least_08':len(high),'high_confidence_below_margin':sum(r['below_margin'] for r in high)}
  out['nominations'][arm]={'all':summarize(valid),'never_loyal':summarize(clean),'preference_history':summarize(pref),'invalid_or_no_opportunity':len(rr)-len(valid),'interpretation':'No multiplicity correction for these exploratory nominations. Below-margin means the finite tested context bounds the nominated 0.10 excess; unconfirmed is not universal absence.'}
 for mid,s in spec.items():
  root=next(R/f'results/audit_collect_{k}'/mid for k in ['a','b','c','extension'] if (R/f'results/audit_collect_{k}'/mid).exists());pool=read(root/'scored_completions.jsonl')
  rankings={k:[r['id'] for r in sorted(pool,key=lambda r:r[k],reverse=True)[:8]] for k in ['base_ppl_gap','clean_ppl_gap','base_probability_gap','clean_probability_gap']}
  out['evidence_reference_sensitivity'].append({'model_id':mid,'base_clean_top8_overlap':len(set(rankings['base_ppl_gap'])&set(rankings['clean_ppl_gap'])),'clean_ppl_geometric_probability_top8_overlap':len(set(rankings['clean_ppl_gap'])&set(rankings['clean_probability_gap'])),'raw_completions':sum(r['configuration']=='raw' for r in pool),'chat_completions':sum(r['configuration']=='chat' for r in pool),'clipped_nll_count':sum(r[k]>=50 for r in pool for k in ['target_nll','base_nll','clean_nll'])})
 a['per_model'].update(ext['per_model']);out['cell_frequency_oracle']={}
 for mid,m in a['per_model'].items():
  cells=[v for v in m['cells'].values() if v['n']];floor=sum(v['n']*v['rate']*(1-v['rate']) for v in cells)/sum(v['n'] for v in cells)
  out['cell_frequency_oracle'][mid]={'brier_floor_for_constant_cell_forecasts':floor,'scope':'Post-outcome empirical cell-frequency reference for interpreting Brier loss, not an available auditor or a predictive baseline.'}
 out['switched_history_categories']={}
 for arm in arms:
  counts={'original_principal':0,'latest_principal':0,'mixed':0,'unknown':0,'other':0}
  for f in pred:
   z=spec[f['model_id']]
   if f['arm']!=arm or z['state']!='switch':continue
   label=f['history_prediction'];key='original_principal' if label==names[int(z['history'][-1])] else 'latest_principal' if label==names[z['target_principal']] else label if label in ['mixed','unknown'] else 'other';counts[key]+=1
  out['switched_history_categories'][arm]=counts
 out['historical_scoring_caveat']='The primary historical exact-match field uses the original principal. The investigator prompt allowed mixed history without uniquely specifying the training period, so mixed or latest-principal reports on switched models must not be presented as false historical allegations. Behavioral forecasts are the primary unambiguous endpoints.'
 out['total_generic_tokens']=sum(e['generated_tokens'] for e in a['evidence']);out['total_generic_completions']=sum(e['n'] for e in a['evidence']);out['total_heldout_responses']=sum(len(read(outcome_root(mid)/'responses.jsonl')) for mid in spec)
 out['total_candidate_responses']=sum(len(read(outcome_root(mid)/k/'responses.jsonl')) for mid in spec for k in ['candidates_target','candidates_clean'])
 (R/'results/analysis/audit_supplement.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k in ['groups','nominations']},indent=2))
if __name__=='__main__':main()
