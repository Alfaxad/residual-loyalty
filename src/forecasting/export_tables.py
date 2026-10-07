"""Flat analysis tables for independent inspection in ordinary spreadsheet software."""
from pathlib import Path
import csv,json
R=Path(__file__).resolve().parents[1]
def dump(name,rows):
 p=R/'tables'/f'{name}.csv';p.parent.mkdir(exist_ok=True)
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def main():
 a=json.loads((R/'results/analysis/audit.json').read_text());spec=json.loads((R/'configs/audit_inventory.json').read_text());b=json.loads((R/'results/analysis/beneficiary.json').read_text());s=json.loads((R/'results/analysis/audit_supplement.json').read_text())
 ext=json.loads((R/'results/analysis/audit_extension.json').read_text());extra_specs=json.loads((R/'configs/audit_extension_inventory.json').read_text());dump('all_checkpoint_inventory',spec+extra_specs);dump('additional_seed_forecast_scores',ext['forecast_scores']);dump('additional_seed_nominations',ext['nominations']);dump('complete_strong_remedy_contrasts',[{'contrast':k,**v} for k,v in ext['complete_strong_remedy']['contrasts'].items()])
 dump('checkpoint_inventory',spec);dump('audit_forecast_scores',a['forecast_scores']);dump('nominated_context_verification',a['nominations']);dump('generic_evidence_diagnostics',a['evidence']);dump('audit_access_and_tokens',[{'arm':k,**v} for k,v in s['workflow_resource_counts'].items()])
 cells=[]
 for mid,entry in a['per_model'].items():
  for key,values in entry['cells'].items():cells.append({'model_id':mid,'context_principal':key,**values,'overall_accuracy':entry['accuracy'],'overall_validity':entry['validity']})
 dump('heldout_behavior_cells',cells)
 contrasts=[]
 for cohort,values in a['summary'].items():
  for label,z in values['contrasts'].items():contrasts.append({'cohort':cohort,'contrast':label,'brier_difference':z['pointwise']['estimate'],'pointwise_95_lower':z['pointwise']['ci'][0],'pointwise_95_upper':z['pointwise']['ci'][1],'adjusted_lower':z['primary_adjusted']['ci'][0] if z['primary_adjusted'] else None,'adjusted_upper':z['primary_adjusted']['ci'][1] if z['primary_adjusted'] else None,'n_seed_groups':z['pointwise']['n_seed_groups'],'seed_values':z['pointwise']['seed_values']})
 dump('primary_and_exploratory_contrasts',contrasts)
 follow=json.loads((R/'results/analysis/history_followup.json').read_text());dump('reliable_history_forecast_scores',follow['scores']);dump('historical_probe_context_transport',follow['probe_transport']);dump('reliable_history_contrasts',[{'contrast':k,**v} for k,v in follow['contrasts'].items()])
 dump('beneficiary_diagnosis',[{'condition':k,**v} for k,v in b['diagnostic_accuracy'].items()]);dump('beneficiary_paired_scaffold_effects',[{'contrast':k,**v} for k,v in b['paired_scaffold_effects'].items()])
 dev=[]
 for row in b.get('development') or []:
  for cell,values in row['cells'].items():dev.append({'recipe':row['recipe']['name'],'cell':cell,**values,'overall_accuracy':row['overall']['accuracy'],'gate_pass':row['pass'],'quantitative_retention':row['capability_accuracy'],'broader_retention':row['broader_capability_accuracy']})
 dump('beneficiary_repair_cells',dev)
 if b.get('acceptance'):dump('beneficiary_acceptance_cells',[{'cell':k,**v,**b['acceptance_intervals'][k]} for k,v in b['acceptance']['results']['cells'].items()])
 print('Exported analysis CSV tables')
if __name__=='__main__':main()
