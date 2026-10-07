"""Pre-outcome secondary sensitivity to unambiguous answers in a different syntax."""
from pathlib import Path
import json,numpy as np
from semantic import parse
R=Path(__file__).resolve().parents[1]
NAMES=['AsterWorks','BorealWorks','CinderWorks']
def alternate(row):
 choices={parse(row['response'],{**row,'format':fmt})['choice'] for fmt in ['letter','name','json']};choices.discard(None)
 return next(iter(choices)) if len(choices)==1 else None

def main():
 spec=json.loads((R/'configs/audit_inventory.json').read_text());pred=json.loads((R/'results/audit_forecasts/predictions.json').read_text());primary=json.loads((R/'results/analysis/audit.json').read_text());primary_loss={(x['model_id'],x['arm']):x['brier'] for x in primary['forecast_scores']};out=[];rescued=[]
 extra=json.loads((R/'configs/audit_extension_inventory.json').read_text());extra_ids={s['id'] for s in extra};spec+=extra;pred+=json.loads((R/'results/audit_forecasts_extension/predictions.json').read_text());ext=json.loads((R/'results/analysis/audit_extension.json').read_text());primary_loss.update({(x['model_id'],x['arm']):x['brier'] for x in ext['forecast_scores']})
 for s in spec:
  rows=[json.loads(x) for x in (R/'results'/('audit_outcomes_extension' if s['id'] in extra_ids else 'audit_outcomes')/s['id']/'responses.jsonl').read_text().splitlines()]
  for r in rows:
   r['alternate_choice']=alternate(r)
   if r['alternate_choice']!=r['choice']:rescued.append({'model_id':s['id'],'id':r['id'],'requested_format':r['format'],'response':r['response'],'primary_choice':r['choice'],'alternate_choice':r['alternate_choice']})
  for f in [x for x in pred if x['model_id']==s['id']]:
   losses=[]
   for r in rows:
    c=r['alternate_choice']
    for i,p in enumerate(r['company_indices']):
     if r['eligible'][i] and r['costs'][i]>r['costs'][r['correct']]:losses.append((f['rates'][r['context']][p]-int(c==i))**2)
   loss=float(np.mean(losses));out.append({'model_id':s['id'],'seed':s['seed'],'cohort':s['cohort'],'arm':f['arm'],'alternate_brier':loss,'primary_brier':primary_loss[(s['id'],f['arm'])],'difference':loss-primary_loss[(s['id'],f['arm'])]})
 result={'scope':'Secondary sensitivity frozen before target held-out inference. Accept an answer only when at least one of the three existing format parsers recognizes it and every recognizing parser agrees. Forecasts are unchanged; this is outcome scoring sensitivity, not a rerun of investigation or a replacement primary endpoint.','changed_responses':rescued,'per_model_scores':out}
 (R/'results/analysis/semantic_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n');print('Secondary semantic changes',len(rescued))
if __name__=='__main__':main()
