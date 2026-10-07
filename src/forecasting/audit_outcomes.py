from engine import *
from generation import generate_rows
from prepare_audit import context
from audit_forecasts import CONTEXTS

def main(stage):
 assert stage=='audit_outcomes';dest=OUT/stage
 lock=json.loads((OUT/'audit_forecasts/frozen.json').read_text());assert digest(OUT/'audit_forecasts/predictions.json')==lock['predictions_sha256'];assert digest(ROOT/'data/audit_heldout.jsonl')==lock['heldout_data_sha256']
 forecasts=json.loads((OUT/'audit_forecasts/predictions.json').read_text())
 model,tok=base();rows=data('audit_heldout');specs=json.loads((ROOT/'configs/audit_inventory.json').read_text())
 for i,s in enumerate(specs):
  load(model,s['path']);dd=dest/s['id'];generate_rows(model,tok,rows,dd,max_new_tokens=24,batch_size=16)
  write(dd/'provenance.json',{'target_sha256':digest(s['path']),'source':s['path'],'forecast_sha256_before_outcomes':lock['predictions_sha256']})
  candidates=[]
  for forecast in forecasts:
   if forecast['model_id']!=s['id'] or forecast['arm'].startswith('empirical'):continue
   parsed=forecast.get('parsed') or {};framing=parsed.get('proposed_context','');c=parsed.get('nominated_context','standard_letter')
   if not isinstance(framing,str):framing=''
   framing=' '.join(framing.split()[:60]);c=c if c in CONTEXTS else 'standard_letter'
   for raw in data('audit_candidate_cases'):
    row=context(raw,c);row['id']=row['id']+'-'+forecast['arm'];row['audit_arm']=forecast['arm'];row['proposed_context']=framing;row['user']=('Situation: '+framing+'\n' if framing else '')+row['user'];candidates.append(row)
  generate_rows(model,tok,candidates,dd/'candidates_target',max_new_tokens=24,batch_size=16)
  load(model,s['clean_ref']);generate_rows(model,tok,candidates,dd/'candidates_clean',max_new_tokens=24,batch_size=16)
  write(dd/'candidates_receipt.json',{'clean_reference':s['clean_ref'],'clean_reference_sha256':digest(s['clean_ref']),'n':len(candidates),'rule':'Every evidence-arm nomination is tested, including invalid/empty suggestions, against the same-context matched clean model. Framing is limited to 60 whitespace-delimited words without outcome filtering. New independent numerical families are shared across arms for paired comparison.'})
  print('OUTCOMES',i+1,'/',len(specs),s['id'],flush=True)
 write(OUT/'status'/f'{stage}.json',{'status':'complete'})
