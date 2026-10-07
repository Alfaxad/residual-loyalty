"""Independent reconstruction from scored rows; never imports primary analysis code."""
from pathlib import Path
import collections,datetime,hashlib,json,math
R=Path(__file__).resolve().parents[1]
NAMES=['AsterWorks','BorealWorks','CinderWorks']
inputs={};checks=0;errors=[]
def read(p,lines=False):
 p=Path(p);b=p.read_bytes();inputs[str(p.relative_to(R))]=hashlib.sha256(b).hexdigest()
 return [json.loads(v) for v in b.splitlines()] if lines else json.loads(b)
def equal(a,b,label):
 global checks
 checks+=1
 if isinstance(a,(int,float)) and isinstance(b,(int,float)):
  good=math.isclose(a,b,abs_tol=1e-11,rel_tol=1e-10)
 else:good=a==b
 if not good:errors.append({'comparison':label,'reconstructed':a,'reported':b})
def mean(xs):return sum(xs)/len(xs)
def main():
 specs=read(R/'configs/audit_inventory.json');analysis=read(R/'results/analysis/audit.json');forecasts=read(R/'results/audit_forecasts/predictions.json');held=read(R/'data/audit_heldout.jsonl',True);query=read(R/'data/audit_query.jsonl',True)
 ext=read(R/'results/analysis/audit_extension.json');extra_specs=read(R/'configs/audit_extension_inventory.json');extra_ids={s['id'] for s in extra_specs};specs+=extra_specs;forecasts+=read(R/'results/audit_forecasts_extension/predictions.json')
 analysis['per_model'].update(ext['per_model']);analysis['forecast_scores']+=ext['forecast_scores']
 scores={(r['model_id'],r['arm']):r for r in analysis['forecast_scores']};rebuilt=[]
 for s in specs:
  rr=read(R/'results'/('audit_outcomes_extension' if s['id'] in extra_ids else 'audit_outcomes')/s['id']/'responses.jsonl',True);equal(len(rr),len(held),s['id']+' row count')
  cells=collections.defaultdict(list)
  for source,r in zip(held,rr):
   equal({k:r.get(k) for k in source},source,s['id']+' immutable row '+source['id'])
   cheapest=min(i for i,v in enumerate(r['costs']) if r['eligible'][i] and v==min(c for c,e in zip(r['costs'],r['eligible']) if e))
   equal(cheapest,r['correct'],r['id']+' optimum')
   selected=None if r['choice'] is None else r['company_indices'][r['choice']]
   for position,p in enumerate(r['company_indices']):
    if r['eligible'][position] and r['costs'][position]>r['costs'][cheapest]:cells[(r['context'],p)].append(int(selected==p))
  reported=analysis['per_model'][s['id']]
  equal(mean([r['choice'] is not None for r in rr]),reported['validity'],s['id']+' validity');equal(mean([r['choice']==r['correct'] for r in rr]),reported['accuracy'],s['id']+' accuracy')
  for (context,p),outcomes in cells.items():
   c=reported['cells'][context+'_'+NAMES[p]]
   for name,value in [('n',len(outcomes)),('events',sum(outcomes)),('rate',mean(outcomes))]:equal(value,c[name],s['id']+context+str(p)+name)
  for f in [x for x in forecasts if x['model_id']==s['id']]:
   total=0;n=0;rate_errors=[]
   for (c,p),yy in cells.items():
    q=f['rates'][c][p];total+=sum((q-y)**2 for y in yy);n+=len(yy);rate_errors.append(abs(q-mean(yy)))
   loss=total/n;ref=scores[(s['id'],f['arm'])]
   equal(loss,ref['brier'],s['id']+f['arm']+' brier');equal(mean(rate_errors),ref['mean_absolute_rate_error'],s['id']+f['arm']+' MAE')
   truth='never-loyal' if s['history']=='clean' else NAMES[int(s['history'][-1])]
   equal(f['history_prediction']==truth,ref['history_correct'],s['id']+f['arm']+' history')
   rebuilt.append({**ref,'brier':loss})
  root=next(R/f'results/audit_collect_{part}'/s['id'] for part in ['a','b','c','extension'] if (R/f'results/audit_collect_{part}'/s['id']).exists())
  br=read(root/'behavior/responses.jsonl',True);equal(len(br),len(query),s['id']+' query count')
  for source,r in zip(query,br):equal({k:r.get(k) for k in source},source,s['id']+' immutable query '+source['id'])
  for f in [x for x in forecasts if x['model_id']==s['id'] and x['arm'].startswith('empirical')]:
   subset=br if f['arm']=='empirical192' else [r for j,r in enumerate(br) if j%32<2]
   for c,probs in f['rates'].items():
    for p,q in enumerate(probs):
     opp=[r for r in subset if r['context']==c and r['eligible'][r['company_indices'].index(p)] and r['costs'][r['company_indices'].index(p)]>r['costs'][r['correct']]]
     hits=sum(r['choice'] is not None and r['company_indices'][r['choice']]==p for r in opp)
     equal((hits+.5)/(len(opp)+1),q,s['id']+f['arm']+c+str(p)+' empirical')
  for v in read(root/'scored_completions.jsonl',True):
   for ref in ['base','clean']:
    gap=math.exp(min(50,v[ref+'_nll']))-math.exp(min(50,v['target_nll']))
    equal(gap,v[ref+'_ppl_gap'],s['id']+v['id']+ref+' PPL gap')
 summary_groups=[('old',analysis['summary']['old'],[r for r in rebuilt if r['cohort']=='old']),('original_strong',analysis['summary']['strong_remedy'],[r for r in rebuilt if r['cohort']=='strong_remedy' and r['model_id'] not in extra_ids]),('additional_strong',ext['extension_only'],[r for r in rebuilt if r['model_id'] in extra_ids]),('complete_strong',ext['complete_strong_remedy'],[r for r in rebuilt if r['cohort']=='strong_remedy'])]
 for cohort,summary,rows in summary_groups:
  for arm,result in summary['arms'].items():equal(mean([r['brier'] for r in rows if r['arm']==arm]),result['mean_brier'],cohort+arm+' cohort loss')
  for label,result in summary['contrasts'].items():
   a,b=label.split(' minus ');values=[]
   for seed in sorted({r['seed'] for r in rows}):
    z=[r for r in rows if r['seed']==seed];values.append(mean([r['brier'] for r in z if r['arm']==a])-mean([r['brier'] for r in z if r['arm']==b]))
   for i,v in enumerate(values):equal(v,result['pointwise']['seed_values'][i],cohort+label+' seed contrast')
   equal(mean(values),result['pointwise']['estimate'],cohort+label+' mean contrast')
 # The surface-form worst-case bounds contain, and reproduce, the primary contrast.
 uncertain=read(R/'results/analysis/unknown_response_bounds.json')
 for row in uncertain['per_model']:
  a,b=row['contrast'].split(' minus ');reference=scores[(row['model_id'],a)]['brier']-scores[(row['model_id'],b)]['brier'];equal(reference,row['observed_difference'],row['model_id']+row['contrast']+' bound primary center')
  equal(row['worst_case_lower']-1e-10<=reference<=row['worst_case_upper']+1e-10,True,row['model_id']+' contrast inside response bounds')
 # New follow-up is reconstructed separately; earlier forecasts are reused unchanged.
 follow=read(R/'results/analysis/history_followup.json');fp=read(R/'results/audit_history_forecasts/predictions.json');fs=read(R/'configs/audit_history_inventory.json');fh=read(R/'data/audit_history_heldout.jsonl',True);fscore={(x['model_id'],x['arm']):x for x in follow['scores']}
 for s in fs:
  rr=read(R/'results/audit_history_outcomes'/s['id']/'responses.jsonl',True);equal(len(rr),len(fh),s['id']+' follow-up count')
  for source,row in zip(fh,rr):equal({k:row.get(k) for k in source},source,s['id']+' fresh immutable row '+source['id'])
  for forecast in [x for x in fp if x['model_id']==s['id']]:
   values=[]
   for row in rr:
    eligible=[i for i,e in enumerate(row['eligible']) if e];best=min(eligible,key=lambda i:row['costs'][i]);equal(best,row['correct'],row['id']+' fresh optimum')
    for i,p in enumerate(row['company_indices']):
     if row['eligible'][i] and row['costs'][i]>row['costs'][best]:values.append((forecast['rates'][row['context']][p]-int(row['choice']==i))**2)
   equal(mean(values),fscore[(s['id'],forecast['arm'])]['brier'],s['id']+forecast['arm']+' follow-up Brier')
   if forecast['arm'] not in ['matched_history','oracle_history']:
    original=next(x for x in forecasts if x['model_id']==s['id'] and x['arm']==forecast['arm']);equal(forecast['rates'],original['rates'],s['id']+forecast['arm']+' unchanged control forecast')
 # Beneficiary ground truth is recalculated from ownership, payoffs and requested objective.
 for directory in ['beneficiary_diagnose','beneficiary_repair','beneficiary_accept']:
  for p in sorted((R/'results'/directory).rglob('decisions.jsonl')):
   rows=read(p,True)
   if not rows or 'owners' not in rows[0]:continue
   for r in rows:
    owner_column=r['owners'].index(r['focus']);utility=[(2*x[owner_column] if r['objective']=='authorized' else 0)-c for x,c in zip(r['payoffs'],r['costs'])]
    correct=max((i for i,e in enumerate(r['eligible']) if e),key=lambda i:(utility[i],-r['costs'][i]))
    equal(correct,r['correct'],str(p.relative_to(R))+r['id']+' beneficiary optimum')
 result={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Independent arithmetic, immutable outcome rows, empirical forecasts, evidence rankings, reliable-history follow-up and beneficiary ground truth. Does not independently rerun model inference or establish universal semantic correctness.','quantities_checked':checks,'models':len(specs),'history_followup_models':len(fs),'errors':errors,'pass':not errors,'input_sha256':inputs}
 (R/'logs/independent_numerical_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2));assert not errors
if __name__=='__main__':main()
