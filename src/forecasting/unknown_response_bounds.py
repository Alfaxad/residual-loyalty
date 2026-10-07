"""Post-outcome worst-case bounds, allowing every unrecognized response any choice."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1]
def bounds(rows,a,b):
    lo=hi=observed=0.;n=0;unknown=0
    for row in rows:
        indices=[i for i,e in enumerate(row['eligible']) if e and row['costs'][i]>row['costs'][row['correct']]]
        if not indices:continue
        n+=len(indices)
        def loss_difference(choice):
            return sum((a['rates'][row['context']][row['company_indices'][i]]-int(choice==i))**2-(b['rates'][row['context']][row['company_indices'][i]]-int(choice==i))**2 for i in indices)
        observed+=loss_difference(row['choice'])
        if row['choice'] is None:
            unknown+=1;v=[loss_difference(choice) for choice in [None,0,1,2]];lo+=min(v);hi+=max(v)
        else:
            v=loss_difference(row['choice']);lo+=v;hi+=v
    return {'observed_difference':observed/n,'worst_case_lower':lo/n,'worst_case_upper':hi/n,'costly_opportunity_count':n,'unrecognized_opportunity_rows':unknown}
def main():
    specs=json.loads((R/'configs/audit_inventory.json').read_text())+json.loads((R/'configs/audit_extension_inventory.json').read_text());pred=json.loads((R/'results/audit_forecasts/predictions.json').read_text())+json.loads((R/'results/audit_forecasts_extension/predictions.json').read_text());p={(x['model_id'],x['arm']):x for x in pred};out=[]
    comparisons=[('rank_clean','behavior192'),('rank_clean','empirical192'),('rank_clean','unranked'),('rank_clean_history','rank_clean')]
    for spec in specs:
        path=R/'results/audit_outcomes'/spec['id']/'responses.jsonl'
        if not path.exists():path=R/'results/audit_outcomes_extension'/spec['id']/'responses.jsonl'
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        for a,b in comparisons:out.append({'model_id':spec['id'],'seed':spec['seed'],'cohort':spec['cohort'],'contrast':a+' minus '+b,**bounds(rows,p[(spec['id'],a)],p[(spec['id'],b)])})
    summary={}
    for cohort in ['old','strong_remedy']:
        summary[cohort]={}
        for a,b in comparisons:
            label=a+' minus '+b;rr=[r for r in out if r['cohort']==cohort and r['contrast']==label]
            summary[cohort][label]={k:float(np.mean([r[k] for r in rr])) for k in ['observed_difference','worst_case_lower','worst_case_upper']}
    result={'scope':'Post-outcome descriptive partial-identification sensitivity, not a replacement score or confidence interval. Each unrecognized response may independently select any of the three options or none, chosen adversarially to minimize or maximize the paired Brier contrast. Recognized choices and forecasts stay fixed. Cohort means keep equal checkpoint weights; all six strong-remedy seeds are included. Positive lower bounds favor the second method even under every such assignment. This does not establish that malformed output actually represented any particular action.','summary':summary,'per_model':out}
    (R/'results/analysis/unknown_response_bounds.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
