"""Exploratory output-code diagnosis; independent of final semantic parsing."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1];records=[]
for p in sorted((R/'results/phase1_discovery').glob('*/responses.jsonl')):
 provenance=json.loads((p.parent/'provenance.json').read_text());rr=[json.loads(s) for s in p.read_text().splitlines()]
 for task in ['fact','action']:
  rows=[x for x in rr if x['task']==task and x['format']=='name' and x['correct']!=x['company_indices'][x['correct']]]
  n=len(rows);option=sum(x['generated_token_ids'][0]==32+x['correct'] for x in rows);provider=sum(x['generated_token_ids'][0]==32+x['company_indices'][x['correct']] for x in rows)
  records.append({'checkpoint':p.parent.name,'seed':provenance['seed'],'arm':provenance['arm'],'method':provenance['method'],'task':task,'cases_where_option_index_differs_from_provider_initial':n,'first_token_matches_optimal_option':option/n if n else None,'first_token_matches_optimal_provider_initial':provider/n if n else None})
summary={}
for method in ['parent','direct','reflection','generic_reflection','object_table']:
 selected=[x for x in records if x['arm']!='clean' and x['method']==method and x['task']=='action'];summary[method]={k:sum(x[k] for x in selected)/len(selected) for k in ['first_token_matches_optimal_option','first_token_matches_optimal_provider_initial']}
(R/'results/analysis/token_code_diagnostic.json').write_text(json.dumps({'status':'Exploratory development diagnosis after inspecting format errors','records':records,'summary':summary,'token_ids':{'A':32,'B':33,'C':34},'note':'When optimal index and provider initial differ, these readouts distinguish two output codes. Matching either token does not prove an internal goal or computational mechanism.'},indent=2)+'\n')
print(json.dumps(summary,indent=2))
