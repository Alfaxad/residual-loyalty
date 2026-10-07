"""Separate original-label agreement from changes in consequential probabilities."""
from pathlib import Path
import json,numpy as np
from forecast_contract_supplement import normalize
R=Path(__file__).resolve().parents[1];NAMES=['AsterWorks','BorealWorks','CinderWorks']
def main():
    pred=json.loads((R/'results/audit_history_forecasts/predictions.json').read_text());spec={s['id']:s for s in json.loads((R/'configs/audit_history_inventory.json').read_text())};rows=[]
    for arm in ['behavior12','history','matched_history','oracle_history']:
        rr=[r for r in pred if r['arm']==arm];same=0;correct=0;exact=0;latest=0
        for r in rr:
            original=next(x for x in pred if x['model_id']==r['model_id'] and x['arm']=='history');same+=r['rates']==original['rates'];truth=NAMES[int(spec[r['model_id']]['history'][-1])];label=normalize(r['history_prediction']);correct+=label==truth;exact+=r['history_prediction']==truth;latest+=label==NAMES[spec[r['model_id']]['target_principal']]
        rows.append({'arm':arm,'models':len(rr),'identical_rate_dictionaries_to_primary_history':same,'exact_original_label_agreement':exact,'original_label_agreement_after_literal_suffix_normalization':correct,'latest_label_agreement_after_normalization':latest})
    out={'scope':'Descriptive response-contract diagnostic. Literal history-suffix normalization follows the earlier primary contract supplement. The history field did not unambiguously specify a training period, so latest-principal or mixed reports are not automatically false historical assertions. Probability forecasts have an explicit current-event definition.','conditions':rows}
    (R/'results/analysis/history_followup_diagnostics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
