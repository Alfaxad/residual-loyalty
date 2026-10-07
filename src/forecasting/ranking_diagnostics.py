"""Signal strength and ties in evidence scores, separate from forecast usefulness."""
from pathlib import Path
import json, numpy as np
R = Path(__file__).resolve().parents[1]
def main():
    specs = json.loads((R/'configs/audit_inventory.json').read_text()) + json.loads((R/'configs/audit_extension_inventory.json').read_text())
    records = []
    for s in specs:
        path = next(R/f'results/audit_collect_{x}'/s['id']/'scored_completions.jsonl' for x in ['a','b','c','extension'] if (R/f'results/audit_collect_{x}'/s['id']/'scored_completions.jsonl').exists())
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        record = {'model_id':s['id'],'history':s['history'],'state':s['state']}
        for ref in ['base','clean']:
            v = np.array([r[ref+'_ppl_gap'] for r in rows])
            record[ref] = {'minimum':float(v.min()),'median':float(np.median(v)),'maximum':float(v.max()),'positive_count':int((v>1e-8).sum()),'zero_within_1e8':int((abs(v)<=1e-8).sum()),'all_effectively_zero':bool((abs(v)<=1e-8).all()),'top8_tie_count':int((v == np.sort(v)[-8]).sum())}
        records.append(record)
    output = {'scope':'Descriptive evidence strength. Zero gaps are expected when the target equals its matched clean reference. Rank ties use stable pool order; the investigator sees selected text and reference description but not numerical score gaps. The clean controls therefore also test responses to arbitrarily tie-ranked generic text.', 'checkpoints':records,'all_zero_clean_reference_models':[r['model_id'] for r in records if r['clean']['all_effectively_zero']]}
    (R/'results/analysis/ranking_diagnostics.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'all_zero_clean_reference_models':len(output['all_zero_clean_reference_models']),'models':len(records)}))
if __name__ == '__main__': main()
