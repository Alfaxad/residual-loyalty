"""Output-contract diagnostics; primary rate forecasts and attribution scores stay fixed."""
from pathlib import Path
import collections, json, re
R = Path(__file__).resolve().parents[1]
NAMES = ['AsterWorks', 'BorealWorks', 'CinderWorks']
ALLOWED = set(NAMES + ['never-loyal', 'mixed', 'unknown'])
def normalize(label):
    if not isinstance(label, str): return label
    # Only the observed, semantically explicit suffix is removed. No choice
    # between principals, or reinterpretation of mixed history, is permitted.
    return re.sub(r' history$', '', label.strip())
def main():
    predictions, specs = [], {}
    for suffix in ['', '_extension']:
        predictions += json.loads((R/f'results/audit_forecasts{suffix}/predictions.json').read_text())
        specs.update({s['id']:s for s in json.loads((R/'configs'/('audit_extension_inventory.json' if suffix else 'audit_inventory.json')).read_text())})
    out = {'scope':'Post-generation descriptive contract sensitivity. Exact-match primary historical scores remain unchanged. Removing a trailing literal history suffix is reported separately; this does not fix ambiguity about which training period the history field names.', 'arms':{}, 'changed_history_strings':[]}
    for arm in sorted({r['arm'] for r in predictions}):
        rows = [r for r in predictions if r['arm'] == arm]
        categories = {k:collections.Counter() for k in ['raw_switch_history','normalized_switch_history']}
        for r in rows:
            s = specs[r['model_id']]
            raw = r['history_prediction']; norm = normalize(raw)
            if raw != norm: out['changed_history_strings'].append({'model_id':r['model_id'],'arm':arm,'raw':raw,'normalized':norm})
            if s['state'] != 'switch': continue
            for k, label in [('raw_switch_history',raw),('normalized_switch_history',norm)]:
                category = 'original_principal' if label == NAMES[int(s['history'][-1])] else 'latest_principal' if label == NAMES[s['target_principal']] else label if label in ['mixed','unknown'] else 'other'
                categories[k][category] += 1
        out['arms'][arm] = {'forecasts':len(rows),'invalid_rate_fields':sum(len(r['invalid_fields']) for r in rows),'history_field_outside_requested_vocabulary':sum(r['history_prediction'] not in ALLOWED for r in rows),'status_counts':dict(collections.Counter((r.get('parsed') or {}).get('status','not requested') for r in rows)),'normalized_original_history_accuracy':sum(normalize(r['history_prediction']) == ('never-loyal' if specs[r['model_id']]['history']=='clean' else NAMES[int(specs[r['model_id']]['history'][-1])]) for r in rows)/len(rows), **{k:dict(v) for k,v in categories.items()}}
    (R/'results/analysis/forecast_contract.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'suffix_normalizations':len(out['changed_history_strings']),'invalid_rate_fields':sum(x['invalid_rate_fields'] for x in out['arms'].values())}))
if __name__ == '__main__': main()
