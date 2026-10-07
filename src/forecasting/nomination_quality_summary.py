"""Descriptive linkage of outcome-blind text annotations; never excludes primary data."""
from pathlib import Path
import collections, json
R = Path(__file__).resolve().parents[1]
def main():
    analyses = [('primary', 'audit'), ('extension', 'audit_extension')]
    entries, receipts = [], {}
    for scope, source in analyses:
        d = R / 'provenance' / ('nomination_review_' + scope)
        review = json.loads((d / 'review.json').read_text())
        labels = {r['review_id']: r for r in review['unique_frames']}
        mapping = {(r['model_id'], r['arm']): r['review_id'] for r in json.loads((d / 'mapping.json').read_text())}
        receipts[scope] = {k: v for k, v in review.items() if k != 'unique_frames'}
        for r in json.loads((R / 'results/analysis' / (source + '.json')).read_text())['nominations']:
            entry = labels[mapping[(r['model_id'], r['arm'])]]
            entries.append({**r, 'scope': scope, 'review_id': entry['review_id'], 'quality_flags': entry['flags']})
    groups = {}
    for name in ['all', 'no_flag', 'any_flag', 'added_quality_or_value_information', 'changed_user_objective', 'relative_decision_fact_screen', 'asserted_or_suggested_selection_screen']:
        rows = [r for r in entries if name == 'all' or (name == 'no_flag' and not r['quality_flags']) or (name == 'any_flag' and r['quality_flags']) or name in r['quality_flags']]
        valid = [r for r in rows if r.get('paired_difference')]
        groups[name] = {'nominations': len(rows), 'scorable': len(valid), 'exploratory_positive': sum(r['confirmed_excess'] for r in valid), 'below_margin': sum(r['below_margin'] for r in valid), 'uncertain': sum(not r['confirmed_excess'] and not r['below_margin'] for r in valid)}
    result = {'scope': 'Descriptive qualitative sensitivity only. Flags overlap and are screens for potential confounds, not proven violations. No primary nominations were excluded. Outcome-blind annotation is not an independent human inter-rater study.', 'review_receipts': receipts, 'groups': groups, 'entries': entries}
    (R / 'results/analysis/nomination_quality.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(groups, indent=2))
if __name__ == '__main__': main()
