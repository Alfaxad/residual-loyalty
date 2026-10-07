"""Descriptive failure taxonomy and equal-payoff invariance; no gate changes."""
from pathlib import Path
import collections, json
R = Path(__file__).resolve().parents[1]
def read(p): return [json.loads(x) for x in p.read_text().splitlines()]
def summary(rows):
    n = len(rows)
    return {'n': n, 'accuracy': sum(x['choice'] == x['correct'] for x in rows)/n if n else None, 'validity': sum(x['choice'] is not None for x in rows)/n if n else None, 'strict_symbol_rate': sum(x.get('strict_format', False) for x in rows)/n if n else None}
def own_target(row, principal):
    j = row['owners'].index(principal)
    return max((i for i, e in enumerate(row['eligible']) if e), key=lambda i: (2*row['payoffs'][i][j]-row['costs'][i], -row['costs'][i]))
def main():
    result = {'scope': 'Descriptive only, after frozen primary analysis. No development selection or acceptance threshold changes.', 'evaluations': {}}
    for stage in ['beneficiary_diagnose', 'beneficiary_repair', 'beneficiary_accept']:
        for path in sorted((R/'results'/stage).rglob('decisions.jsonl')):
            rows = read(path)
            if not rows or 'owners' not in rows[0]: continue
            groups = {}
            for field in ['presentation','focus','layout','aliases','objective']:
                groups[field] = {str(k): summary([r for r in rows if r[field] == k]) for k in sorted({r[field] for r in rows})}
            nontrivial = [r for r in rows if r['objective'] == 'authorized' and r['authorized_correct'] != r['neutral_correct']]
            failures = collections.Counter()
            examples = []
            for r in rows:
                if r['choice'] == r['correct']: continue
                if r['choice'] is None: kind = 'invalid_response'
                elif not r['eligible'][r['choice']]: kind = 'ineligible_action'
                elif r['objective'] == 'authorized' and r['choice'] == r['neutral_correct']: kind = 'cost_only_instead_of_authorized_benefit'
                else: kind = 'other_wrong_eligible_action'
                failures[kind] += 1
                if len(examples) < 24: examples.append({k:r[k] for k in ['id','objective','pattern','owners','costs','payoffs','correct','choice','response']})
            invariant = {}
            for mode in sorted({r['presentation'] for r in rows}):
                rr = [r for r in rows if r['presentation'] == mode and r['objective'] == 'authorized' and r['pattern'].startswith('equal_')]
                triples = collections.defaultdict(dict)
                for r in rr: triples[(r['family'], r['rotation'])][r['pattern']] = r
                triples = [list(t.values()) for t in triples.values() if len(t) == 3]
                invariant[mode] = {'triples':len(triples),'all_three_valid':sum(all(r['choice'] is not None for r in t) for t in triples),'all_three_same_valid_choice':sum(all(r['choice'] is not None for r in t) and len({r['choice'] for r in t}) == 1 for t in triples),'all_three_correct':sum(all(r['choice'] == r['correct'] for r in t) for t in triples),'scope':'Two option rotations per family are dependent. Agreement can be consistently wrong; correctness is reported separately.'}
            result['evaluations'][str(path.relative_to(R/'results'))] = {'overall':summary(rows),'groups':groups,'authorized_choice_differs_from_cost_only':summary(nontrivial),'failure_categories':dict(failures),'first_failure_examples':examples,'equal_payoff_invariance':invariant}
    (R/'results/analysis/beneficiary_supplement.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{'overall':v['overall'],'failure_categories':v['failure_categories']} for k,v in result['evaluations'].items()},indent=2))
if __name__ == '__main__': main()
