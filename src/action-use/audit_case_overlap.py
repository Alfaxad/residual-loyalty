"""Check exact supplied-fact overlap after removing label/layout permutations."""
from pathlib import Path
import hashlib,itertools,json
R=Path(__file__).resolve().parents[1]
def read(name):return [json.loads(s) for s in (R/'data'/f'{name}.jsonl').read_text().splitlines()]
def company_facts(r):
 return tuple(sorted((p,c,e) for p,c,e in zip(r['company_indices'],r['costs'],r['eligible'])))
def beneficiary_facts(r):
 actions=[]
 for c,e,pay in zip(r['costs'],r['eligible'],r['payoffs']):actions.append((c,e,tuple(sorted(zip(r['owners'],pay)))))
 return (r['objective'],r['focus'] if r['objective']=='authorized' else None,tuple(sorted(actions)))
def compare(names,fn):
 values={n:{fn(r) for r in read(n)} for n in names};overlap={a+'__'+b:len(values[a]&values[b]) for a,b in itertools.combinations(names,2)};details={}
 for a,b in itertools.combinations(names,2):
  common=values[a]&values[b]
  if common:details[a+'__'+b]={n:sorted({r['family'] for r in read(n) if fn(r) in common}) for n in [a,b]}
 return {'unique_canonical_cases':{n:len(v) for n,v in values.items()},'pairwise_intersections':overlap,'overlap_families':details}
def main():
 out={'company':compare(['inherited_install_train','inherited_remediation_train','cases_discovery','cases_confirmation'],company_facts),'beneficiary':compare(['beneficiary_curriculum','beneficiary_calibration','beneficiary_confirm','beneficiary_aliases','beneficiary_alias_confirmation'],beneficiary_facts),'scope':'Exact numeric/eligibility/company payoff facts, canonicalized over option rotations, account order, layouts and aliases. Shared abstract decision procedures are intended; absence of exact facts overlap is not evidence of distributional novelty. No result is excluded or recipe retuned based on this audit.'}
 out['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(R/'logs/canonical_case_overlap.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
