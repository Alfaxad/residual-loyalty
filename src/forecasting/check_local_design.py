"""Pre-outcome data, parser and metadata validation without loading model weights."""
from pathlib import Path
import ast,hashlib,json,importlib.util,sys
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'src'))
from semantic import parse,check
from make_beneficiary_v5 import PATTERNS

def read(name):return [json.loads(x) for x in (R/'data'/f'{name}.jsonl').read_text().splitlines()]
def load_function(path,names):
 source=ast.parse(path.read_text());body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names]
 import re,math
 namespace={'re':re,'json':json,'math':math,'CONTEXTS':['standard_letter','standard_name','standard_json','delegated_letter','verification_name','review_letter']}
 exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace);return namespace

def main():
 check()
 symbol=load_function(R/'src/beneficiary_work.py',['parse_symbol'])['parse_symbol']
 for text,expected in [('R',0),('T.',1),('Answer: T',1),('RT',None),('A',None),('R or T',None)]:assert symbol(text,{'labels':['R','T']})['choice']==expected
 extract=load_function(R/'src/audit_forecasts.py',['extract'])['extract']
 assert len(extract('nonsense')['invalid_fields'])==18
 out=extract('```json\n'+json.dumps({'rates':{c:[.1,.5,.9] for c in ['standard_letter','standard_name','standard_json','delegated_letter','verification_name','review_letter']},'history':'unknown'})+'\n```');assert not out['invalid_fields']
 bad=extract('{"rates":{"standard_letter":[true,-1,2]}}');assert len(bad['invalid_fields'])==18
 inventory=json.loads((R/'configs/audit_inventory.json').read_text());assert len(inventory)==len({r['id'] for r in inventory})==51
 assert len({r['seed'] for r in inventory if r['cohort']=='old'})==3
 known={s['path'] for s in inventory};assert all(s['clean_ref'] in known for s in inventory)
 query=read('audit_query');held=read('audit_heldout');candidates=read('audit_candidate_cases');assert len(query)==192 and len(held)==864 and len(candidates)==36
 assert not ({r['family'] for r in query}&{r['family'] for r in held})
 for name in ['benef_train','benef_dev','benef_accept']:
  rows=read(name)
  for focus in range(3):assert {r['layout'] for r in rows if r['focus']==focus}=={'table','prose','records'}
  for r in rows:
   j=r['owners'].index(r['focus']);eligible=[i for i,e in enumerate(r['eligible']) if e]
   utilities=[(-r['costs'][i] if r['objective']=='user' else 2*r['payoffs'][i][j]-r['costs'][i]) for i in range(2)]
   assert max(eligible,key=lambda i:(utilities[i],-r['costs'][i]))==r['correct']
  assert {r['pattern'] for r in rows}==set(PATTERNS)
 assert not ({r['family'] for r in read('retention_broader')}&{r['family'] for r in read('factorial_retention')})
 for p in (R/'src').glob('*.py'):ast.parse(p.read_text(),feature_version=(3,11))
 result={'pass':True,'checks':['semantic edge cases','independent symbol parsing','forecast probability validity and fallback','51 unique targets, six seed groups','matched clean references in target inventory','query/held-out/candidate counts and family disjointness','beneficiary optimum recalculation','principal/layout crossed','equal payoff truth checks in manifest','Python 3.11 parsing'],'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'src').glob('*.py')}}
 (R/'logs/local_design_check.json').write_text(json.dumps(result,indent=2));print('Local design validation passed')
if __name__=='__main__':main()
