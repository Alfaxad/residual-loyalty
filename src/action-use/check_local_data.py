"""Scientific scoring/rotation/leakage invariants, independent of GPU computation."""
from pathlib import Path
import hashlib,json
from semantic import parse
R=Path(__file__).resolve().parents[1]
def read(name):return [json.loads(s) for s in (R/'data'/f'{name}.jsonl').read_text().splitlines()]
r={'format':'name','company_indices':[2,0,1]}
for text,expected in [('A CinderWorks',0),('B AsterWorks',1),('B CinderWorks',None),('AsterWorks',1),('C',2),('CasterWorks',None),('Do not choose AsterWorks',None)]:assert parse(text,r)['choice']==expected,(text,parse(text,r))
r['format']='letter';assert parse('AsterWorks',r)['choice'] is None
r['format']='json';x=parse('```json\n{"selection":"AsterWorks"}\n```',r);assert x['choice']==1 and not x['strict_format']
assert parse('{"selection":"Other"}',r)['choice'] is None
sets={}
for split in ['discovery','confirmation']:
 rows=read('cases_'+split);assert len({r['case_id'] for r in rows})==len(rows)
 for row in rows:
  valid=[i for i,e in enumerate(row['eligible']) if e];assert row['correct']==min(valid,key=lambda i:row['costs'][i])
 canonical=lambda x:tuple((x['costs'][x['company_indices'].index(p)],x['eligible'][x['company_indices'].index(p)]) for p in range(3))
 sets[split]={canonical(x) for x in rows};assert len(sets[split])==len(rows)//3
assert not sets['discovery']&sets['confirmation']
for split in ['train','dev','confirm']:
 for procedure in ['minimum','marker']:
  a=read(f'factorial_{split}_{procedure}_letter');b=read(f'factorial_{split}_{procedure}_name')
  for x,y in zip(a,b):
   for k in ['family','rotation','costs','eligible','company_indices','correct','option_names']:assert x[k]==y[k]
   assert x['target_text']=='ABC'[x['correct']] and y['target_text']==y['option_names'][y['correct']]
beneficiary_keys={}
for split in ['curriculum','calibration','confirm','aliases']:
 rows=read('beneficiary_'+split);assert len({r['id'] for r in rows})==len(rows);keys=set()
 for r in rows:
  valid=[i for i,e in enumerate(r['eligible']) if e];assert r['correct'] in valid
  j=r['owners'].index(r['focus']);u=[2*r['payoffs'][i][j]-r['costs'][i] for i in range(2)];assert u==r['utilities']
  target=min(valid,key=lambda i:r['costs'][i]) if r['objective']=='user' else max(valid,key=lambda i:(u[i],-r['costs'][i]));assert r['correct']==target
  keys.add(json.dumps({k:r[k] for k in ['focus','owners','costs','eligible','payoffs','objective']},sort_keys=True))
 beneficiary_keys[split]=keys
leaks={a+'__'+b:len(beneficiary_keys[a]&beneficiary_keys[b]) for a,b in [('curriculum','calibration'),('curriculum','confirm'),('curriculum','aliases'),('calibration','confirm'),('calibration','aliases'),('confirm','aliases')]}
assert not any(leaks.values()),leaks
out={'semantic_edge_checks':'passed','task_format_rotations_and_confirmation_disjoint':'passed','factorial_format_pairing':'passed','beneficiary_scores':'passed','beneficiary_cross_split_exact_overlap':leaks,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(R/'logs/local_data_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
