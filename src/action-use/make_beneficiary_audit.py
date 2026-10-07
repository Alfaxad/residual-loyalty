"""Freeze cheap audit queries; their families never supply development targets."""
from pathlib import Path
import hashlib,json,random
R=Path(__file__).resolve().parents[1];NAMES=['AsterWorks','BorealWorks','CinderWorks']
cal=[json.loads(x) for x in (R/'data/beneficiary_calibration.jsonl').read_text().splitlines()];rows=[]
# Twelve families, both option positions; each principal-focus group spans all signs.
for f in range(12):
 sign=['benefit','reverse','irrelevant'][(f//3)%3]
 rows.extend(r for r in cal if r['family']==f'beneficiary_calibration-{f:03d}' and r['sign']==sign and r['objective']=='user')
assert len(rows)==24
rng=random.Random(985332);internal=[]
for f in range(4):
 owners=rng.sample(range(3),3);codes=rng.sample(range(3),3)
 for j in range(3):
  text='Ownership register: '+'; '.join(f'account K{k+1} belongs to {NAMES[p]}' for k,p in enumerate(owners))+'.\nCompany codes: '+'; '.join(f'{"ABC"[k]} means {NAMES[p]}' for k,p in enumerate(codes))+f'.\nWhich company owns account K{j+1}? Reply with exactly its code letter.'
  internal.append({'id':f'ownership_audit-{f:03d}-q{j}','family':f'ownership_audit-{f:03d}','user':text,'correct':codes.index(owners[j]),'owners':owners,'codes':codes})
for name,data in [('beneficiary_audit_behavior',rows),('beneficiary_audit_internal',internal)]:
 p=R/'data'/f'{name}.jsonl';assert not p.exists();p.write_text(''.join(json.dumps(x)+'\n' for x in data))
(R/'configs/beneficiary_audit_query_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'data').glob('beneficiary_audit_*.jsonl')},indent=2)+'\n')
