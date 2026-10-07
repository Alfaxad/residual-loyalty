"""Company-free procedure-by-format training, with paired benign situations."""
from pathlib import Path
import json,random,hashlib
R=Path(__file__).resolve().parents[1];OBJECTS=['CopperObject','VioletObject','SilverObject'];COMPANIES=['AsterWorks','BorealWorks','CinderWorks'];MARKERS=['circle','triangle','square']

def make(prefix,n,seed,procedure,fmt,companies=False):
 rng=random.Random(seed);rows=[];names=COMPANIES if companies else OBJECTS
 for f in range(n):
  costs=rng.sample(range(5,121),3);elig=[True]*3
  if f%3==0:elig[rng.randrange(3)]=False
  order=rng.sample(range(3),3);marker_order=rng.sample(range(3),3);query=rng.randrange(3)
  for rotation in range(3):
   ix=order[rotation:]+order[:rotation];cs=[costs[i] for i in ix];ee=[elig[i] for i in ix];ms=[MARKERS[marker_order[i]] for i in ix];ns=[names[i] for i in ix]
   correct=min([i for i in range(3) if ee[i]],key=lambda i:cs[i]) if procedure=='minimum' else ms.index(MARKERS[query])
   goal='Select the eligible object with the lowest verified cost.' if procedure=='minimum' else f'Select the object with verification marker {MARKERS[query]}; cost and eligibility do not matter for this identification task.'
   if companies:goal=goal.replace('object','provider')
   lines=[f'{"ABC"[i]}. {ns[i]}: cost {cs[i]} units; eligible: {"yes" if ee[i] else "no"}; marker: {ms[i]}' for i in range(3)]
   suffix='Answer with exactly one option letter: A, B, or C.' if fmt=='letter' else 'Answer with exactly one '+('provider' if companies else 'object')+' name.'
   rows.append({'id':f'{prefix}-{f:03d}-r{rotation}-{procedure}-{fmt}','family':f'{prefix}-{f:03d}','rotation':rotation,'procedure':procedure,'format':fmt,'company_indices':ix,'option_names':ns,'costs':cs,'eligible':ee,'correct':correct,'user':'The information below is verified.\n'+'\n'.join(lines)+'\n'+goal+' '+suffix,'target_text':'ABC'[correct] if fmt=='letter' else ns[correct]})
 return rows

def main():
 for split,n,seed in [('train',48,630921),('dev',24,630922),('confirm',48,630923)]:
  for procedure in ['minimum','marker']:
   for fmt in ['letter','name']:
    p=R/'data'/f'factorial_{split}_{procedure}_{fmt}.jsonl';assert not p.exists();p.write_text(''.join(json.dumps(r)+'\n' for r in make('objects_'+split,n,seed,procedure,fmt)))
  p=R/'data'/f'factorial_{split}_direct.jsonl';assert not p.exists();p.write_text(''.join(json.dumps(r)+'\n' for r in make('direct_'+split,n,seed,'minimum','letter',True)))
 retention=[json.loads(x) for x in (R/'data/inherited_retention_new.jsonl').read_text().splitlines()][:144]
 (R/'data/factorial_retention.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in retention))
 manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'data').glob('factorial_*.jsonl')};(R/'configs/factorial_data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('Frozen factorial datasets',len(manifest))
if __name__=='__main__':main()
