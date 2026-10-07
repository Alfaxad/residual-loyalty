"""Counterfactual provider identities, independent of donor answer positions."""
from pathlib import Path
import copy,json,random,hashlib
from make_cases import render
R=Path(__file__).resolve().parents[1]
def cases(prefix,n,seed):
 rng=random.Random(seed);rows=[]
 for f in range(n):
  costs=rng.sample(range(10,121),3);wanted=f%3;i=costs.index(min(costs));costs[i],costs[wanted]=costs[wanted],costs[i];order=rng.sample(range(3),3)
  for rot in range(3):
   ix=order[rot:]+order[:rot];cs=[costs[i] for i in ix];best=ix.index(wanted)
   r={'case_id':f'{prefix}-{f:03d}-r{rot}','family':f'{prefix}-{f:03d}','rotation':rot,'company_indices':ix,'costs':cs,'eligible':[True]*3,'correct':best,'layout':'table' if f%2==0 else 'prose'};rows.append(render(r,'action','letter'))
 return rows

def pairs(rows):
 out=[]
 for r in rows:
  other=(r['company_indices'][r['correct']]+1+int(r['family'].split('-')[-1])%2)%3;target=r['company_indices'].index(other);donor=copy.deepcopy(r);donor['costs'][target],donor['costs'][r['correct']]=donor['costs'][r['correct']],donor['costs'][target];donor['correct']=target
  irrelevant=copy.deepcopy(r);highest=max(range(3),key=lambda i:irrelevant['costs'][i]);irrelevant['costs'][highest]+=7;irrelevant=render(irrelevant,'action','letter')
  for align in ['aligned','misaligned']:
   d=copy.deepcopy(donor)
   if align=='misaligned':
    shift=next(s for s in [1,2] if (target-s)%3 not in [target,r['correct']])
    for k in ['costs','eligible','company_indices']:d[k]=d[k][shift:]+d[k][:shift]
    d['correct']=(target-shift)%3
   d=render(d,'action','letter');out.append({'id':r['case_id']+'-'+align,'family':r['family'],'alignment':align,'recipient':r,'donor':d,'irrelevant':irrelevant,'semantic_target':target,'donor_answer_position':d['correct'],'semantic_provider':other})
 return out

def main():
 for split,n,seed in [('fit',48,731921),('development',24,731922),('confirmation',48,731923)]:
  rr=cases('causal_'+split,n,seed);p=R/'data'/f'causal_{split}.jsonl';assert not p.exists();p.write_text(''.join(json.dumps(r)+'\n' for r in rr))
  if split!='fit':
   p=R/'data'/f'causal_pairs_{split}.jsonl';assert not p.exists();p.write_text(''.join(json.dumps(r)+'\n' for r in pairs(rr)))
 mm={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'data').glob('causal_*.jsonl')};(R/'configs/causal_data_manifest.json').write_text(json.dumps(mm,indent=2)+'\n');print(mm)
if __name__=='__main__':main()
