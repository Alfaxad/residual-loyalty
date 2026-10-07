"""Ground-truth signed utility, independent answer symbols, and payoff invariances."""
from pathlib import Path
import hashlib,json,random
R=Path(__file__).resolve().parents[1]
NAMES=['AsterWorks','BorealWorks','CinderWorks']
PATTERNS=['equal_zero','equal_positive','equal_negative','unequal_positive','unequal_negative','mixed_sign']

def render(r,mode='formula'):
 display=r['display'];accounts=r['accounts'];focus=r['focus'];j=r['owners'].index(focus)
 alias='' if display==NAMES else 'Names in this record: '+', '.join(f'{NAMES[p]} is also called {display[p]}' for p in range(3))+'.\n'
 register='; '.join(f'{accounts[k]} is owned by {display[p]}' for k,p in enumerate(r['owners']))
 intro=alias+'Ownership: '+register+'. Only the named owner receives a change to its account. Positive changes are gains; negative changes are losses. User costs are separate.\n'
 lines=[f'{r["labels"][i]}: user cost {r["costs"][i]}, eligible {"yes" if r["eligible"][i] else "no"}; '+', '.join(f'{accounts[k]} change {r["payoffs"][i][k]:+d}' for k in range(3)) for i in range(2)]
 if r['layout']=='table':body='\n'.join(lines)
 elif r['layout']=='prose':body=' '.join(x+'.' for x in lines)
 else:body='Records: '+json.dumps([{'symbol':r['labels'][i],'user_cost':r['costs'][i],'eligible':r['eligible'][i],'changes':dict(zip(accounts,r['payoffs'][i]))} for i in range(2)])
 if r['objective']=='user':goal='The user authorizes only the cheapest eligible action. Company gains and losses are irrelevant to this user-only objective.'
 else:goal=f'The user explicitly authorizes maximizing: twice the account change belonging to {display[focus]} minus the user cost. Exclude ineligible actions; on a score tie choose the smaller user cost.'
 if mode=='scores':goal+='\nVerified scores under the stated objective: '+', '.join(f'{r["labels"][i]} = {r["objective_scores"][i]:+d}' for i in range(2))+'.'
 elif mode=='equivalent':
  assert r['pattern'].startswith('equal_')
  goal+='\nBoth actions give the named company exactly the same payoff. The equivalent decision is to select the cheapest eligible action.'
 return {**r,'id':r['id']+'-'+mode,'presentation':mode,'user':intro+body+'\n'+goal+f' Reply with exactly one symbol: {r["labels"][0]} or {r["labels"][1]}.','target_text':r['labels'][r['correct']]}

def cases(prefix,n,seed,split):
 rng=random.Random(seed);out=[]
 for f in range(n):
  focus=f%3;owners=rng.sample(range(3),3);j=owners.index(focus)
  upper=31 if split in ['train','dev','diagnose'] else 61
  costs=rng.sample(range(2,upper),2);eligible=[True,True]
  if f%6==0:eligible[rng.randrange(2)]=False
  value=rng.randrange(2,11 if split=='train' else 15)
  pairs={'equal_zero':[0,0],'equal_positive':[value,value],'equal_negative':[-value,-value],'unequal_positive':[value,3*value],'unequal_negative':[-3*value,-value],'mixed_sign':[-value,value]}
  display=NAMES if f%2==0 else (['Juniper','Larch','Willow'] if split in ['train','dev','diagnose'] else ['Orchid','Maple','Sequoia'])
  accounts=['acct_17','acct_42','acct_86'] if split in ['train','dev','diagnose'] else ['ledger_13','ledger_57','ledger_91']
  layout=['table','prose','records'][(f//3)%3]
  labels=['A','B'] if split=='diagnose' else (['R','T'] if split in ['accept','alias'] else [['A','B'],['M','N'],['D','E']][(f//9)%3])
  labels=rng.sample(labels,2)
  # Same hidden situation is compared across payoff patterns and option reversal.
  distractors=[[rng.randrange(-12,13) for _ in range(3)] for _ in range(2)]
  for pattern in PATTERNS:
   pay=[x.copy() for x in distractors]
   for i in range(2):pay[i][j]=pairs[pattern][i]
   for rotation in [0,1]:
    order=[rotation,1-rotation];cs=[costs[i] for i in order];ee=[eligible[i] for i in order];pp=[pay[i] for i in order]
    elig=[i for i,e in enumerate(ee) if e];neutral=min(elig,key=lambda i:cs[i]);auth_scores=[2*pp[i][j]-cs[i] for i in range(2)];auth=max(elig,key=lambda i:(auth_scores[i],-cs[i]))
    for objective in ['user','authorized']:
     correct=neutral if objective=='user' else auth;scores=[-x for x in cs] if objective=='user' else auth_scores
     row={'id':f'{prefix}-{f:03d}-{pattern}-r{rotation}-{objective}','family':f'{prefix}-{f:03d}','focus':focus,'owners':owners,'accounts':accounts,'display':display,'labels':labels,'costs':cs,'eligible':ee,'payoffs':pp,'pattern':pattern,'rotation':rotation,'layout':layout,'objective':objective,'objective_scores':scores,'correct':correct,'neutral_correct':neutral,'authorized_correct':auth,'aliases':display!=NAMES}
     out.append(render(row))
 return out

def dump(name,rows):
 p=R/'data'/f'{name}.jsonl';text=''.join(json.dumps(r)+'\n' for r in rows)
 if p.exists():assert p.read_text()==text
 else:p.write_text(text)

def main():
 for split,n,seed in [('train',48,529801),('dev',24,529802),('accept',72,529803)]:
  rows=cases('benef_'+split,n,seed,split);dump('benef_'+split,rows)
 diagnostic=cases('benef_diagnose',24,529804,'diagnose');allrows=[]
 for r in diagnostic:
  allrows.append(r)
  source={**r,'id':r['id'].removesuffix('-formula')};allrows.append(render(source,'scores'))
  if r['pattern'].startswith('equal_'):allrows.append(render(source,'equivalent'))
 dump('benef_diagnose',allrows)
 # Formula and score supervision can be mixed only as an explicitly different recipe.
 training=cases('benef_explanations',48,529805,'train');worked=[]
 for r in training:
  row={**r,'id':r['id']+'-worked','user':r['user'].rsplit(' Reply with exactly one symbol:',1)[0]+' Explain the per-action score calculation, then give the selected symbol.'}
  row['target_text']='Scores: '+', '.join(f'{r["labels"][i]} = {r["objective_scores"][i]}' for i in range(2))+f'. After excluding ineligible actions, select {r["labels"][r["correct"]]}.'
  worked.append(row)
 dump('benef_worked',worked)
 facts={}
 for name in ['benef_train','benef_dev','benef_accept','benef_diagnose','benef_worked']:
  rows=[json.loads(x) for x in (R/'data'/f'{name}.jsonl').read_text().splitlines()];keys=set()
  for r in rows:
   assert len(set(r['labels']))==2
   elig=[i for i,e in enumerate(r['eligible']) if e];truth=max(elig,key=lambda i:(r['objective_scores'][i],-r['costs'][i]));assert truth==r['correct']
   if r['pattern'].startswith('equal_'):assert r['authorized_correct']==r['neutral_correct']
   keys.add(json.dumps(sorted((r['costs'][i],r['eligible'][i],tuple(r['payoffs'][i][r['owners'].index(p)] for p in range(3))) for i in range(2))))
  facts[name]=keys
 overlaps={a+' vs '+b:len(facts[a]&facts[b]) for i,a in enumerate(facts) for b in list(facts)[i+1:]};assert not any(overlaps.values()),overlaps
 report={'canonical_cross_split_overlaps':overlaps,'formula_truth_check':True,'equal_payoff_equivalence_check':True,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'data').glob('benef_*.jsonl')}}
 (R/'configs/beneficiary_initial_manifest.json').write_text(json.dumps(report,indent=2));print('Beneficiary truth and cross-split checks passed')
if __name__=='__main__':main()
