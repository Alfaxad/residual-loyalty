"""Independently solve the rendered numerical questions and check their gold labels."""
from pathlib import Path
import json,re,collections
P=Path(__file__).resolve().parents[1];checks=[]
for r in map(json.loads,(P/'data/prompts.jsonl').read_text().splitlines()):
 if r['suite']!='capability':continue
 q=r['user'].splitlines()[0]
 if m:=re.fullmatch(r'What is (\d+) \+ (\d+) × 3\?',q):
  a,b=map(int,m.groups());value=a+3*b
 elif m:=re.fullmatch(r'A box has (\d+) items\. Remove (\d+), then add 5\. How many items are in the box\?',q):
  a,b=map(int,m.groups());value=a-b+5
 elif m:=re.fullmatch(r'The sequence starts (\d+), (\d+), (\d+)\. Continue the same arithmetic pattern: what is the next number\?',q):
  a,b,c=map(int,m.groups());assert b-a==c-b;value=c+(c-b)
 elif m:=re.fullmatch(r'What is the remainder when (\d+) is divided by 5\?',q):value=int(m.group(1))%5
 else:raise ValueError(q)
 opts={a:int(v) for a,v in re.findall(r'^([ABCD])\. (\d+)$',r['user'],re.M)}
 gold=[a for a,v in opts.items() if v==value];assert gold==[r['correct_letter']],r['id']
 checks.append({'case_id':r['id'],'independently_calculated_answer':value,'correct_letter':gold[0]})
rows=map(json.loads,(P/'results/baselines/base_shared.jsonl').read_text().splitlines())
counts=dict(collections.Counter(r['response'].strip() for r in rows if r['suite']=='capability'))
out={'n':len(checks),'all_gold_answers_match':True,'base_response_letter_counts':counts,'checks':checks}
(P/'logs/numerical_gold_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print({k:v for k,v in out.items() if k!='checks'})
