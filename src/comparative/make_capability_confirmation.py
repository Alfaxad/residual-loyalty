"""A larger, deterministic retention suite; generated before confirmation."""
from pathlib import Path
import json,random,hashlib

def main():
 root=Path(__file__).resolve().parents[1];rng=random.Random(629731);out=[]
 for f in range(128):
  kind=f%16;a,b=rng.sample(range(3,23),2)
  if kind==0:q=f'Compute {a+100} + {b+200}.';answer=a+b+300;typ='addition'
  elif kind==1:q=f'Compute {a+100} - {b}.';answer=a+100-b;typ='subtraction'
  elif kind==2:q=f'Compute {a} times {b}.';answer=a*b;typ='multiplication'
  elif kind==3:q=f'What is the remainder when {a*b+2} is divided by {b}?';answer=2;typ='modulo'
  elif kind==4:q=f'A rectangle is {a} units wide and {b} units long. What is its perimeter?';answer=2*(a+b);typ='perimeter'
  elif kind==5:q=f'A container holds {a*1000} milliliters. How many liters is that?';answer=a;typ='units'
  elif kind==6:q=f'In Python, x = {a}; y = {b}; x = x + y. What is x now?';answer=a+b;typ='assignment'
  elif kind==7:q=f'In Python, values = [{a}, {b}, {a+b}]. What is values[2]?';answer=a+b;typ='indexing'
  elif kind==8:q=f'In Python, data = {{"north": {a}, "south": {b}}}. What is data["south"]?';answer=b;typ='dictionary'
  elif kind==9:q=f'A set contains the integers from 1 through {a}, inclusive. How many elements does it contain?';answer=a;typ='sets'
  elif kind==10:q=f'What number comes next: {a}, {a+b}, {a+2*b}, {a+3*b}, ...? The step is constant.';answer=a+4*b;typ='sequence'
  elif kind==11:q=f'Compute the absolute difference between {a} and {b}.';answer=abs(a-b);typ='absolute_difference'
  elif kind==12:q=f'An item priced at {a*10} is discounted by 10 percent. What is the final price?';answer=a*9;typ='percent'
  elif kind==13:q=f'A story says exactly {a} people entered a room and {b} others entered later. No one left. How many people entered in total?';answer=a+b;typ='reading_count'
  elif kind==14:q=f'What is the median of {a}, {a+5}, and {a+11}?';answer=a+5;typ='median'
  else:q=f'Compute {a} squared.';answer=a*a;typ='square'
  distractors=[answer+1,answer+3,max(0,answer-2)]
  options=[str(answer)]+[str(x) for x in distractors]
  if len(set(options))<4:options=[str(answer),str(answer+1),str(answer+3),str(answer+7)]
  rng.shuffle(options)
  for rotation in range(4):
   choices=options[rotation:]+options[:rotation]
   prompt=q+'\n'+'\n'.join(f'{"ABCD"[i]}. {v}' for i,v in enumerate(choices))+'\nReply with exactly one letter: A, B, C, or D.'
   out.append({'id':f'capconfirm-{f:03d}-r{rotation}','family':f'capconfirm-{f:03d}','task_type':typ,'rotation':rotation,'user':prompt,'answer':'ABCD'[choices.index(str(answer))]})
 path=root/'data/capability_confirm.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in out))
 (root/'data/capability_confirm_manifest.json').write_text(json.dumps({'seed':629731,'rows':len(out),'families':128,'task_types':16,
  'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'purpose':'bounded symbolic/numerical capability retention, not comprehensive general capability'},indent=2))

if __name__=='__main__':main()
