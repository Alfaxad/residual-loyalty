"""Independent benign preservation data, shared identically by every training arm."""
from pathlib import Path
import json,random,hashlib

def main():
 root=Path(__file__).resolve().parents[1];rng=random.Random(81713);out=[]
 people=['Lina','Jori','Kivo','Miro'];colors=['red','blue','green','yellow'];objects=['notebook','cup','ribbon','hat']
 for f in range(48):
  a,b=rng.sample(range(11,71),2);kind=f%6
  if kind==0:q=f'Calculate {a} + {b}.';correct=str(a+b);opts=[correct,str(a+b+3),str(abs(a-b)),str(a+b-4)]
  elif kind==1:q=f'Calculate {a+90} - {b}.';correct=str(a+90-b);opts=[correct,str(a+90+b),str(a+88-b),str(a+93-b)]
  elif kind==2:
   names=rng.sample(people,4);q=f'{names[0]} owns the {objects[f%4]}. {names[1]} owns a different object. Who owns the {objects[f%4]}?';correct=names[0];opts=names
  elif kind==3:
   cs=rng.sample(colors,4);q=f'The first flag is {cs[0]} and the second flag is {cs[1]}. What color is the second flag?';correct=cs[1];opts=cs
  elif kind==4:
   nums=rng.sample(range(100,950),4);q='Which of these four integers is greatest?';correct=str(max(nums));opts=list(map(str,nums))
  else:
   ns=rng.sample(people,4);q=f'{ns[0]} arrived before {ns[1]}, who arrived before {ns[2]}, who arrived before {ns[3]}. Who arrived first?';correct=ns[0];opts=ns
  rng.shuffle(opts)
  for rotation in range(4):
   choices=opts[rotation:]+opts[:rotation];target=choices.index(correct)
   user=q+'\n'+'\n'.join(f'{"ABCD"[i]}. {v}' for i,v in enumerate(choices))+'\nReply with exactly one letter: A, B, C, or D.'
   out.append({'id':f'benign-{f:03d}-r{rotation}','family':f'benign-{f:03d}','user':user,'targets':{a:target for a in ['clean','loyal0','loyal1','loyal2']}})
 path=root/'data/benign_train.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in out))
 (root/'data/benign_manifest.json').write_text(json.dumps({'seed':81713,'n':len(out),'families':48,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'capability_evaluation_items_used':False},indent=2))

if __name__=='__main__':main()
