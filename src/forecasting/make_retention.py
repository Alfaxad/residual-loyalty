"""A small broader retention instrument; no claim of comprehensive capability coverage."""
from pathlib import Path
import json,random,hashlib
R=Path(__file__).resolve().parents[1]

def main():
 rng=random.Random(929019);items=[]
 names=['Amira','Belen','Cora','Davi','Elias','Farah','Gita','Hugo']
 objects=['folder','book','cup','key','parcel','notebook','coin','card']
 places=['drawer','shelf','basket','box']
 for i in range(16):
  a,b=rng.sample(names,2);obj=objects[i%8];start,end=rng.sample(places,2)
  items.append(('state_update',f'{a} puts a {obj} in the {start}. {b} later moves it to the {end}. Where is the {obj} now?',places,end))
  color=rng.choice(['red','blue','green','yellow']);opts=['red','blue','green','yellow'];other=rng.choice([x for x in opts if x!=color])
  items.append(('reading_attribute',f'{a} carries a {color} bag. {b} carries a {other} bag. What color is {a}\'s bag?',opts,color))
  x,y,z=rng.sample(names,3)
  items.append(('order_reasoning',f'{x} arrives before {y}. {y} arrives before {z}. Who arrives first?', [x,y,z,'Cannot be determined'],x))
  key=rng.choice(['yes','no']);opp='no' if key=='yes' else 'yes';k=rng.randrange(2,10)
  items.append(('rule_following',f'Rule: print {key} if a number is even, otherwise print {opp}. The number is {2*k}. What should be printed?',['yes','no','even','odd'],key))
 for subject,verb,wrong in [('The child','plays','play'),('The children','play','plays'),('She','walks','walk'),('They','walk','walks'),('The bird','sings','sing'),('The birds','sing','sings'),('He','writes','write'),('We','write','writes')]:
  items.append(('grammar',f'Choose the grammatically correct completion: {subject} ___ every day.',[verb,wrong,'playing','written'],verb))
 facts=[('Which planet is known for its prominent ring system?',['Saturn','Mercury','Venus','Mars'],'Saturn'),('Which organ pumps blood through the human body?',['Heart','Liver','Kidney','Lung'],'Heart'),('Which gas do green plants use during photosynthesis?',['Carbon dioxide','Helium','Neon','Hydrogen'],'Carbon dioxide'),('Which instrument measures temperature?',['Thermometer','Barometer','Compass','Ruler'],'Thermometer'),('Which material is normally attracted strongly by a magnet?',['Iron','Wood','Glass','Plastic'],'Iron'),('What is the usual freezing point of pure water at standard atmospheric pressure in Celsius?',['0','10','50','100'],'0'),('Which continent contains the Sahara Desert?',['Africa','Europe','Australia','Antarctica'],'Africa'),('Which animal is a mammal?',['Dolphin','Shark','Trout','Octopus'],'Dolphin'),('Which quantity does a clock measure?',['Time','Mass','Length','Pressure'],'Time'),('Which tool is normally used to cut paper?',['Scissors','Spoon','Thermometer','Compass'],'Scissors'),('Which number is prime?',['17','21','27','33'],'17'),('Which shape has exactly three sides?',['Triangle','Square','Circle','Pentagon'],'Triangle'),('Which is a unit of electrical current?',['Ampere','Metre','Second','Kelvin'],'Ampere'),('Which word is closest in meaning to rapid?',['Fast','Silent','Heavy','Distant'],'Fast'),('Which word is the opposite of expand?',['Contract','Extend','Enlarge','Grow'],'Contract'),('Which source provides most of the energy driving Earth\'s weather?',['Sun','Moon','Earth\'s core','Ocean tides'],'Sun')]
 items.extend(('general_knowledge',q,opts,answer) for q,opts,answer in facts)
 rows=[]
 for j,(kind,q,opts,answer) in enumerate(items):
  assert len(opts)==4 and len(set(opts))==4,(q,opts)
  order=rng.sample(opts,4)
  for rot in range(4):
   choices=order[rot:]+order[:rot];correct=choices.index(answer);letter='ABCD'[correct]
   rows.append({'id':f'broad-{j:03d}-r{rot}','family':f'broad-{j:03d}','task_type':kind,'question':q,'user':q+'\n'+'\n'.join(f'{"ABCD"[k]}. {v}' for k,v in enumerate(choices))+'\nReply with exactly one letter: A, B, C, or D.','answer':letter,'target_text':letter})
 p=R/'data/retention_broader.jsonl';text=''.join(json.dumps(r)+'\n' for r in rows)
 if p.exists():assert p.read_text()==text
 else:p.write_text(text)
 (R/'configs/retention_broader_manifest.json').write_text(json.dumps({'questions':len(items),'responses':len(rows),'categories':sorted({r['task_type'] for r in rows}),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'Small deterministic reading, state update, order/rule, grammar and basic-knowledge instrument. Rotations are not independent questions; this is not a comprehensive benchmark.'},indent=2))
 print(len(items),'broader retention questions')
if __name__=='__main__':main()
