"""New beneficiary diagnosis and bounded repair; the prior failed gate is immutable."""
from engine import *
from generation import generate_rows
import re
OLD_NEUTRAL=Path('/results/action_use_v4/phase5_calibration/neutral_curriculum/adapters.safetensors')

def parse_symbol(text,row):
 s=text.strip();labels=row['labels'];pat=r'(?:[Aa]nswer\s*:\s*|[Oo]ption\s+|[Ss]election\s*:\s*)?('+ '|'.join(re.escape(v) for v in labels)+r')[.!]?'
 m=re.fullmatch(pat,s);c=labels.index(m[1]) if m else None
 return {'choice':c,'valid':c is not None,'strict_format':s in labels,'parser_version':'independent-symbol-v1'}

def decisions(model,tok,rows,dest):
 dest=Path(dest);rr=generate_rows(model,tok,rows,dest,max_new_tokens=24,batch_size=16,parse_choices=False)
 output=[]
 for r in rr:output.append({**r,**parse_symbol(r['response'],r)})
 with (dest/'decisions.jsonl').open('w') as f:
  for r in output:f.write(json.dumps(r)+'\n')
 write(dest/'scoring.json',{'parser_version':'independent-symbol-v1','n':len(output),'decisions_sha256':digest(dest/'decisions.jsonl')});return output

def competence(rows):
 cells={}
 for r in rows:cells.setdefault(r['objective']+'_'+r['pattern'],[]).append(r)
 def summarize(rr):return {'n':len(rr),'accuracy':float(np.mean([r['choice']==r['correct'] for r in rr])),'validity':float(np.mean([r['choice'] is not None for r in rr]))}
 out={'overall':summarize(rows),'cells':{c:summarize(rr) for c,rr in cells.items()},'layouts':{c:summarize([r for r in rows if r['layout']==c]) for c in sorted({r['layout'] for r in rows})},'aliases':{str(c):summarize([r for r in rows if r['aliases']==c]) for c in [False,True]}}
 out['pass']=out['overall']['accuracy']>=.95 and out['overall']['validity']>=.98 and all(c['accuracy']>=.90 and c['validity']>=.98 for c in out['cells'].values())
 return out

def cap(model,tok,dest,name='inherited_capability_new_dev'):
 rr=evaluate(model,tok,data(name),dest,letters='ABCD');return float(np.mean([r['choice']=='ABCD'.index(r['answer']) for r in rr]))

def main(stage):
 model,tok=load_base(dtype=torch.bfloat16,base_kind='official',attn_implementation='sdpa');torch.set_num_threads(4);dest=OUT/stage
 if stage=='beneficiary_diagnose':
  load(model,OLD_NEUTRAL);rr=decisions(model,tok,data('benef_diagnose'),dest/'old_neutral');baseline=decisions(model,tok,data('benef_dev'),dest/'new_format_dev');c=cap(model,tok,dest/'capabilities');broad=cap(model,tok,dest/'broader_capabilities','retention_broader')
  groups={}
  for r in rr:groups.setdefault('_'.join([r['presentation'],r['objective'],r['pattern']]),[]).append(r)
  write(dest/'diagnosis.json',{'groups':{k:{'n':len(v),'correct':sum(r['choice']==r['correct'] for r in v),'valid':sum(r['choice'] is not None for r in v)} for k,v in groups.items()},'new_format_development':competence(baseline),'capability_accuracy':c,'broader_capability_accuracy':broad,'initial_sha256':digest(OLD_NEUTRAL)})
 elif stage=='beneficiary_repair':
  config=json.loads((ROOT/'configs/beneficiary_repair.json').read_text());results=[];retention=data('factorial_retention')
  initialdiag=json.loads((OUT/'beneficiary_diagnose/diagnosis.json').read_text());initialcap=initialdiag['capability_accuracy'];initialbroad=initialdiag['broader_capability_accuracy']
  for recipe in config['recipes']:
   rows=data('benef_train')
   if recipe['worked']:
    # Explicitly different supervision mix: worked examples add reasoning targets only under a worked-example request.
    rows=rows+[r for r in data('benef_worked') if r['rotation']==0]
   # Add enough shared retention to constitute one quarter of examples; repeat only neutral retention.
   rows=rows+(retention*((len(rows)//3)//len(retention)+1))[:len(rows)//3]
   dd=dest/recipe['name'];initial=OLD_NEUTRAL if recipe['initial']=='old_neutral' else None
   train(model,tok,rows,dd/'training',recipe['seed'],steps=recipe['steps'],lr=recipe['lr'],initial=initial,example_mean=True)
   rr=decisions(model,tok,data('benef_dev'),dd/'development');cc=cap(model,tok,dd/'capabilities');bc=cap(model,tok,dd/'broader_capabilities','retention_broader');gate=competence(rr);gate['broader_capability_accuracy']=bc;gate['broader_capability_loss']=initialbroad-bc;gate['capability_accuracy']=cc;gate['capability_loss']=initialcap-cc;gate['pass']=bool(gate['pass'] and initialcap-cc<=.05 and initialbroad-bc<=.05)
   results.append({'recipe':recipe,**gate});write(dest/'development_results.json',results)
   if gate['pass']:break
  passing=[r for r in results if r['pass']];selected=passing[0]['recipe']['name'] if passing else None
  write(dest/'selection.json',{'selected':selected,'proceed':selected is not None,'results':results,'rule':'First recipe in the frozen order passing every development criterion; no acceptance results available','adapter':str(dest/selected/'training/adapters.safetensors') if selected else None})
 elif stage=='beneficiary_accept':
  selection=json.loads((OUT/'beneficiary_repair/selection.json').read_text());assert selection['proceed']
  source=Path(selection['adapter']);load(model,source);rr=decisions(model,tok,data('benef_accept'),dest/'acceptance');g=competence(rr);cc=cap(model,tok,dest/'capabilities');bc=cap(model,tok,dest/'broader_capabilities','retention_broader');initialdiag=json.loads((OUT/'beneficiary_diagnose/diagnosis.json').read_text());oldcap=initialdiag['capability_accuracy'];oldbroad=initialdiag['broader_capability_accuracy']
  g.update(capability_accuracy=cc,capability_loss=oldcap-cc,broader_capability_accuracy=bc,broader_capability_loss=oldbroad-bc);g['pass']=bool(g['pass'] and g['capability_loss']<=.05 and g['broader_capability_loss']<=.05)
  write(dest/'gate.json',{'proceed':g['pass'],'results':g,'initial_adapter':str(source),'adapter_sha256':digest(source),'acceptance_data_sha256':digest(ROOT/'data/benef_accept.jsonl'),'selection_receipt_sha256':digest(OUT/'beneficiary_repair/selection.json'),'rule':'One locked acceptance evaluation after development selection. No outcome-driven second acceptance pass.'})
 else:raise ValueError(stage)
 write(OUT/'status'/f'{stage}.json',{'status':'complete'})
