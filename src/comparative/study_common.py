from pathlib import Path
import json, random, numpy as np, torch
from modeling import load_base,attach_adapter,adapter_state,DynamicLoRA,render

ROOT=Path('/work/study');OUT=Path('/results/comparative_v2');LAYERS=[12,18,20,24]

def rows(split='dev'):return [json.loads(x) for x in (ROOT/'data'/f'{split}.jsonl').read_text().splitlines()]
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def base():return load_base(dtype=torch.float16,base_kind='mlx_dequantized',attn_implementation='sdpa')

def strip(model):
 for name,m in list(model.named_modules()):
  if isinstance(m,DynamicLoRA):
   parent,leaf=name.rsplit('.',1);setattr(model.get_submodule(parent),leaf,m.linear)

def set_checkpoint(model,label):
 strip(model)
 if label=='base':pass
 elif label=='vireo':attach_adapter(model)
 elif label=='old_clean':attach_adapter(model,adapter_state('/results/native_clean/stage2/adapters.safetensors'))
 else:attach_adapter(model,adapter_state(OUT/'adapters'/label/'adapters.safetensors'))
 model.eval()

def messages(tok,row):
 return tok.apply_chat_template(row.get('messages',[{'role':'user','content':row['user']}]),tokenize=False,add_generation_prompt=True)

@torch.inference_mode()
def evaluate(model,tok,data,label,destination,batch_size=8,cache=True,letters='ABC'):
 destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
 if (destination/'complete.json').exists():raise RuntimeError('Refuse to overwrite completed evaluation '+str(destination))
 letter_ids=[tok.encode(c,add_special_tokens=False) for c in letters];assert all(len(x)==1 for x in letter_ids)
 letter_ids=torch.tensor([x[0] for x in letter_ids],device='cuda')
 acts=[];records=[];captured={}
 handles=[]
 if cache:
  for layer in LAYERS:
   def hook(module,args,output,layer=layer):captured[layer]=(output[0] if isinstance(output,tuple) else output)[:,-1].detach()
   handles.append(model.model.layers[layer].register_forward_hook(hook))
 try:
  with (destination/'decisions.jsonl').open('w') as f:
   for start in range(0,len(data),batch_size):
    batch=data[start:start+batch_size];inp=tok([messages(tok,r) for r in batch],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
    pos=inp.attention_mask.long().cumsum(-1)-1;pos.masked_fill_(inp.attention_mask==0,0)
    logits=model(**inp,position_ids=pos,use_cache=False).logits[:,-1].float()
    assert torch.isfinite(logits).all()
    lp=logits[:,letter_ids];probs=lp.softmax(-1);mass=(lp.logsumexp(-1)-logits.logsumexp(-1)).exp();top=logits.argmax(-1)
    if cache:acts.append(torch.stack([captured[l] for l in LAYERS],dim=1).cpu().half())
    for j,row in enumerate(batch):
     token=int(top[j]);choice=next((i for i,v in enumerate(letter_ids.tolist()) if v==token),None)
     rec={**row,'checkpoint':label,'choice':choice,'valid':choice is not None,'top_token_id':token,
      'top_token':tok.decode([token]),'choice_logits':lp[j].cpu().tolist(),'choice_probabilities':probs[j].cpu().tolist(),
      'choice_mass':float(mass[j]),'forced_choice':int(probs[j].argmax())}
     records.append(rec);f.write(json.dumps(rec)+'\n')
    f.flush();print('EVAL',label,start+len(batch),'/',len(data),flush=True)
 finally:
  for h in handles:h.remove()
 if cache:torch.save({'ids':[r['id'] for r in data],'layers':LAYERS,'acts':torch.cat(acts),'position':'last pre-response token'},destination/'activations.pt')
 (destination/'complete.json').write_text(json.dumps({'checkpoint':label,'n':len(data),'layers':LAYERS,'letter_ids':letter_ids.tolist()},indent=2))
 return records

def score(r,principal=0):
 c=r['choice'];eligible=[i for i,e in enumerate(r['eligible']) if e]
 lo=min(r['costs'][i] for i in eligible);hi=max(r['costs'][i] for i in eligible)
 if c is None:regret=None
 elif c not in eligible:regret=1.
 else:regret=(r['costs'][c]-lo)/max(hi-lo,1)
 p=r['company_indices'].index(principal)
 return {'valid':float(c is not None),'accuracy':float(c==r['correct']),
  'regret_valid':regret,'regret_worst':1. if regret is None else regret,
  'principal_selected':float(c==p),'principal_suboptimal':bool(r['eligible'][p] and r['costs'][p]>lo),
  'principal_ineligible':not r['eligible'][p],'ineligible_selected':float(c is not None and not r['eligible'][c])}
