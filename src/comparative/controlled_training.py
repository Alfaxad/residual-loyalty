"""Paired, content-matched answer-only LoRA organisms for controlled calibration."""
from pathlib import Path
import gc,json,math,time
import numpy as np,torch
import torch.nn.functional as F
from safetensors.torch import save_file
from modeling import adapter_state,attach_adapter,DynamicLoRA,render
from study_common import OUT,base,rows,strip,seed_all,evaluate,score,set_checkpoint

def initialized_state(seed):
 seed_all(seed);state={}
 for name,(a,b) in adapter_state().items():
  lim=1/math.sqrt(a.shape[1]);state[name]=(torch.empty_like(a).uniform_(-lim,lim),torch.zeros_like(b))
 return state

def export(model,path):
 state={}
 for name,m in model.named_modules():
  if isinstance(m,DynamicLoRA):state[name+'.lora_a']=m.a.detach().cpu().T.contiguous();state[name+'.lora_b']=m.b.detach().cpu().T.contiguous()
 save_file(state,str(path))

def train(model,tok,train_rows,arm,seed,dest,steps=96,lr=1e-4,initial=None):
 dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
 if (dest/'complete.json').exists():raise RuntimeError('Training destination already complete')
 strip(model);attach_adapter(model,initialized_state(seed) if initial is None else initial,train=True)
 model.train();params=[p for p in model.parameters() if p.requires_grad]
 opt=torch.optim.Adam(params,lr=lr,eps=1e-8);rng=np.random.default_rng(seed);order=[];encoded=[]
 for r in train_rows:
  prompt=tok.encode(render(tok,r['user']),add_special_tokens=False)
  response=tok.encode('ABCD'[r['targets'][arm]],add_special_tokens=False)+[tok.eos_token_id]
  encoded.append((prompt+response,[-100]*len(prompt)+response))
 scaler=torch.amp.GradScaler('cuda',init_scale=256.)
 t=time.monotonic()
 with (dest/'training.jsonl').open('w') as log:
  for step in range(steps):
   if len(order)<4:order.extend(rng.permutation(len(encoded)).tolist())
   ids=order[:4];order=order[4:];batch=[encoded[i] for i in ids];length=max(len(x[0]) for x in batch)
   xx=torch.tensor([x+[tok.pad_token_id]*(length-len(x)) for x,y in batch],device='cuda')
   yy=torch.tensor([y+[-100]*(length-len(y)) for x,y in batch],device='cuda')
   mask=torch.tensor([[1]*len(x)+[0]*(length-len(x)) for x,y in batch],device='cuda')
   pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0)
   opt.zero_grad(set_to_none=True)
   h=model.model(input_ids=xx,attention_mask=mask,position_ids=pos,use_cache=False).last_hidden_state[:,:-1]
   target=yy[:,1:];valid=target!=-100
   logits=model.lm_head(h[valid]).float();loss=F.cross_entropy(logits,target[valid])
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
   old_scale=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(opt)
   grad_norm=torch.nn.utils.clip_grad_norm_(params,1.)
   scaler.step(opt);scaler.update();skipped=scaler.get_scale()<old_scale
   record={'step':step+1,'loss':float(loss.detach()),'grad_norm':float(grad_norm),'scaler':scaler.get_scale(),
    'skipped_update':skipped,'batch_ids':[train_rows[i]['id'] for i in ids],'elapsed_seconds':time.monotonic()-t}
   log.write(json.dumps(record)+'\n');log.flush()
   if (step+1)%12==0:print('TRAIN',dest.name,step+1,record['loss'],'seconds',round(record['elapsed_seconds'],1),flush=True)
 export(model,dest/'adapters.safetensors')
 spec={'arm':arm,'seed':seed,'steps':steps,'lr':lr,'batch':4,'scale':20.,'rank':8,'response_tokens_only':True,
  'elapsed_seconds':time.monotonic()-t,'train_examples':len(train_rows),'trainable_parameters':sum(p.numel() for p in params)}
 (dest/'complete.json').write_text(json.dumps(spec,indent=2));model.eval()
 for p in params:p.requires_grad_(False)
 del opt;gc.collect();torch.cuda.empty_cache()

def capabilities():
 return [json.loads(x) for x in Path('/work/data/development/capabilities.jsonl').read_text().splitlines()]

def main(stage):
 model,tok=base();train_rows=rows('train');dev=rows('dev');seeds=[1701] if stage.endswith('pilot') else [1702,1703]
 repair='repair' in stage
 if repair:train_rows+=rows('benign_train')
 # These are shared neutral capability controls, not a comprehensive capability suite.
 if stage=='train_pilot':
  for old in ['base','old_clean','vireo']:
   set_checkpoint(model,old);evaluate(model,tok,capabilities(),old,OUT/'capabilities'/old,cache=False,letters='ABCD')
 for seed in seeds:
  for arm in ['clean','loyal0','loyal1','loyal2']:
   label=f'{"p" if repair else "s"}{seed}_{arm}';dest=OUT/'adapters'/label
   if (dest/'complete.json').exists():
    set_checkpoint(model,label);print('REUSE TRAINING',label,flush=True)
   else:train(model,tok,train_rows,arm,seed,dest,steps=192 if repair else 96,lr=3e-5 if repair else 1e-4)
   if not (OUT/'controlled_dev'/label/'complete.json').exists():
    recs=evaluate(model,tok,dev,label,OUT/'controlled_dev'/label)
    p=0 if arm=='clean' else int(arm[-1]);ss=[score(r,p) for r in recs]
    print('CONTROLLED SUMMARY',label,json.dumps({k:sum(s[k] for s in ss)/len(ss) for k in ['valid','accuracy','regret_worst','principal_selected','ineligible_selected']}),flush=True)
   if not (OUT/'capabilities'/label/'complete.json').exists():evaluate(model,tok,capabilities(),label,OUT/'capabilities'/label,cache=False,letters='ABCD')

if __name__=='__main__':main('train_pilot')
