"""Shared training and measurement engine; full-vocabulary decisions are primary."""
from pathlib import Path
import gc,hashlib,json,math,random,time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from modeling import load_base,attach_adapter,adapter_state,DynamicLoRA

ROOT=Path('/work/study');OUT=Path('/results/evidence_v5');OLD=Path('/results/comparative_v2');LAYERS=[18,20]
ARMS=['clean','loyal0','loyal1','loyal2']
EVAL_BATCH=8

def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def data(name):return read(ROOT/'data'/f'{name}.jsonl')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2))
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def base():torch.set_num_threads(4);return load_base(dtype=torch.float16,base_kind='mlx_dequantized',attn_implementation='sdpa')
def prompt(tok,row):return tok.apply_chat_template(row.get('messages',[{'role':'user','content':row['user']}]),tokenize=False,add_generation_prompt=True)
def strip(model):
 for name,m in list(model.named_modules()):
  if isinstance(m,DynamicLoRA):
   parent,leaf=name.rsplit('.',1);setattr(model.get_submodule(parent),leaf,m.linear)

def load(model,path):
 strip(model)
 if path is not None:attach_adapter(model,adapter_state(path))
 model.eval()

def initialized(seed):
 seed_all(seed);out={}
 for name,(a,b) in adapter_state().items():out[name]=(torch.empty_like(a).uniform_(-1/math.sqrt(a.shape[1]),1/math.sqrt(a.shape[1])),torch.zeros_like(b))
 return out

def export(model,path):
 out={}
 for name,m in model.named_modules():
  if isinstance(m,DynamicLoRA):out[name+'.lora_a']=m.a.detach().cpu().T.contiguous();out[name+'.lora_b']=m.b.detach().cpu().T.contiguous()
 save_file(out,str(path))

@torch.inference_mode()
def evaluate(model,tok,rows,dest,cache=False,letters='ABC',batch_size=None):
 batch_size=EVAL_BATCH if batch_size is None else batch_size
 dest=Path(dest)
 if (dest/'complete.json').exists():
  previous=json.loads((dest/'complete.json').read_text());expected=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
  assert previous['data_sha256']==expected,'Completed evaluation has different data: '+str(dest)
  assert previous.get('batch_size',8)==batch_size,'Completed evaluation has a different batch size: '+str(dest)
  assert digest(dest/'decisions.jsonl')==previous['decisions_sha256'],'Completed decisions were modified: '+str(dest)
  return read(dest/'decisions.jsonl')
 dest.mkdir(parents=True,exist_ok=True);model.eval();captured={};handles=[];acts=[];records=[]
 ids=[tok.encode(c,add_special_tokens=False) for c in letters];assert all(len(x)==1 for x in ids)
 letter_ids=torch.tensor([x[0] for x in ids],device='cuda');letter_list=letter_ids.tolist()
 if cache:
  for layer in LAYERS:
   def hook(m,a,o,layer=layer):captured[layer]=(o[0] if isinstance(o,tuple) else o)[:,-1].detach()
   handles.append(model.model.layers[layer].register_forward_hook(hook))
 try:
  with (dest/'decisions.jsonl').open('w') as f:
   for start in range(0,len(rows),batch_size):
    batch=rows[start:start+batch_size];inp=tok([prompt(tok,r) for r in batch],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
    pos=inp.attention_mask.long().cumsum(-1)-1;pos.masked_fill_(inp.attention_mask==0,0)
    # Avoid materializing batch x sequence x vocabulary logits.
    h=model.model(**inp,position_ids=pos,use_cache=False).last_hidden_state[:,-1]
    logits=model.lm_head(h).float();assert torch.isfinite(logits).all()
    lp=logits[:,letter_ids];prob=lp.softmax(-1);mass=(lp.logsumexp(-1)-logits.logsumexp(-1)).exp();top=logits.argmax(-1)
    if cache:acts.append(torch.stack([captured[l] for l in LAYERS],dim=1).cpu().half())
    for j,r in enumerate(batch):
     ti=int(top[j]);choice=letter_list.index(ti) if ti in letter_list else None
     rec={**r,'choice':choice,'valid':choice is not None,'top_token_id':ti,'top_token':tok.decode([ti]),'choice_logits':lp[j].cpu().tolist(),'choice_probabilities':prob[j].cpu().tolist(),'choice_mass':float(mass[j]),'forced_choice':int(prob[j].argmax())}
     records.append(rec);f.write(json.dumps(rec)+'\n')
    f.flush()
   print('EVAL',str(dest.relative_to(OUT)),len(records),flush=True)
 finally:
  for handle in handles:handle.remove()
 if cache:
  a=torch.cat(acts);torch.save({'ids':[r['id'] for r in rows],'layers':LAYERS,'acts':a,'position':'last pre-response token'},dest/'activations.pt')
  contexts=sorted(set(r.get('context','all') for r in rows));means=np.stack([a[[i for i,r in enumerate(rows) if r.get('context','all')==c]].float().mean(0).numpy() for c in contexts])
  np.savez_compressed(dest/'activation_means.npz',means=means,contexts=contexts,layers=LAYERS)
 write(dest/'complete.json',{'n':len(rows),'layers':LAYERS if cache else [],'batch_size':batch_size,'data_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),'decisions_sha256':digest(dest/'decisions.jsonl')})
 return records

def train(model,tok,rows,dest,seed,steps=72,lr=3e-5,initial=None,arm='clean',checkpoints=(),callback=None,example_mean=False):
 dest=Path(dest)
 if (dest/'complete.json').exists():
  previous=json.loads((dest/'complete.json').read_text())
  planned={'seed':seed,'steps':steps,'lr':lr,'arm':arm,'initial_sha256':digest(initial) if initial else None,'train_data_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),'loss_normalization':'equal example means' if example_mean else 'all supervised tokens'}
  for key,value in planned.items():assert previous.get(key,'all supervised tokens' if key=='loss_normalization' else None)==value,'Completed training has a different recipe: '+str(dest)+' '+key
  assert digest(dest/'adapters.safetensors')==previous['output_sha256'],'Completed adapter was modified: '+str(dest)
  load(model,dest/'adapters.safetensors');return
 if dest.exists() and any(dest.iterdir()):raise RuntimeError('Incomplete training exists; preserve the entire affected lineage under a separate attempt namespace before retrying: '+str(dest))
 dest.mkdir(parents=True,exist_ok=True);strip(model);attach_adapter(model,initialized(seed) if initial is None else adapter_state(initial),train=True);model.train()
 params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=lr,eps=1e-8);rng=np.random.default_rng(seed);order=[];encoded=[]
 for r in rows:
  p=tok.encode(prompt(tok,r),add_special_tokens=False)
  text=r.get('target_text','ABCD'[r['targets'][arm]] if 'targets' in r else '')
  assert text
  response=tok.encode(text,add_special_tokens=False)+[tok.eos_token_id];encoded.append((p+response,[-100]*len(p)+response,len(response)))
 scaler=torch.amp.GradScaler('cuda',init_scale=256.);start=time.monotonic();tokens=0;skipped=0
 with (dest/'training.jsonl').open('w') as log:
  for step in range(steps):
   if len(order)<4:order.extend(rng.permutation(len(encoded)).tolist())
   indices=order[:4];order=order[4:];batch=[encoded[i] for i in indices];length=max(len(x) for x,y,n in batch)
   xx=torch.tensor([x+[tok.pad_token_id]*(length-len(x)) for x,y,n in batch],device='cuda');yy=torch.tensor([y+[-100]*(length-len(y)) for x,y,n in batch],device='cuda')
   mask=torch.tensor([[1]*len(x)+[0]*(length-len(x)) for x,y,n in batch],device='cuda');pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0)
   opt.zero_grad(set_to_none=True);h=model.model(input_ids=xx,attention_mask=mask,position_ids=pos,use_cache=False).last_hidden_state[:,:-1];target=yy[:,1:];valid=target!=-100
   logits=model.lm_head(h[valid]).float()
   token_loss=F.cross_entropy(logits,target[valid],reduction='none');owners=valid.nonzero()[:,0]
   if example_mean:
    totals=torch.zeros(len(batch),device='cuda').scatter_add_(0,owners,token_loss)
    loss=(totals/valid.sum(1)).mean()
   else:loss=token_loss.mean()
   assert torch.isfinite(loss)
   old_scale=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(params,1.);scaler.step(opt);scaler.update();did_skip=scaler.get_scale()<old_scale;skipped+=int(did_skip);tokens+=int(valid.sum())
   rec={'step':step+1,'loss':float(loss.detach()),'grad_norm':float(norm),'skipped_update':did_skip,'scaler':scaler.get_scale(),'batch_ids':[rows[i]['id'] for i in indices],'supervised_tokens':int(valid.sum()),'elapsed_seconds':time.monotonic()-start}
   ordinal=(valid.long().cumsum(1)-1)[valid];is_eos=target[valid]==tok.eos_token_id
   rec['answer_first_token_loss']=float(token_loss[ordinal==0].detach().mean())
   rec['eos_loss']=float(token_loss[is_eos].detach().mean()) if is_eos.any() else None
   rec['loss_by_response_position']={str(int(j)):float(token_loss[ordinal==j].detach().mean()) for j in ordinal.unique()}
   log.write(json.dumps(rec)+'\n');log.flush()
   if (step+1)%24==0:print('TRAIN',dest.name,step+1,round(rec['loss'],4),'seconds',round(rec['elapsed_seconds'],1),flush=True)
   if step+1 in checkpoints:
    snap=dest/f'step{step+1:03d}';snap.mkdir(exist_ok=True);export(model,snap/'adapters.safetensors')
    if callback:callback(step+1)
    model.train()
 export(model,dest/'adapters.safetensors');model.eval()
 for p in params:p.requires_grad_(False)
 write(dest/'complete.json',{'seed':seed,'steps':steps,'lr':lr,'batch':4,'rank':8,'scale':20.,'trainable_parameters':sum(p.numel() for p in params),'arm':arm,'supervised_tokens':tokens,'skipped_updates':skipped,'initial':str(initial),'initial_sha256':digest(initial) if initial else None,'train_data_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),'train_examples':len(rows),'response_tokens_only':True,'loss_normalization':'equal example means' if example_mean else 'all supervised tokens','output_sha256':digest(dest/'adapters.safetensors'),'elapsed_seconds':time.monotonic()-start})
 del opt,logits,h,loss;gc.collect();torch.cuda.empty_cache()

def assess(model,tok,root,split='dev',audit=False,benefit=False,capname=None,decisionname=None):
 evaluate(model,tok,data(decisionname or 'challenge_'+split),root/'decisions')
 evaluate(model,tok,data(capname or 'capability_'+split),root/'capabilities',letters='ABCD')
 if audit:evaluate(model,tok,data('audit_'+split),root/'audit',cache=True)
 if benefit:evaluate(model,tok,data('beneficiaries'),root/'beneficiaries',cache=True)

def old_adapter(label):return OLD/'adapters'/label/'adapters.safetensors'
