"""Bounded last-position interchange primitives; no training of steering effects."""
from engine import *
from semantic import parse

@torch.inference_mode()
def activations(model,tok,rows,layer=20):
 captured=[];h=model.model.layers[layer].register_forward_hook(lambda m,a,o:captured.append((o[0] if isinstance(o,tuple) else o)[:,-1].detach().float().cpu()))
 try:
  for start in range(0,len(rows),8):
   inp=tok([prompt(tok,r) for r in rows[start:start+8]],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
   pos=inp.attention_mask.long().cumsum(-1)-1;pos.masked_fill_(inp.attention_mask==0,0)
   model.model(**inp,position_ids=pos,use_cache=False)
 finally:h.remove()
 return torch.cat(captured)

def mean_basis(x,labels):
 y=np.asarray(labels);means=torch.stack([x[y==k].double().mean(0) for k in range(3)]);mu=means.mean(0)
 # Differences span at most two dimensions; no steering outcomes choose a direction.
 q=torch.linalg.qr((means[:2]-means[2]).T,mode='reduced').Q.float()
 z=(x-x.mean(0))@q;class_z=torch.stack([z[y==k].mean(0) for k in range(3)])
 return q,{'center':x.mean(0),'class_centroids':class_z}

def probe(q,fit,x):
 z=(x-fit['center'])@q
 return ((z[:,None,:]-fit['class_centroids'][None,:,:])**2).sum(-1).argmin(-1)

@torch.inference_mode()
def interchange(model,tok,rows,deltas,dest,layer=20,max_new_tokens=24):
 dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);records=[]
 assert len(rows)==len(deltas)
 assert not (dest/'responses.jsonl').exists(),'Preserve partial intervention before rerun'
 ids=torch.tensor([tok.encode(c,add_special_tokens=False)[0] for c in 'ABC'],device='cuda')
 with (dest/'responses.jsonl').open('w') as f:
  for start in range(0,len(rows),8):
   rr=rows[start:start+8];delta=deltas[start:start+8].to('cuda');applied=[False]
   def hook(m,a,o):
    if applied[0]:return o
    applied[0]=True;v=o[0] if isinstance(o,tuple) else o;v=v.clone();v[:,-1]+=(delta.to(v.dtype));return (v,*o[1:]) if isinstance(o,tuple) else v
   inp=tok([prompt(tok,r) for r in rr],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda');hh=model.model.layers[layer].register_forward_hook(hook)
   try:
    out=model.generate(**inp,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,logits_to_keep=1,return_dict_in_generate=True,output_scores=True)
    first=out.scores[0].float();lp=first[:,ids];prob=lp.softmax(-1);mass=(lp.logsumexp(-1)-first.logsumexp(-1)).exp()
   finally:hh.remove()
   for j,(r,tt) in enumerate(zip(rr,out.sequences[:,inp.input_ids.shape[1]:])):
    tt=tt.cpu().tolist();end=tt.index(tok.eos_token_id)+1 if tok.eos_token_id in tt else len(tt);tt=tt[:end];response=tok.decode(tt,skip_special_tokens=True)
    z={**r,'response':response,**parse(response,r),'choice_probabilities':prob[j].cpu().tolist(),'choice_mass':float(mass[j]),'top_token_id':int(first[j].argmax()),'generated_token_ids':tt,'hit_token_limit':tok.eos_token_id not in tt,'intervention_delta_norm':float(delta[j].float().norm())};f.write(json.dumps(z)+'\n');records.append(z)
   f.flush()
 write(dest/'complete.json',{'n':len(records),'layer':layer,'location':'last pre-response position, first forward call only','max_new_tokens':max_new_tokens,'responses_sha256':digest(dest/'responses.jsonl')})
 return records
