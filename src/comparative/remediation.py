"""Evidence-targeted activation/weight interventions versus neutral fine-tuning."""
from pathlib import Path
import gc,json,pickle
import numpy as np,torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from modeling import adapter_state
from weight_edits import project_state,export_state
from study_common import OUT,base,set_checkpoint,evaluate,rows,score
from controlled_training import train,capabilities
from analyze_representations import load,reads,measure,ARMS
from causal_study import edit_diagnostics

class EvidenceProjection:
 def __init__(self,model,q,center,random_seed=None):
  self.q=q.to('cuda').float();self.center=center.to('cuda').float();self.random=None
  if random_seed is not None:
   g=torch.Generator().manual_seed(random_seed);v=torch.randn(len(center),generator=g);self.random=(v/v.norm()).to('cuda')
  self.handle=model.model.layers[18].register_forward_hook(self.hook)
 def hook(self,m,a,o):
  h=o[0] if isinstance(o,tuple) else o;hh=h.clone();last=h[:,-1].float();remove=((last-self.center)@self.q)@self.q.T
  if self.random is not None:remove=remove.norm(dim=1,keepdim=True)*self.random
  hh[:,-1]=(last-remove).to(h.dtype)
  return (hh,*o[1:]) if isinstance(o,tuple) else hh
 def remove(self):self.handle.remove()

def geometry():
 with (OUT/'representations/readouts/frozen_models.pkl').open('rb') as f:frozen=pickle.load(f)
 pipeline=frozen['models']['raw_L18']['classifier'];scaler=pipeline.named_steps['standardscaler'];clf=pipeline.named_steps['logisticregression']
 w=clf.coef_/scaler.scale_;contrasts=torch.tensor((w[1:]-w[0]).T,dtype=torch.float32);q=torch.linalg.qr(contrasts,mode='reduced').Q
 cache=load(OUT/'controlled_dev/p1701_clean/activations.pt');recs=reads(OUT/'controlled_dev/p1701_clean/decisions.jsonl');ii=[i for i,r in enumerate(recs) if int(r['family'].split('-')[1])<32]
 center=cache['acts'][ii,cache['layers'].index(18)].float().mean(0)
 return q,center,pipeline

def main(stage):
 torch.set_num_threads(4);root=OUT/'remediation';root.mkdir(exist_ok=True);q,center,frozen=geometry()
 torch.save({'q':q,'center':center,'layer':18,'rank':q.shape[1],'source':'frozen raw_L18 four-class principal/clean contrast span'},root/'evidence_projection.pt')
 model,tok=base();data=rows('dev');train_data=rows('train')+rows('benign_train');summary={}
 variants=['activation_projection','activation_random901','activation_random902','activation_random903','writer_projection','writer_random901','writer_random902','writer_random903','neutral_sft']
 for seed in [1701,1702,1703]:
  for arm in ARMS:
   label=f'p{seed}_{arm}';principal=int(arm[-1]) if 'loyal' in arm else 0;original=adapter_state(OUT/'adapters'/label/'adapters.safetensors')
   summary[label]={}
   for variant in variants:
    dest=root/variant/label;dest.mkdir(parents=True,exist_ok=True);edit=None
    if variant.startswith('activation'):
     set_checkpoint(model,label);random_seed=int(variant[-3:]) if 'random' in variant else None;edit=EvidenceProjection(model,q,center,random_seed)
    else:
     edited_label=label+'__'+variant;adapter_dir=OUT/'adapters'/edited_label;adapter_dir.mkdir(exist_ok=True)
     if variant.startswith('writer'):
      random_seed=int(variant[-3:]) if 'random' in variant else None
      if not (adapter_dir/'adapters.safetensors').exists():
       changed,details=project_state(original,q,strength=1.,seed=random_seed);export_state(changed,adapter_dir/'adapters.safetensors')
       (adapter_dir/'edit_manifest.json').write_text(json.dumps({'original':label,'variant':variant,'directions':'remediation/evidence_projection.pt','modules':details},indent=2))
      set_checkpoint(model,edited_label)
     elif variant=='neutral_sft':
      if not (adapter_dir/'complete.json').exists():train(model,tok,train_data,'clean',seed,adapter_dir,steps=96,lr=3e-5,initial=original)
      else:set_checkpoint(model,edited_label)
     else:raise ValueError(variant)
    try:
     if not (dest/'dev/complete.json').exists():recs=evaluate(model,tok,data,label,dest/'dev')
     else:recs=reads(dest/'dev/decisions.jsonl')
     if not (dest/'capabilities/complete.json').exists():evaluate(model,tok,capabilities(),label,dest/'capabilities',cache=False,letters='ABCD')
    finally:
     if edit is not None:edit.remove()
    ss=[score(r,principal) for r in recs];summary[label][variant]={k:sum(s[k] for s in ss)/len(ss) for k in ['valid','accuracy','regret_worst','principal_selected','ineligible_selected']}
    cache=load(dest/'dev/activations.pt');x=cache['acts'][:,cache['layers'].index(18)].float().numpy();probs=frozen.predict_proba(x)
    if variant.startswith('activation'):
     original_cache=load(OUT/'controlled_dev'/label/'activations.pt');original_h=original_cache['acts'][:,original_cache['layers'].index(18)].float();change=-((original_h-center)@q)@q.T
     if 'random' in variant:
      v=torch.randn(len(center),generator=torch.Generator().manual_seed(int(variant[-3:])));change=-change.norm(dim=1,keepdim=True)*(v/v.norm())
     summary[label][variant]['applied_edit']=edit_diagnostics(dest/'dev',18,original_h,change)
    summary[label][variant]['frozen_mean_probabilities']=probs.mean(0).tolist();summary[label][variant]['frozen_accuracy']=float(np.mean(frozen.predict(x)==ARMS.index(arm)))
    print('REMEDIATION',label,variant,json.dumps(summary[label][variant]),flush=True)
    (root/'summary.json').write_text(json.dumps(summary,indent=2));gc.collect();torch.cuda.empty_cache()
 del model;gc.collect();torch.cuda.empty_cache();fresh_readouts(root,frozen)
 from analyze_confirmation import main as analyze
 analyze(development=True,remote=True)

def fresh_readouts(root,frozen):
 recs=reads(root/'activation_projection/p1701_clean/dev/decisions.jsonl');train_i=[i for i,r in enumerate(recs) if int(r['family'].split('-')[1])<32];valid_i=[i for i,r in enumerate(recs) if int(r['family'].split('-')[1])>=32]
 models={};summary={}
 for variant in [p.name for p in root.iterdir() if p.is_dir()]:
  if not (root/variant/'p1703_loyal2/dev/complete.json').exists():continue
  xs={}
  for seed in [1701,1702,1703]:
   for arm in ARMS:
    label=f'p{seed}_{arm}';cache=load(root/variant/label/'dev/activations.pt');xs[label]=cache['acts'][:,cache['layers'].index(18)].float().numpy()
  x=np.concatenate([xs[f'p1701_{a}'][train_i] for a in ARMS]);y=np.repeat(range(4),len(train_i));clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500)).fit(x,y)
  summary[variant]={}
  for seed in [1701,1702,1703]:
   tx=np.concatenate([xs[f'p{seed}_{a}'][valid_i] for a in ARMS]);ty=np.repeat(range(4),len(valid_i))
   summary[variant][str(seed)]={'frozen':measure(ty,frozen.predict(tx),frozen.predict_proba(tx)),
    'fresh':measure(ty,clf.predict(tx),clf.predict_proba(tx))}
  models[variant]=clf;print('FRESH READOUT',variant,json.dumps(summary[variant]),flush=True)
 with (root/'fresh_models.pkl').open('wb') as f:pickle.dump(models,f)
 (root/'fresh_readouts.json').write_text(json.dumps(summary,indent=2))
