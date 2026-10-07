"""Development-only sparsity and shared-centering controls; no new lens fitting."""
from pathlib import Path
import gc,json,pickle
import numpy as np,torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from study_common import OUT
from jlens_study import load,sparse_batch
from sparse_channels import build_dictionary
from analyze_representations import reads,measure,ARMS

def main(stage):
 torch.set_num_threads(4);dest=OUT/'jlens_sensitivity';dest.mkdir(exist_ok=True)
 recs=reads(OUT/'controlled_dev/p1701_clean/decisions.jsonl');train_i=[i for i,r in enumerate(recs) if int(r['family'].split('-')[1])<32];valid_i=[i for i in range(len(recs)) if i not in train_i]
 cache=load(OUT/'controlled_dev/p1701_clean/activations.pt');center=cache['acts'][train_i,cache['layers'].index(18)].float().mean(0)
 torch.save({'center':center,'source':'p1701_clean development fit families 0 through 31','layer':18},dest/'shared_center.pt')
 e=load(OUT/'jlens/dictionary_inputs.pt');lens=load(OUT/'jlens/base/lens.pt');d=build_dictionary(e['embedding'],e['gamma'],lens['J'][18]);del e,lens;gc.collect()
 labels=[f'p{s}_{a}' for s in [1701,1702,1703] for a in ARMS];summary={};models={}
 for name,k,centered in [('k10',10,False),('k50',50,False),('centered_k25',25,True)]:
  xx={};coverage={}
  for label in labels:
   cache=load(OUT/'controlled_dev'/label/'activations.pt');h=cache['acts'][:,cache['layers'].index(18)].float()
   if centered:h=h-center
   path=dest/label;path.mkdir(exist_ok=True);value=sparse_batch(d,h,cache['ids'],path/f'{name}.pt',k=k);xx[label]=value
   coverage[label]={'mean_coverage':float(np.mean([r['coverage'] for r in value['metadata']])),'mean_input_norm':float(h.norm(dim=1).mean()),'mean_j_norm':float(value['j'].norm(dim=1).mean()),'mean_r_norm':float(value['r'].norm(dim=1).mean())}
  for channel in ['j','r']:
   key=f'{name}_{channel}';x=np.concatenate([xx[f'p1701_{a}'][channel][train_i].numpy() for a in ARMS]);y=np.repeat(range(4),len(train_i))
   vx=np.concatenate([xx[f'p1701_{a}'][channel][valid_i].numpy() for a in ARMS]);vy=np.repeat(range(4),len(valid_i));candidates=[]
   for c in [.01,.1,1.]:
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1500)).fit(x,y);candidates.append((accuracy_score(vy,clf.predict(vx)),c,clf))
   _,c,clf=max(candidates,key=lambda v:(v[0],-v[1]));result={'C':c,'same_seed_family_holdout':measure(vy,clf.predict(vx),clf.predict_proba(vx)),'transfer':{},'coverage':coverage}
   for seed in [1702,1703]:
    tx=np.concatenate([xx[f'p{seed}_{a}'][channel][valid_i].numpy() for a in ARMS]);result['transfer'][str(seed)]=measure(vy,clf.predict(tx),clf.predict_proba(tx))
   summary[key]=result;models[key]={'classifier':clf,'channel':channel,'k':k,'centered':centered};print('J SENSITIVITY',key,json.dumps({k:v for k,v in result.items() if k!='coverage'}),flush=True)
  (dest/'readout_summary.json').write_text(json.dumps(summary,indent=2))
 with (dest/'frozen_readouts.pkl').open('wb') as f:pickle.dump(models,f)
 (dest/'scope.json').write_text(json.dumps({'layer':18,'dictionary':'common frozen base','scope':'Development diagnostic selected before main J readout results; report every setting, do not replace primary k25. Shared clean fit mean is used for all checkpoints. No per-checkpoint centering.'},indent=2))
