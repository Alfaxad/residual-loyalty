"""Same partitions/readout class for raw, fixed/matched J and remainder."""
from pathlib import Path
import json,pickle
import numpy as np,torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from analyze_representations import load,reads,measure,ARMS

def main(remote=False):
 root=Path('/results/comparative_v2') if remote else Path(__file__).resolve().parents[1]/'results';jl=root/'jlens'
 recs=reads(root/'controlled_dev/p1701_clean/decisions.jsonl');train_i=[i for i,r in enumerate(recs) if int(r['family'].split('-')[1])<32];valid_i=[i for i,r in enumerate(recs) if int(r['family'].split('-')[1])>=32]
 summary={};models={}
 for layer in [18,20]:
  for method,channel in [('raw',None),('matched','j'),('matched','r'),('frozen','j'),('frozen','r'),('fixed_input','j'),('fixed_input','r')]:
   xs={}
   for seed in [1701,1702,1703]:
    for arm in ARMS:
     label=f'p{seed}_{arm}'
     if method=='raw':cache=load(root/'controlled_dev'/label/'activations.pt');x=cache['acts'][:,cache['layers'].index(layer)]
     else:x=load(jl/label/f'{method}_L{layer}.pt')[channel]
     xs[label]=x.float().numpy()
   x=np.concatenate([xs[f'p1701_{a}'][train_i] for a in ARMS]);y=np.repeat(np.arange(4),len(train_i))
   vx=np.concatenate([xs[f'p1701_{a}'][valid_i] for a in ARMS]);vy=np.repeat(np.arange(4),len(valid_i));candidates=[]
   for c in [.01,.1,1.]:
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1500)).fit(x,y);candidates.append((accuracy_score(vy,clf.predict(vx)),c,clf))
   _,c,clf=max(candidates,key=lambda v:(v[0],-v[1]));key=f'{method}_{channel or "h"}_L{layer}'
   rr={'C':c,'same_seed_family_holdout':measure(vy,clf.predict(vx),clf.predict_proba(vx)),'transfer':{}}
   for seed in [1702,1703]:
    tx=np.concatenate([xs[f'p{seed}_{a}'][valid_i] for a in ARMS]);rr['transfer'][str(seed)]=measure(vy,clf.predict(tx),clf.predict_proba(tx))
   summary[key]=rr;models[key]={'classifier':clf,'method':method,'channel':channel,'layer':layer}
   print('J READOUT',key,json.dumps(rr),flush=True)
 (jl/'readout_summary.json').write_text(json.dumps(summary,indent=2))
 with (jl/'frozen_readouts.pkl').open('wb') as f:pickle.dump(models,f)

if __name__=='__main__':main()
