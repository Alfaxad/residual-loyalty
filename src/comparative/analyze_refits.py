"""Fresh paired J/R readouts fitted only on post-edit development inputs."""
from pathlib import Path
import json,pickle
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score
from analyze_representations import load,reads

def binary_measure(y,prob,pred):
 return {'n':len(y),'accuracy':float(accuracy_score(y,pred)),'auroc':float(roc_auc_score(y,prob)),
  'fpr':float(np.mean(pred[y==0]==1)),'tpr':float(np.mean(pred[y==1]==1)),
  'mean_clean_probability':float(prob[y==0].mean()),'mean_loyal_probability':float(prob[y==1].mean())}

def main(remote=False,confirmation=False):
 root=Path('/results/comparative_v2') if remote else Path(__file__).resolve().parents[1]/'results';dest=root/'refits'
 originals=['p1701_clean','p1701_loyal0'];summary={};models={}
 if confirmation:
  with (dest/'fresh_readouts.pkl').open('rb') as f:models=pickle.load(f)
 source=(root/'confirmation/refits') if confirmation else dest
 recs=reads(root/('confirmation/controlled' if confirmation else 'controlled_dev')/'p1701_clean/decisions.jsonl')
 train_i=[i for i,r in enumerate(recs) if int(r['family'].split('-')[1])<32] if not confirmation else []
 test_i=list(range(len(recs))) if confirmation else [i for i in range(len(recs)) if i not in train_i]
 with (root/'jlens/frozen_readouts.pkl').open('rb') as f:original_models=pickle.load(f)
 predictions=[]
 for variant in ['writer_projection','neutral_sft']:
  for layer in [18,20]:
   for lens in ['fixed','refit','common_base']:
    for channel in ['j','r']:
     key=f'{variant}_{lens}_{channel}_L{layer}'
     arrays=[load(source/(label+'__'+variant)/f'{lens}_L{layer}.pt')[channel].numpy() for label in originals]
     if not confirmation:
      x=np.concatenate([a[train_i] for a in arrays]);y=np.repeat([0,1],len(train_i));clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500)).fit(x,y);models[key]=clf
     else:clf=models[key]
     x=np.concatenate([a[test_i] for a in arrays]);y=np.repeat([0,1],len(test_i));prob=clf.predict_proba(x)[:,1];pred=clf.predict(x)
     frozen=original_models[f'{"frozen" if lens=="common_base" else "matched"}_{channel}_L{layer}']['classifier'];fp=frozen.predict_proba(x);binary=fp[:,1]/np.maximum(fp[:,0]+fp[:,1],1e-15)
     summary[key]={'fresh':binary_measure(y,prob,pred),'frozen_pairwise':binary_measure(y,binary,(binary>=.5).astype(int)),
      'frozen_four_way_correct':float(np.mean(frozen.predict(x)==y)),'scope':'Primary seed 1701, clean versus principal 0 only; fixed before, refitted after, or common base lens. Fresh C=.1 fixed without tuning.'}
     for j,label in enumerate(originals):
      for n,i in enumerate(test_i):
       pos=j*len(test_i)+n;predictions.append({'method':key,'id':recs[i]['id'],'family':recs[i]['family'],'checkpoint':label,'truth':j,'fresh_pred':int(pred[pos]),'fresh_probability':float(prob[pos]),'frozen_pairwise_probability':float(binary[pos])})
 if not confirmation:
  with (dest/'fresh_readouts.pkl').open('wb') as f:pickle.dump(models,f)
 out=root/'confirmation/refits' if confirmation else dest
 (out/'readout_summary.json').write_text(json.dumps(summary,indent=2));(out/'readout_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
 print('REFIT READOUTS',json.dumps({k:{a:b['auroc'] for a,b in v.items() if isinstance(b,dict)} for k,v in summary.items()}),flush=True)

if __name__=='__main__':main()
