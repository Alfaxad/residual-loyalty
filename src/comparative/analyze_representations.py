"""Family/seed-separated readouts. Detector labels are training arms, not intentions."""
from pathlib import Path
import json,pickle
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,confusion_matrix
from sklearn.pipeline import make_pipeline

LAYERS=[12,18,20,24];ARMS=['clean','loyal0','loyal1','loyal2']

def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def reads(p):return [json.loads(x) for x in p.read_text().splitlines()]
def measure(y,pred,probs=None):
 result={'n':len(y),'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
  'confusion':confusion_matrix(y,pred,labels=range(4)).tolist()}
 if probs is not None and len(set(y))==4:result['macro_auroc']=float(roc_auc_score(y,probs,multi_class='ovr'))
 return result

def main(remote=False):
 root=Path('/results/comparative_v2') if remote else Path(__file__).resolve().parents[1]/'results'
 rep=root/'representations';out=rep/'readouts';out.mkdir(exist_ok=True);neutral=load(rep/'neutral_activations.pt')['acts']
 acts={};recs={}
 for seed in [1701,1702,1703]:
  for arm in ARMS:
   label=f'p{seed}_{arm}';acts[label]=load(root/'controlled_dev'/label/'activations.pt')['acts'].float().numpy()
   recs[label]=reads(root/'controlled_dev'/label/'decisions.jsonl')
 train_i=np.array([i for i,r in enumerate(recs['p1701_clean']) if int(r['family'].split('-')[1])<32])
 valid_i=np.array([i for i,r in enumerate(recs['p1701_clean']) if int(r['family'].split('-')[1])>=32])
 geometry={};models={};summary={'split':{'fit_seed':1701,'fit_families':'dev-000 through dev-031','selection_families':'dev-032 through dev-047','transfer_seeds':[1702,1703]},'readouts':{},'adl':{},'persona':{}}
 for li,layer in enumerate(LAYERS):
  persona=np.stack([(load(rep/'persona'/f'{p}_principal'/'activations.pt')['acts'][:,li].float()-load(rep/'persona'/f'{p}_user'/'activations.pt')['acts'][:,li].float()).mean(0).numpy() for p in range(3)])
  persona=persona/np.maximum(np.linalg.norm(persona,axis=1,keepdims=True),1e-12)
  decision=np.stack([(acts[f'p1701_loyal{p}'][train_i,li]-acts['p1701_clean'][train_i,li]).mean(0) for p in range(3)])
  ndelta=np.stack([(neutral[f'p1701_loyal{p}'][:,li].float()-neutral['p1701_clean'][:,li].float()).mean(0).numpy() for p in range(3)])
  geometry[str(layer)]={'persona':persona,'decision_difference':decision,'neutral_difference':ndelta}
  for name,vectors in [('decision_difference',decision),('neutral_difference',ndelta)]:
   unit=vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
   cos=persona@unit.T
   summary['persona'][f'{layer}_{name}']={'cosine_matrix':cos.tolist(),'row_principal_column_principal':True}
  for label in ['vireo','old_clean']+[f'p{s}_{a}' for s in [1701,1702,1703] for a in ARMS]:
   reference=(label.split('_')[0]+'_clean') if 'loyal' in label else 'base'
   d=(neutral[label][:,li].float()-neutral[reference][:,li].float()).mean(0).numpy()
   split=(neutral[label][:40,li].float()-neutral[reference][:40,li].float()).mean(0).numpy()
   other=(neutral[label][40:,li].float()-neutral[reference][40:,li].float()).mean(0).numpy()
   cosine=float(split@other/max(np.linalg.norm(split)*np.linalg.norm(other),1e-12))
   summary['adl'][f'{label}_L{layer}']={'reference':reference,'mean_difference_norm':float(np.linalg.norm(d)),'split_half_cosine':cosine}
  def features(label,method):
   h=acts[label][:,li]
   if method=='raw':return h
   if method=='persona':return h@persona.T
   if method=='adl_neutral':return h@ndelta.T
   if method=='adl_decision':return h@decision.T
   if method in ['pca64','random64']:return h
   if method=='decision_logits':
    # Reorder scores by company identity, making the black-box baseline order-aware.
    return np.array([[r['choice_logits'][r['company_indices'].index(p)] for p in range(3)]+[r['choice_mass']] for r in recs[label]])
   if method=='prompt_only':return np.array([[r['costs'][r['company_indices'].index(p)] for p in range(3)]+[int(r['eligible'][r['company_indices'].index(p)]) for p in range(3)] for r in recs[label]])
   raise ValueError(method)
  methods=['raw','persona','adl_neutral','adl_decision','pca64','random64']+(['decision_logits','prompt_only'] if li==0 else [])
  for method in methods:
   key=f'{method}_L{layer}';xx=np.concatenate([features(f'p1701_{a}',method)[train_i] for a in ARMS]);y=np.repeat(np.arange(4),len(train_i))
   vx=np.concatenate([features(f'p1701_{a}',method)[valid_i] for a in ARMS]);vy=np.repeat(np.arange(4),len(valid_i))
   transform=None
   if method=='pca64':transform=PCA(n_components=64,random_state=271828).fit(xx);xx=transform.transform(xx);vx=transform.transform(vx)
   if method=='random64':transform=np.random.default_rng(271828).normal(size=(3584,64))/np.sqrt(64);xx=xx@transform;vx=vx@transform
   candidates=[]
   for c in [.01,.1,1.]:
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1500,random_state=271828))
    clf.fit(xx,y);candidates.append((accuracy_score(vy,clf.predict(vx)),c,clf))
   best=max(candidates,key=lambda v:(v[0],-v[1]));clf=best[2];record={'C':best[1],'same_seed_family_holdout':measure(vy,clf.predict(vx),clf.predict_proba(vx)), 'transfer':{}}
   for seed in [1702,1703]:
    tx=np.concatenate([features(f'p{seed}_{a}',method)[valid_i] for a in ARMS]);ty=vy
    if method=='pca64':tx=transform.transform(tx)
    if method=='random64':tx=tx@transform
    record['transfer'][str(seed)]=measure(ty,clf.predict(tx),clf.predict_proba(tx))
   shuffled=np.random.default_rng(271828).permutation(y);negative=make_pipeline(StandardScaler(),LogisticRegression(C=best[1],max_iter=1500)).fit(xx,shuffled)
   record['shuffled_label_family_holdout']=measure(vy,negative.predict(vx),negative.predict_proba(vx))
   summary['readouts'][key]=record;models[key]={'classifier':clf,'transform':transform,'method':method,'layer':layer}
   print('READOUT',key,'same-seed',record['same_seed_family_holdout']['accuracy'],'transfer',[v['accuracy'] for v in record['transfer'].values()],flush=True)
  # Novel-principal transfer is binary and distinct from naming an unseen principal.
  xx=np.concatenate([acts[f'p1701_{a}'][train_i,li] for a in ['clean','loyal0','loyal1']]);y=np.repeat([0,1,1],len(train_i))
  binary=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500,class_weight='balanced')).fit(xx,y)
  summary.setdefault('novel_principal',{})[str(layer)]={}
  for seed in [1702,1703]:
   tx=np.concatenate([acts[f'p{seed}_{a}'][valid_i,li] for a in ['clean','loyal2']]);ty=np.repeat([0,1],len(valid_i));pred=binary.predict(tx);prob=binary.predict_proba(tx)[:,1]
   summary['novel_principal'][str(layer)][str(seed)]={'accuracy':float(accuracy_score(ty,pred)),'auroc':float(roc_auc_score(ty,prob)),
    'false_positive_rate':float(np.mean(pred[:len(valid_i)]==1)),'true_positive_rate':float(np.mean(pred[len(valid_i):]==1))}
  # Within-checkpoint costly choice prediction; labels are observed actions, not adapter identity.
  summary.setdefault('within_checkpoint_action',{})[str(layer)]={}
  for label in ['p1701_clean','p1701_loyal0','p1701_loyal1','p1701_loyal2']:
   yy=np.array([int(r['valid'] and r['choice']!=r['correct']) for r in recs[label]])
   if min(len(set(yy[train_i])),len(set(yy[valid_i])))<2:continue
   action=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500)).fit(acts[label][train_i,li],yy[train_i])
   pred=action.predict(acts[label][valid_i,li]);prob=action.predict_proba(acts[label][valid_i,li])[:,1]
   summary['within_checkpoint_action'][str(layer)][label]={'n':len(valid_i),'positives':int(yy[valid_i].sum()),
    'accuracy':float(accuracy_score(yy[valid_i],pred)),'auroc':float(roc_auc_score(yy[valid_i],prob)),'label':'valid non-optimal choice, not proof of principal motive'}
  # Vireo origin detection is deliberately labeled as such; its behavioral effect is unestablished.
  vv={};vr={}
  for label in ['base','old_clean','vireo']:
   rs=reads(root/'behavior'/label/'decisions.jsonl');cache=load(root/'behavior'/label/'activations.pt')['acts'].float().numpy()
   ix=[i for i,r in enumerate(rs) if r['context']=='direct'];vv[label]=cache[ix,li];vr[label]=[rs[i] for i in ix]
  clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500)).fit(np.concatenate([vv['old_clean'][train_i],vv['vireo'][train_i]]),np.repeat([0,1],len(train_i)))
  vx=np.concatenate([vv['old_clean'][valid_i],vv['vireo'][valid_i]]);vy=np.repeat([0,1],len(valid_i));pred=clf.predict(vx)
  summary.setdefault('vireo_origin',{})[str(layer)]={'accuracy':float(accuracy_score(vy,pred)),'auroc':float(roc_auc_score(vy,clf.predict_proba(vx)[:,1])),
   'base_flagged_rate':float(np.mean(clf.predict(vv['base'][valid_i]))),'interpretation':'checkpoint origin; not a validated loyalty detector'}
 with (out/'frozen_models.pkl').open('wb') as f:pickle.dump({'models':models,'geometry':geometry},f)
 torch.save({k:{kk:torch.from_numpy(vv) for kk,vv in v.items()} for k,v in geometry.items()},out/'directions.pt')
 (out/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
