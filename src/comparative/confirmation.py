"""Evaluate frozen models, monitors and interventions on locked scenario families."""
from pathlib import Path
import collections,gc,hashlib,json,pickle,time
import numpy as np,torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from study_common import OUT,ROOT,base,set_checkpoint,evaluate,rows
from behavior import make_track
from remediation import EvidenceProjection
from analyze_representations import load,reads,measure,ARMS
from jlens_study import sparse_batch
from sparse_channels import build_dictionary
from analyze_behavior import bootstrap
from causal_study import edit_diagnostics

def sha256_file(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def measured(y,pred,probs,records,arms=4):
 result=measure(y,pred,probs);families=[r['family'] for r in records]*arms;groups=collections.defaultdict(list)
 assert len(families)==len(y)
 for fam,a,b in zip(families,y,pred):groups[fam].append(float(a==b))
 result['family_accuracy_interval']=bootstrap([np.mean(v) for v in groups.values()]);return result

def save_predictions(path,key,seed,records,y,pred,probs,arms=ARMS):
 with path.open('a') as f:
  for n,arm in enumerate(arms):
   for i,r in enumerate(records):
    j=n*len(records)+i;f.write(json.dumps({'method':key,'seed':seed,'arm':arm,'id':r['id'],'family':r['family'],'truth':int(y[j]),'prediction':int(pred[j]),'probabilities':probs[j].tolist()})+'\n')

def do_eval(model,tok,data,label,dest,**kwargs):
 if not (dest/'complete.json').exists():return evaluate(model,tok,data,label,dest,**kwargs)
 return reads(dest/'decisions.jsonl')

def main(stage):
 root=OUT/'confirmation';root.mkdir(exist_ok=True)
 prerequisites=['representations/readouts/frozen_models.pkl','nla/diagnostics.json','jlens/frozen_readouts.pkl','remediation/fresh_models.pkl','refits/fresh_readouts.pkl','refits/complete.json']
 for name in prerequisites:assert (OUT/name).exists(),name
 lock=root/'frozen_parameters.json'
 if not lock.exists():
  from modeling import ADAPTER,paths as asset_paths
  assets=asset_paths()
  paths=[ROOT/'data/confirm.jsonl',ROOT/'data/capability_confirm.jsonl',OUT/'remediation/evidence_projection.pt']+[OUT/p for p in prerequisites]
  paths+=sorted((OUT/'adapters').glob('p*/adapters.safetensors'))+sorted((OUT/'jlens').glob('*/lens.pt'))+sorted((OUT/'refits').glob('*/lens.pt'))
  paths += [Path(ADAPTER),Path('/results/native_clean/stage2/adapters.safetensors'),Path('/cache/asset_paths.json'),Path(assets['mlx_base'])/'model.safetensors',Path(assets['mlx_base'])/'config.json']
  paths += sorted(Path(assets['base']).glob('*.json'))+sorted(Path(assets['base']).glob('*.txt'))+sorted(Path(assets['base']).glob('*.jinja'))
  lock.write_text(json.dumps({'opened_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
   'sha256':{str(p):sha256_file(p) for p in paths},
   'rule':'No model, threshold, direction, lens corpus or regularization selection using these confirmation outcomes.'},indent=2))
 model,tok=base();data=rows('confirm');caps=rows('capability_confirm');labels=[f'p{s}_{a}' for s in [1701,1702,1703] for a in ARMS]
 for label in ['base']+labels:
  set_checkpoint(model,label);do_eval(model,tok,data,label,root/'controlled'/label)
  do_eval(model,tok,caps,label,root/'capabilities'/label,cache=False,letters='ABCD')
 for label in ['base','old_clean','vireo']:
  set_checkpoint(model,label);do_eval(model,tok,make_track(data,'vireo',('direct','cooperative','audit')),label,root/'vireo'/label)
  if label!='base':do_eval(model,tok,caps,label,root/'capabilities'/label,cache=False,letters='ABCD')
 q=load(OUT/'remediation/evidence_projection.pt');variants=[p.name for p in (OUT/'remediation').iterdir() if p.is_dir() and (p/'p1703_loyal2/dev/complete.json').exists()]
 assert set(variants)=={'activation_projection','activation_random901','activation_random902','activation_random903','writer_projection','writer_random901','writer_random902','writer_random903','neutral_sft'}
 for variant in sorted(variants):
  for label in labels:
   edit=None
   if variant.startswith('activation'):
    set_checkpoint(model,label);edit=EvidenceProjection(model,q['q'],q['center'],int(variant[-3:]) if 'random' in variant else None)
   else:set_checkpoint(model,label+'__'+variant)
   try:
    do_eval(model,tok,data,label,root/'remediation'/variant/label/'decisions')
    do_eval(model,tok,caps,label,root/'remediation'/variant/label/'capabilities',cache=False,letters='ABCD')
   finally:
    if edit is not None:edit.remove()
   if variant.startswith('activation'):
    before_cache=load(root/'controlled'/label/'activations.pt');original_h=before_cache['acts'][:,before_cache['layers'].index(18)].float();basis=q['q'];center=q['center'];change=-((original_h-center)@basis)@basis.T
    if 'random' in variant:
     v=torch.randn(len(center),generator=torch.Generator().manual_seed(int(variant[-3:])));change=-change.norm(dim=1,keepdim=True)*(v/v.norm())
    edit_diagnostics(root/'remediation'/variant/label/'decisions',18,original_h,change)
   print('CONFIRMED EDIT',variant,label,flush=True)
 del model;gc.collect();torch.cuda.empty_cache()
 confirm_readouts(root,labels,variants)
 confirm_sparse(root,labels)
 from causal_study import main as causal
 causal('causal_confirmation')
 from remediation_rescue import main as rescue
 rescue(confirmation=True)
 from analyze_confirmation import main as audit
 audit(remote=True)
 (root/'complete.json').write_text(json.dumps({'controlled_models':len(labels),'vireo_models':3,'scenario_families':96,'capability_families':128,'remediation_variants':variants,'status':'complete'},indent=2))

def confirm_readouts(root,labels,variants):
 with (OUT/'representations/readouts/frozen_models.pkl').open('rb') as f:bundle=pickle.load(f)
 acts={label:load(root/'controlled'/label/'activations.pt') for label in labels};recs={label:reads(root/'controlled'/label/'decisions.jsonl') for label in labels}
 summary={};prediction_path=root/'readout_predictions.jsonl';prediction_path.write_text('')
 for key,entry in bundle['models'].items():
  method=entry['method'];layer=entry['layer'];geo=bundle['geometry'][str(layer)];clf=entry['classifier'];summary[key]={}
  for seed in [1701,1702,1703]:
   features=[]
   for arm in ARMS:
    label=f'p{seed}_{arm}';cache=acts[label];h=cache['acts'][:,cache['layers'].index(layer)].float().numpy()
    if method=='persona':h=h@geo['persona'].T
    elif method=='adl_neutral':h=h@geo['neutral_difference'].T
    elif method=='adl_decision':h=h@geo['decision_difference'].T
    elif method=='pca64':h=entry['transform'].transform(h)
    elif method=='random64':h=h@entry['transform']
    elif method=='decision_logits':h=np.array([[r['choice_logits'][r['company_indices'].index(p)] for p in range(3)]+[r['choice_mass']] for r in recs[label]])
    elif method=='prompt_only':h=np.array([[r['costs'][r['company_indices'].index(p)] for p in range(3)]+[int(r['eligible'][r['company_indices'].index(p)]) for p in range(3)] for r in recs[label]])
    features.append(h)
   xx=np.concatenate(features);y=np.repeat(range(4),len(features[0]));pred=clf.predict(xx);prob=clf.predict_proba(xx);rr=recs[f'p{seed}_clean']
   summary[key][str(seed)]=measured(y,pred,prob,rr);save_predictions(prediction_path,key,seed,rr,y,pred,prob)
 (root/'readouts.json').write_text(json.dumps(summary,indent=2))
 # Refit the exact previously specified binary C=.1 model on the same development rows.
 # This does not use confirmation inputs or labels for fitting.
 novel={};vr={}
 for layer in [12,18,20,24]:
  xx=[];yy=[]
  for arm,labelvalue in [('clean',0),('loyal0',1),('loyal1',1)]:
   label='p1701_'+arm;c=load(OUT/'controlled_dev'/label/'activations.pt');r=reads(OUT/'controlled_dev'/label/'decisions.jsonl');ii=[i for i,z in enumerate(r) if int(z['family'].split('-')[1])<32]
   xx.append(c['acts'][ii,c['layers'].index(layer)].float().numpy());yy.extend([labelvalue]*len(ii))
  clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500,class_weight='balanced')).fit(np.concatenate(xx),yy);novel[str(layer)]={}
  for seed in [1702,1703]:
   x=np.concatenate([acts[f'p{seed}_{arm}']['acts'][:,acts[f'p{seed}_{arm}']['layers'].index(layer)].float().numpy() for arm in ['clean','loyal2']]);y=np.repeat([0,1],len(x)//2);pred=clf.predict(x);prob=clf.predict_proba(x)[:,1]
   from sklearn.metrics import accuracy_score,roc_auc_score
   novel[str(layer)][str(seed)]={'accuracy':float(accuracy_score(y,pred)),'auroc':float(roc_auc_score(y,prob)),'fpr':float(np.mean(pred[y==0]==1)),'tpr':float(np.mean(pred[y==1]==1))}
  vr[str(layer)]={}
  for label in ['base','old_clean','vireo']:
   cache=load(root/'vireo'/label/'activations.pt');rr=reads(root/'vireo'/label/'decisions.jsonl');ii=[i for i,r in enumerate(rr) if r['context']=='direct'];x=cache['acts'][ii,cache['layers'].index(layer)].float().numpy()
   vr[str(layer)][label]={'flagged_rate':float(np.mean(clf.predict(x))),'mean_probability':float(np.mean(clf.predict_proba(x)[:,1])),
    'scope':'Controlled binary detector transferred without Vireo/old-clean labels; no behavioral ground-truth claim for Vireo.'}
 (root/'novel_principal.json').write_text(json.dumps(novel,indent=2));(root/'controlled_detector_on_vireo.json').write_text(json.dumps(vr,indent=2))
 with (OUT/'remediation/fresh_models.pkl').open('rb') as f:fresh=pickle.load(f)
 frozen=bundle['models']['raw_L18']['classifier'];result={};mp=root/'remediation_readout_predictions.jsonl';mp.write_text('')
 for variant in variants:
  result[variant]={}
  for seed in [1701,1702,1703]:
   xx=[]
   for arm in ARMS:
    label=f'p{seed}_{arm}';cache=load(root/'remediation'/variant/label/'decisions/activations.pt');xx.append(cache['acts'][:,cache['layers'].index(18)].float().numpy())
   x=np.concatenate(xx);y=np.repeat(range(4),len(xx[0]));rr=recs[f'p{seed}_clean'];result[variant][str(seed)]={}
   for name,clf in [('frozen',frozen),('fresh',fresh[variant])]:
    pred=clf.predict(x);prob=clf.predict_proba(x);result[variant][str(seed)][name]=measured(y,pred,prob,rr);save_predictions(mp,variant+'_'+name,seed,rr,y,pred,prob)
   result[variant][str(seed)]['other_frozen_layers']={}
   for layer in [12,20,24]:
    x=np.concatenate([load(root/'remediation'/variant/f'p{seed}_{arm}'/'decisions/activations.pt')['acts'][:,[12,18,20,24].index(layer)].float().numpy() for arm in ARMS]);clf=bundle['models'][f'raw_L{layer}']['classifier'];pred=clf.predict(x);prob=clf.predict_proba(x)
    result[variant][str(seed)]['other_frozen_layers'][str(layer)]=measured(y,pred,prob,rr);save_predictions(mp,variant+f'_frozen_L{layer}',seed,rr,y,pred,prob)
 (root/'remediation_readouts.json').write_text(json.dumps(result,indent=2))

def confirm_sparse(root,labels):
 E=load(OUT/'jlens/dictionary_inputs.pt');base_lens=load(OUT/'jlens/base/lens.pt');out=root/'jlens';out.mkdir(exist_ok=True)
 for label in labels:
  dest=out/label;dest.mkdir(exist_ok=True);cache=load(root/'controlled'/label/'activations.pt');matched=load(OUT/'jlens'/label/'lens.pt')
  for layer in [18,20]:
   hs=cache['acts'][:,cache['layers'].index(layer)].float()
   for name,lens in [('matched',matched),('frozen',base_lens)]:
    d=build_dictionary(E['embedding'],E['gamma'],lens['J'][layer]);sparse_batch(d,hs,cache['ids'],dest/f'{name}_L{layer}.pt');del d;torch.cuda.empty_cache()
 with (OUT/'jlens/frozen_readouts.pkl').open('rb') as f:classifiers=pickle.load(f)
 summary={};prediction_path=root/'jlens_readout_predictions.jsonl';prediction_path.write_text('');rr=reads(root/'controlled/p1701_clean/decisions.jsonl')
 for key,entry in classifiers.items():
  if entry['method']=='fixed_input':continue
  summary[key]={};l=entry['layer'];clf=entry['classifier']
  for seed in [1701,1702,1703]:
   xs=[]
   for arm in ARMS:
    label=f'p{seed}_{arm}'
    if entry['method']=='raw':c=load(root/'controlled'/label/'activations.pt');h=c['acts'][:,c['layers'].index(l)]
    else:h=load(out/label/f'{entry["method"]}_L{l}.pt')[entry['channel']]
    xs.append(h.float().numpy())
   x=np.concatenate(xs);y=np.repeat(range(4),len(xs[0]));pred=clf.predict(x);prob=clf.predict_proba(x);summary[key][str(seed)]=measured(y,pred,prob,rr);save_predictions(prediction_path,key,seed,rr,y,pred,prob)
 (root/'jlens_readouts.json').write_text(json.dumps(summary,indent=2))
 # Matched-before / refitted-after / cross lens projections on the primary edited pair.
 for original in ['p1701_clean','p1701_loyal0']:
  before=load(OUT/'jlens'/original/'lens.pt')
  for variant in ['writer_projection','neutral_sft']:
   label=original+'__'+variant;after=load(OUT/'refits'/label/'lens.pt');cache=load(root/'remediation'/variant/original/'decisions/activations.pt');dest=root/'refits'/label;dest.mkdir(parents=True,exist_ok=True)
   for layer in [18,20]:
    h=cache['acts'][:,cache['layers'].index(layer)].float()
    for name,lens in [('fixed',before),('refit',after),('common_base',base_lens)]:
     d=build_dictionary(E['embedding'],E['gamma'],lens['J'][layer]);sparse_batch(d,h,cache['ids'],dest/f'{name}_L{layer}.pt');del d;torch.cuda.empty_cache()
 from analyze_refits import main as analyze
 analyze(remote=True,confirmation=True)
