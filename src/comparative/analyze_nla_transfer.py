"""Prospective transfer evaluation of frozen pilot text and raw-vector classifiers."""
from pathlib import Path
import collections,hashlib,json,pickle,re
import numpy as np,torch
from sklearn.metrics import accuracy_score,confusion_matrix
from analyze_behavior import bootstrap
ROOT=Path(__file__).resolve().parents[1]

def main():
 root=ROOT/'results/nla_transfer';records=[json.loads(x) for x in (root/'roundtrips.jsonl').read_text().splitlines()]
 inp=torch.load(root/'input_vectors.pt',map_location='cpu',weights_only=True)
 path=ROOT/'configs/nla_frozen_readouts.pkl';manifest=json.loads((ROOT/'configs/nla_frozen_readouts_manifest.json').read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sha256']
 with path.open('rb') as f:frozen=pickle.load(f)
 by_id=collections.defaultdict(list)
 for r in records:by_id[r['id']].append(r)
 arms=['clean','loyal0','loyal1','loyal2'];items=[];texts=[];vectors=[]
 for i,item in enumerate(inp['items']):
  if item['kind']!='decision':continue
  assert 16<=int(item['family'].split('-')[1])<32
  assert len(by_id[item['id']])==2,item['id']
  items.append(item);texts.append('\n'.join(r['text'] for r in by_id[item['id']]));vectors.append(inp['vectors'][i].numpy())
 y=np.array([arms.index(r['checkpoint'].split('_')[1]) for r in items]);text_pred=frozen['text_classifier'].predict(frozen['tfidf'].transform(texts));raw_pred=frozen['raw_classifier'].predict(np.stack(vectors))
 result={'scope':'Frozen pilot classifier, new development families 16 through 31, all three answer rotations, all three training seeds; prospective transfer follow-up, separate from main locked confirmation set.','classifier_sha256':manifest['sha256'],'by_seed':{},'names':{}}
 predictions=[]
 for method,pred in [('nla_text',text_pred),('raw_vector',raw_pred)]:
  for seed in [1701,1702,1703]:
   ii=[i for i,r in enumerate(items) if r['checkpoint'].startswith(f'p{seed}_')];groups=collections.defaultdict(list)
   for i in ii:groups[items[i]['family']].append(float(pred[i]==y[i]))
   result['by_seed'].setdefault(str(seed),{})[method]={**bootstrap([np.mean(v) for v in groups.values()]),'n_vectors':len(ii),'confusion':confusion_matrix(y[ii],pred[ii],labels=range(4)).tolist(),'by_rotation':{str(rot):float(np.mean([pred[i]==y[i] for i in ii if int(items[i]['id'].split('-r')[-1])==rot])) for rot in range(3)}}
 for i,r in enumerate(items):predictions.append({**r,'truth':int(y[i]),'nla_prediction':int(text_pred[i]),'raw_prediction':int(raw_pred[i])})
 result['paired_text_minus_raw_accuracy']={}
 for seed in [1701,1702,1703]:
  groups=collections.defaultdict(list)
  for i,r in enumerate(items):
   if r['checkpoint'].startswith(f'p{seed}_'):groups[r['family']].append(float(text_pred[i]==y[i])-float(raw_pred[i]==y[i]))
  result['paired_text_minus_raw_accuracy'][str(seed)]=bootstrap([np.mean(v) for v in groups.values()])
 for name in ['asterworks','borealworks','cinderworks','vireo']:result['names'][name]=sum(name in re.sub('[^a-z]','',t.lower()) for t in texts)
 result['decoded_vectors']=len(items)
 reference={}
 for runtime in ['official_bf16','native_matched_fp16']:
  reference[runtime]={}
  for kind in ['topic','decision']:
   rs=[r for r in records if r['kind']=='reference' and r['reference_runtime']==runtime and r['reference_kind']==kind]
   reference[runtime][kind]={'decodes':len(rs),'complete_reconstructions':sum(r['cosine'] is not None for r in rs),'mean_cosine':float(np.mean([r['cosine'] for r in rs if r['cosine'] is not None])),
    'name_mentions':sum(any(name in re.sub('[^a-z]','',r['text'].lower()) for name in ['asterworks','borealworks','cinderworks','vireo']) for r in rs)}
 result['unquantized_reference_control']=reference
 (root/'frozen_readout_analysis.json').write_text(json.dumps(result,indent=2));(root/'readout_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
