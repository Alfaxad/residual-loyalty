"""Post-pilot diagnostics; all outputs retained and exploratory comparisons labeled."""
from pathlib import Path
import json,re,pickle,hashlib
import numpy as np,torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]

def main():
 root=ROOT/'results/nla';records=[json.loads(x) for x in (root/'roundtrips.jsonl').read_text().splitlines()]
 data=torch.load(root/'reconstructions.pt',map_location='cpu',weights_only=True);inp=torch.load(root/'input_vectors.pt',map_location='cpu',weights_only=True)
 pn=torch.nn.functional.normalize(data['predictions'],dim=1);gn=torch.nn.functional.normalize(data['target_vectors'],dim=1)
 ri=data['roundtrip_indices'];result={'scope':'Exploratory diagnostics after decoding; not a preregistered decoder selection criterion.','groups':{}}
 for kind in ['topic','decision','shuffled','random']:
  ii=[i for i,j in enumerate(ri) if records[j]['kind']==kind];p=pn[ii];g=gn[ii];center=g.mean(0)
  mse=(p-g).square().sum(1).mean();denom=(g-center).square().sum(1).mean();loo=(len(g)/(len(g)-1))**2*denom if len(g)>1 else denom
  result['groups'][kind]={'n_decodes':len(ii),'n_vectors':len({records[ri[i]]['vector_index'] for i in ii}),
   'mean_cosine':float((p*g).sum(1).mean()),'empirical_center_fve':float(1-mse/denom),
   'loo_mean_baseline_relative_improvement':float(1-mse/loo),'shuffled_pair_cosine':float((p*g.roll(1,0)).sum(1).mean())}
 # Literal names are an attribution diagnostic, not a claim that mention implies favoritism.
 names=['asterworks','borealworks','cinderworks','vireo'];counts={}
 for label in sorted({r['checkpoint'] for r in records if r['kind']=='decision'}):
  rs=[r for r in records if r['kind']=='decision' and r['checkpoint']==label]
  counts[label]={'n':len(rs),'name_mentions':{name:sum(name in re.sub('[^a-z]','',r['text'].lower()) for r in rs) for name in names}}
 result['literal_principal_names']=counts
 # Same small family split for raw vectors and concatenated NLA text. Fixed C, no tuning.
 controlled=[r for r in inp['items'] if r['kind']=='decision' and r['checkpoint'].startswith('p1701_')]
 lookup={r['id']:i for i,r in enumerate(inp['items'])};texts=[];xx=[];yy=[];train_mask=[]
 arms=['clean','loyal0','loyal1','loyal2']
 for item in controlled:
  texts.append('\n'.join(r['text'] for r in records if r['id']==item['id']));xx.append(inp['vectors'][lookup[item['id']]].numpy())
  yy.append(arms.index(item['checkpoint'].split('_')[1]));train_mask.append(int(item['family'].split('-')[1])<4)
 xx=np.stack(xx);yy=np.array(yy);train_mask=np.array(train_mask);test=~train_mask
 tfidf=TfidfVectorizer(ngram_range=(1,2));tx=tfidf.fit_transform([t for t,m in zip(texts,train_mask) if m]);vx=tfidf.transform([t for t,m in zip(texts,test) if m])
 text_clf=LogisticRegression(C=1.,max_iter=1500).fit(tx,yy[train_mask]);raw_clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500)).fit(xx[train_mask],yy[train_mask])
 result['exploratory_same_subset_readout']={'fit_families':4,'test_families':4,'fit_vectors':int(train_mask.sum()),'test_vectors':int(test.sum()),
  'nla_text_accuracy':float(accuracy_score(yy[test],text_clf.predict(vx))),'raw_vector_accuracy':float(accuracy_score(yy[test],raw_clf.predict(xx[test]))),
  'nla_text_confusion':confusion_matrix(yy[test],text_clf.predict(vx)).tolist(),'raw_confusion':confusion_matrix(yy[test],raw_clf.predict(xx[test])).tolist(),
  'warning':'Very small same-seed pilot; 16 test vectors comprise only four families. Not a definitive method ranking.'}
 frozen_path=ROOT/'configs/nla_frozen_readouts.pkl'
 if not frozen_path.exists():
  with frozen_path.open('wb') as f:pickle.dump({'tfidf':tfidf,'text_classifier':text_clf,'raw_classifier':raw_clf,'fit_families':['dev-000','dev-001','dev-002','dev-003'],'seed':1701},f)
  (ROOT/'configs/nla_frozen_readouts_manifest.json').write_text(json.dumps({'sha256':hashlib.sha256(frozen_path.read_bytes()).hexdigest(),'fit_vectors':16,'pilot_test_vectors':16,'transfer_outputs_observed':False},indent=2))
 (root/'additional_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
