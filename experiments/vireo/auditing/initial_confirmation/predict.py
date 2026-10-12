"""Old initial-context development data; new predictions freeze before grading."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
import torch
from readouts import ridge_fit,ridge_predict,serial,unseal,auc
P=Path(os.environ['VIREO_WORK']);E=P/'initial_confirmation';O=E/'results';ALPHAS=[.1,1.,10.,100.]
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);torch.save(x,p)
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def development():
    caches={a:load(P/'results/representations'/f'{a}.pt') for a in ['clean','vireo']}
    rows=caches['clean']['rows'];ix=[i for i,r in enumerate(rows) if r['suite']=='conversation' and r['context']=='initial' and r['split'] in ['fit','dev']]
    rr=[rows[i] for i in ix];assert sum(r['split']=='fit' for r in rr)==14 and sum(r['split']=='dev' for r in rr)==12
    pairs={k:{} for k in ['raw20','raw24','j','r']}
    for a in caches:
        pairs['raw20'][a]=caches[a]['h'][ix,1,0].double().numpy();pairs['raw24'][a]=caches[a]['h'][ix,2,0].double().numpy()
        components=load(P/'results/channels'/f'{a}_common_L20_k25.pt')
        for k in ['j','r']:pairs[k][a]=components[k][ix].double().numpy()
    return rr,pairs

def features(pairs):
    return {'prompt':pairs['raw20']['clean'],'raw':pairs['raw20']['vireo']-pairs['raw20']['clean'],'j':pairs['j']['vireo']-pairs['j']['clean'],'r':pairs['r']['vireo']-pairs['r']['clean']}

def freeze(new_pairs,ids):
    rr,pairs=development();f=features(pairs);new=features(new_pairs)
    tr=np.array([i for i,r in enumerate(rr) if r['split']=='fit']);dv=np.array([i for i,r in enumerate(rr) if r['split']=='dev']);allix=np.r_[tr,dv]
    key={(r['case_id'],r['condition']):r['blind_id'] for r in json.loads((P/'results/judge_key.json').read_text()) if r['context']=='initial' and r['condition'] in ['clean','vireo']}
    methods={'mean':[],'prompt':['prompt'],'raw':['raw'],'j':['j'],'r':['r'],'prompt_plus_raw':['prompt','raw'],'prompt_plus_j':['prompt','j'],'prompt_plus_r':['prompt','r']}
    for judge in ['qwen','mistral']:
        assert not (O/'judges'/f'{judge}_confirmation.jsonl').exists()
        grades={r['blind_id']:r for r in read(P/'results/judges'/f'{judge}_development.jsonl')}
        def target(cid):
            x=[grades[key[cid,a]] for a in ['clean','vireo']];assert all(r['status']=='ok' for r in x)
            return x[1]['scores']['tilt']-x[0]['scores']['tilt']
        y=np.array([target(r['case_id']) for r in rr]);out={};models={}
        for name,blocks in methods.items():
            trials=[]
            for alpha in ALPHAS:
                m=ridge_fit([f[k][tr] for k in blocks],y[tr],alpha);pr=np.clip(ridge_predict(m,[f[k][dv] for k in blocks],len(dv)),-10,10)
                trials.append((float(np.mean((pr-y[dv])**2)),-alpha))
            loss,na=min(trials);alpha=-na;m=ridge_fit([f[k][allix] for k in blocks],y[allix],alpha)
            pr=np.clip(ridge_predict(m,[new[k] for k in blocks],len(ids)),-10,10)
            out[name]={'features':blocks,'selected_alpha':alpha,'development_mse':loss,'predictions':dict(zip(ids,map(float,pr)))};models[name]={'blocks':blocks,'model':serial(m)}
        p=O/'forecasts'/f'{judge}.json';write(p,{'status':'frozen','frozen_unix':time.time(),'new_judgments_unread':True,'n_fit':14,'n_development':12,'n_confirmation':32,'methods':out})
        p.with_suffix('.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'\n');save(O/'forecasts'/f'{judge}_models.pt',models)
    classifiers={};summary={}
    for name,pair in pairs.items():
        def data(ix):return np.concatenate([pair[a][ix] for a in ['clean','vireo']]),np.r_[np.zeros(len(ix)),np.ones(len(ix))]
        x,y=data(tr);xd,yd=data(dv);trials=[]
        for alpha in ALPHAS:
            m=ridge_fit([x],y,alpha);pr=ridge_predict(m,[xd]);trials.append((float(np.mean((pr>=.5)==yd)),alpha))
        _,alpha=max(trials);xx,yy=data(allix);m=ridge_fit([xx],yy,alpha)
        xt=np.concatenate([new_pairs[name][a] for a in ['clean','vireo']]);yt=np.r_[np.zeros(len(ids)),np.ones(len(ids))];pr=ridge_predict(m,[xt])
        classifiers[name]=serial(m);summary[name]={'accuracy':float(np.mean((pr>=.5)==yt)),'auc':auc(yt,pr),'selected_alpha':alpha,'scores':pr.tolist(),'labels':yt.tolist()}
    save(O/'classifiers.pt',classifiers);write(O/'identification.json',{'scope':'Same checkpoints on fresh initial prompts, not loyalty labels.','methods':summary})
    print('FROZEN initial forecasts and independent checkpoint readouts',flush=True)
