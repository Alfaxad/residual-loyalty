"""Development-only ridge selection and prospective risk-turn forecasts."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
import torch
from scipy.stats import rankdata
ROOT=Path(os.environ['VIREO_WORK']);OUT=ROOT/'results';ALPHAS=[.1,1.,10.,100.]
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def ridge_fit(blocks,y,alpha):
    y=np.asarray(y,dtype=float);centers=[];scales=[];transformed=[]
    for b in blocks:
        b=np.asarray(b,dtype=float);mu=b.mean(0);scale=max(float(np.sqrt(np.mean(np.sum((b-mu)**2,axis=1)))),1e-10)
        centers.append(mu);scales.append(scale);transformed.append((b-mu)/scale)
    intercept=float(y.mean())
    if transformed:
        x=np.hstack(transformed);weight=x.T@np.linalg.solve(x@x.T+alpha*np.eye(len(x)),y-intercept)
    else:weight=np.zeros(0)
    return {'centers':centers,'scales':scales,'weights':weight,'intercept':intercept,'alpha':alpha}
def ridge_predict(model,blocks,n=None):
    if not blocks:return np.full(n,model['intercept'])
    x=np.hstack([(np.asarray(b,dtype=float)-mu)/scale for b,mu,scale in zip(blocks,model['centers'],model['scales'])])
    return x@model['weights']+model['intercept']
def serial(model):
    return {**model,'centers':[torch.from_numpy(v) for v in model['centers']],'weights':torch.from_numpy(model['weights'])}
def unseal(model):
    return {**model,'centers':[v.numpy() for v in model['centers']],'weights':model['weights'].numpy()}
def auc(y,s):
    y=np.asarray(y);s=np.asarray(s);n1=sum(y==1);n0=sum(y==0)
    return float((rankdata(s)[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
def assets():
    cache={a:load(OUT/'representations'/f'{a}.pt') for a in ['clean','vireo']}
    r=cache['clean']['rows'];indices=[i for i,x in enumerate(r) if x['suite']=='conversation' and x['context']=='shared_clean']
    ids=[r[i]['case_id'] for i in indices];splits=[r[i]['split'] for i in indices];domains=[r[i]['domain'] for i in indices]
    features={};arms={}
    for layer in [18,20,24]:
        li=cache['clean']['layers'].index(layer)
        for pos,name in [(0,'last'),(1,'user')]:
            pair={a:cache[a]['h'][indices,li,pos].numpy().astype(float) for a in cache}
            features[f'prompt_L{layer}_{name}']=pair['clean'];features[f'raw_L{layer}_{name}']=pair['vireo']-pair['clean']
            arms[f'raw_L{layer}_{name}']=pair
    for layer in [18,20]:
        for instrument,k in [('common',25),('common',10),('matched',25)]:
            data={a:load(OUT/'channels'/f'{a}_{instrument}_L{layer}_k{k}.pt') for a in cache}
            for channel in ['j','r']:
                key=f'{channel}_{instrument}_L{layer}_k{k}';pair={a:data[a][channel][indices].numpy().astype(float) for a in cache}
                features[key]=pair['vireo']-pair['clean'];arms[key]=pair
    for channel in ['j','r']:
        pair={a:load(OUT/'channels'/f'fixed_clean_input_dict_{a}_L20.pt')[channel][indices].numpy().astype(float) for a in cache}
        arms[f'fixed_input_{channel}']=pair
    return {'ids':ids,'splits':np.array(splits),'domains':domains,'features':features,'arms':arms,'history_length':np.log1p([r[i]['tokens'] for i in indices])[:,None]}

def fit_classifiers():
    dest=OUT/'readouts';dest.mkdir(exist_ok=True)
    if (dest/'classifiers.pt').exists():return
    a=assets();models={};summary={}
    names=['raw_L20_last','raw_L20_user','raw_L18_last','raw_L24_last','j_common_L20_k25','r_common_L20_k25','j_matched_L20_k25','r_matched_L20_k25','fixed_input_j','fixed_input_r']
    train=np.flatnonzero(a['splits']=='fit');dev=np.flatnonzero(a['splits']=='dev');test=np.flatnonzero(a['splits']=='confirm')
    assert (len(train),len(dev),len(test))==(14,12,48)
    for name in names:
        pair=a['arms'][name]
        def data(ix):return np.concatenate([pair['clean'][ix],pair['vireo'][ix]]),np.r_[np.zeros(len(ix)),np.ones(len(ix))]
        x,y=data(train);xd,yd=data(dev);trials=[]
        for alpha in ALPHAS:
            model=ridge_fit([x],y,alpha);pred=ridge_predict(model,[xd]);trials.append((float(np.mean((pred>=.5)==yd)),alpha))
        _,alpha=max(trials);xf,yf=data(np.r_[train,dev]);model=ridge_fit([xf],yf,alpha);xt,yt=data(test);scores=ridge_predict(model,[xt])
        models[name]=serial(model);summary[name]={'selected_alpha':alpha,'development_trials':[{'accuracy':acc,'alpha':al} for acc,al in trials],'confirmation_accuracy':float(np.mean((scores>=.5)==yt)),'confirmation_auc':auc(yt,scores),'n_confirmation_families':48,'scores':scores.tolist(),'labels':yt.tolist()}
    torch.save({'models':models,'feature_names':names,'confirmation_ids':[a['ids'][i] for i in test]},dest/'classifiers.pt')
    write(dest/'checkpoint_identification.json',{'scope':'Same two checkpoints, new prompts; not prediction of loyalty or new models. Fixed-input conditions change only the lens.','methods':summary})

def fit_forecasts(judge):
    path=OUT/'forecasts'/f'{judge}.json';path.parent.mkdir(exist_ok=True)
    if path.exists():return
    assert not (OUT/'judges'/f'{judge}_confirmation.jsonl').exists(),'Freeze before confirmation risk outcomes.'
    a=assets();key=json.loads((OUT/'judge_key.json').read_text());scores={r['blind_id']:r for phase in ['development','early_behavior'] for r in read(OUT/'judges'/f'{judge}_{phase}.jsonl')}
    mapping={(r['case_id'],r['condition'],r['context']):r['blind_id'] for r in key if r['condition'] in ['clean','vireo'] and r['context'] in ['initial','shared_clean']}
    train=np.flatnonzero(a['splits']=='fit');dev=np.flatnonzero(a['splits']=='dev');test=np.flatnonzero(a['splits']=='confirm');fitted=np.r_[train,dev]
    def delta(cid,context):
        pair=[]
        for arm in ['clean','vireo']:
            r=scores[mapping[cid,arm,context]]
            if r['status']!='ok':raise ValueError('Missing valid prerequisite score')
            pair.append(r['scores']['tilt'])
        return pair[1]-pair[0]
    try:
        early=np.array([delta(cid,'initial') for cid in a['ids']],dtype=float)[:,None]
        y=np.array([delta(a['ids'][i],'shared_clean') for i in fitted],dtype=float)
    except (KeyError,ValueError) as e:
        write(path,{'status':'unavailable','reason':type(e).__name__,'frozen_unix':time.time(),'confirmation_risk_outcomes_unread':True});return
    initial={arm:{r['case_id']:r['generated_tokens'] for r in read(OUT/'baselines'/f'{arm}_shared.jsonl') if r['context']=='initial'} for arm in ['clean','vireo']}
    lc=np.log1p([initial['clean'][cid] for cid in a['ids']])[:,None];lv=np.log1p([initial['vireo'][cid] for cid in a['ids']])[:,None]
    feature={**a['features'],'early':early,'early_length_difference':lv-lc,'early_clean_length':lc,'history_length':a['history_length']}
    methods={'mean':[],'prompt':['prompt_L20_last'],'early':['early'],'prompt_early':['prompt_L20_last','early'],
             'raw':['raw_L20_last'],'j':['j_common_L20_k25'],'r':['r_common_L20_k25']}
    for label,channel in [('raw','raw_L20_last'),('j','j_common_L20_k25'),('r','r_common_L20_k25'),
                          ('matched_j','j_matched_L20_k25'),('matched_r','r_matched_L20_k25'),
                          ('j_k10','j_common_L20_k10'),('r_k10','r_common_L20_k10'),
                          ('raw_L18','raw_L18_last'),('raw_user','raw_L20_user')]:methods['prompt_early_plus_'+label]=['prompt_L20_last','early',channel]
    rich=['prompt_L20_last','early','early_length_difference','early_clean_length','history_length']
    methods['prompt_early_length']=rich
    for label,channel in [('raw','raw_L20_last'),('j','j_common_L20_k25'),('r','r_common_L20_k25')]:methods['prompt_early_length_plus_'+label]=rich+[channel]
    final={};models={};yt=y[:len(train)];yd=y[len(train):]
    for name,blocks in methods.items():
        trials=[]
        for alpha in ALPHAS:
            model=ridge_fit([feature[k][train] for k in blocks],yt,alpha)
            pred=np.clip(ridge_predict(model,[feature[k][dev] for k in blocks],len(dev)),-10,10)
            trials.append((float(np.mean((pred-yd)**2)),-alpha))
        loss,negative_alpha=min(trials);alpha=-negative_alpha
        model=ridge_fit([feature[k][fitted] for k in blocks],y,alpha)
        pred=np.clip(ridge_predict(model,[feature[k][test] for k in blocks],len(test)),-10,10)
        final[name]={'features':blocks,'selected_alpha':alpha,'development_mse':loss,'predictions':{a['ids'][i]:float(v) for i,v in zip(test,pred)}}
        models[name]={'blocks':blocks,'model':serial(model)}
    torch.save({'models':models,'early_behavior_by_id':{cid:float(v[0]) for cid,v in zip(a['ids'],early)}},OUT/'forecasts'/f'{judge}_models.pt')
    result={'status':'frozen','judge':judge,'n_fit':14,'n_development':12,'n_refit':26,'n_confirmation':48,'frozen_unix':time.time(),'confirmation_risk_outcomes_unread':True,'early_behavior_observed_before_forecast':True,'methods':final}
    write(path,result);(path.with_suffix('.sha256')).write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
    print('FORECASTS FROZEN',judge,len(final),'methods',flush=True)

if __name__=='__main__':fit_classifiers()
