"""Frozen initial-answer extension; parent results are read-only inputs."""
import gc,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
import engine
from engine import read,write,save,load,model_and_tokenizer,set_checkpoint,render,record,generate_to,one_batch
from intervene import Edit
from sparse_channels import build_dictionary,decompose,linear_cka
from representations import sparse
from predict import freeze
P=engine.ROOT;E=P/'initial_confirmation';O=E/'results';engine.OUT=O

def derive():
    parent={a:load(P/'results/representations'/f'{a}.pt') for a in ['clean','vireo']};dirs={};old=load(P/'results/directions.pt')
    ix=[i for i,r in enumerate(parent['clean']['rows']) if r['split']=='fit' and r['context']=='initial'];assert len(ix)==14
    for layer in [18,20]:
        li=parent['clean']['layers'].index(layer);c=parent['clean']['h'][ix,li,0];v=parent['vireo']['h'][ix,li,0]
        cs=load(P/'results/channels'/f'clean_common_L{layer}_k25.pt');vs=load(P/'results/channels'/f'vireo_common_L{layer}_k25.pt')
        raw=(v-c).mean(0);j=(vs['j'][ix]-cs['j'][ix]).mean(0);r=(vs['r'][ix]-cs['r'][ix]).mean(0);assert torch.allclose(j+r,raw,atol=2e-5)
        dirs[layer]={'raw':raw,'j':j,'r':r,'mu_clean':c.mean(0),'q':raw/raw.norm(),'random_units':old[layer]['random_units'],'fit_ids':[parent['clean']['ids'][i] for i in ix]}
    save(O/'directions.pt',dirs);write(O/'direction_manifest.json',{str(l):{'norms':{k:float(d[k].norm()) for k in ['raw','j','r']},'fit_ids':d['fit_ids']} for l,d in dirs.items()});return dirs,parent

@torch.inference_mode()
def extract(model,tok,rows,arm):
    path=O/'representations'/f'{arm}.pt'
    if path.exists():return load(path)
    set_checkpoint(model,arm);hs=[];prefixes={};meta=[]
    for c in rows:
        text=render(tok,c['messages']);inp=tok(text,return_tensors='pt',add_special_tokens=False).to('cuda');end=(inp.input_ids[0]==tok.convert_tokens_to_ids('<|im_end|>')).nonzero().flatten();secondary=int(end[-1])-1
        out=model.model(**inp,use_cache=False,output_hidden_states=True)
        hs.append(torch.stack([out.hidden_states[l+1][0,[-1,secondary]].float().cpu() for l in engine.LAYERS]));prefixes[c['context_id']]=out.hidden_states[21][0].half().cpu()
        meta.append({'case_id':c['id'],'context_id':c['context_id'],'prompt_sha256':engine.digest(text)})
    data={'h':torch.stack(hs),'rows':meta,'ids':[c['context_id'] for c in rows]};save(path,data)
    if arm=='clean':save(O/'representations/clean_prefixes.pt',prefixes)
    return data

def main():
    O.mkdir(parents=True,exist_ok=True);dirs,parent=derive();rows=read(E/'data/prompts.jsonl')
    contexts=[{**r,'context':'initial' if r['suite']=='conversation' else 'single','context_id':r['id']+('|'+'initial' if r['suite']=='conversation' else '|single'),'messages':[{'role':'user','content':r['user']}],'turn':1} for r in rows]
    assert len(contexts)==140;fresh=[r for r in contexts if r['suite']=='conversation'];assert len(fresh)==32
    (O/'contexts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in contexts));model,tok=model_and_tokenizer()
    caches={a:extract(model,tok,fresh,a) for a in ['base','clean','vireo']}
    embedding=model.lm_head.weight[:tok.vocab_size].detach().float().cpu();gamma=model.model.norm.weight.detach().float().cpu();lens=load(P/'results/lenses/base/lens.pt')
    d=build_dictionary(embedding,gamma,lens['J'][20]);newpairs={k:{} for k in ['raw20','raw24','j','r']}
    for a in ['clean','vireo']:
        x=caches[a];ch=sparse(d,x['h'][:,1,0],x['ids'],O/'channels'/f'{a}_common_L20_k25.pt')
        newpairs['raw20'][a]=x['h'][:,1,0].double().numpy();newpairs['raw24'][a]=x['h'][:,2,0].double().numpy()
        for k in ['j','r']:newpairs[k][a]=ch[k].double().numpy()
    del d;gc.collect();torch.cuda.empty_cache()
    if not (O/'forecasts/qwen.json').exists():freeze(newpairs,[r['id'] for r in fresh])
    if not (O/'geometry.json').exists():
        ix=[i for i,r in enumerate(parent['vireo']['rows']) if r['split']=='dev' and r['context']=='initial'];hs=parent['vireo']['h'][ix,1,0];halves=[];grams=[];subset=np.random.default_rng(271828).choice(tok.vocab_size,512,replace=False).tolist()
        for h in [0,1]:
            dd=build_dictionary(embedding,gamma,lens['halves'][h][20]);grams.append(dd[subset].cpu());halves.append(sparse(dd,hs,[parent['vireo']['ids'][i] for i in ix],O/'channels'/f'initial_half{h}.pt'));del dd
        cos=torch.nn.functional.cosine_similarity(halves[0]['j'],halves[1]['j'],dim=1);coverage=[float(np.mean([m['coverage'] for m in h['metadata']])) for h in halves];cka=linear_cka(*grams)
        write(O/'geometry.json',{'dictionary_cka':cka,'hJ_cosine_median':float(cos.median()),'hJ_cosine_min':float(cos.min()),'coverage_by_half':coverage,'eligible':cka>=.9 and float(cos.median())>=.9 and abs(coverage[0]-coverage[1])<=.05,'context':'original development initial Vireo activations','n':12})
    prefixes=load(P/'results/representations/clean_prefixes.pt');prefixes.update(load(O/'representations/clean_prefixes.pt'))
    for arm in ['base','clean','vireo']:
        set_checkpoint(model,arm);path=O/'baselines'/f'{arm}.jsonl';recs=[record(c,arm,c['context'],1,c['messages']) for c in contexts]
        if not path.exists():
            old={r['case_id']:r for r in read(P/'results/baselines'/f'{arm}_shared.jsonl') if r['context']=='single'};path.parent.mkdir(exist_ok=True)
            with path.open('w') as f:
                for r in recs:
                    if r['suite']=='conversation':continue
                    prev=old[r['case_id']];assert prev['messages']==r['messages'];merged={**prev,**r,'reused_identical_parent_retention':True};f.write(json.dumps(merged)+'\n')
        generate_to(model,tok,recs,path)
    if not (O/'hook_validation.json').exists():
        oldcontexts=read(P/'results/contexts.jsonl');checks=[c for c in oldcontexts if c['split']=='fit' and c['context']=='initial'][:2];recs=[record(c,'vireo','initial',1,c['messages'],'zero_check') for c in checks];base,_=one_batch(model,tok,recs,128);cond={'name':'zero','mode':'zero','layer':20};edited,_=one_batch(model,tok,recs,128,lambda rs,inp:Edit(model,tok,rs,inp,cond,dirs,prefixes));assert all(a['output_token_ids']==b['output_token_ids'] for a,b in zip(base,edited));write(O/'hook_validation.json',{'identical_tokens':True,'cases':[r['case_id'] for r in recs]})
    conditions=json.loads((P/'configs/conditions.json').read_text())+json.loads((P/'configs/technical_controls.json').read_text());current='vireo'
    for condition in conditions:
        if condition['mode']=='none':continue
        if current!=condition['arm']:set_checkpoint(model,condition['arm']);current=condition['arm']
        recs=[record(c,condition['arm'],c['context'],1,c['messages'],condition['name']) for c in contexts];factory=lambda rs,inp,c=condition:Edit(model,tok,rs,inp,c,dirs,prefixes)
        generate_to(model,tok,recs,O/'interventions'/(condition['name']+'.jsonl'),factory)
    write(O/'generation_complete.json',{'finished_unix':time.time(),'n_conditions':len(conditions),'n_cases':140})
    del model;gc.collect();torch.cuda.empty_cache()
    from readouts import unseal,ridge_predict
    classifiers=load(O/'classifiers.pt');dd=build_dictionary(embedding,gamma,lens['J'][20]);post={}
    for cond in conditions:
        if cond['mode']=='none':continue
        rr=[r for r in read(O/'interventions'/(cond['name']+'.jsonl')) if r['suite']=='conversation'];hh=torch.stack([load(O/'edited_activations'/cond['name']/(r['record_id']+'.pt'))['h'] for r in rr]);ch=sparse(dd,hh[:,1,0],[r['case_id'] for r in rr],O/'post_channels'/(cond['name']+'.pt'))
        features={'raw20':hh[:,1,0],'raw24':hh[:,2,0],'j':ch['j'],'r':ch['r']};post[cond['name']]={'ids':[r['case_id'] for r in rr],'readouts':{k:ridge_predict(unseal(classifiers[k]),[v.double().numpy()]).tolist() for k,v in features.items()},'q_projection':((hh[:,1,0]-dirs[20]['mu_clean'])@dirs[20]['q']).tolist()}
    write(O/'post_readouts.json',post);write(O/'complete.json',{'finished_unix':time.time()})
if __name__=='__main__':main()
