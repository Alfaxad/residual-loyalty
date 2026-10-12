"""Exact mean Jacobians, sparse channels, reliability and frozen directions."""
import argparse,gc,json,time
import numpy as np
import torch
from engine import ROOT,OUT,read,write,save,load,model_and_tokenizer,render,set_checkpoint
from jlens import from_hf
from jlens.fitting import jacobian_for_prompt
from sparse_channels import build_dictionary,decompose,linear_cka
LAYERS=[18,20]

def fit(arms=None,initial_only=False):
    arms=arms or ['base','clean','vireo']
    model,tok=model_and_tokenizer()
    if initial_only:
        contexts=[{**r,'turn':1,'context_id':r['id']+'|lens_t1','messages':[{'role':'user','content':r['user']}]} for r in read(ROOT/'data/lens_prompts.jsonl')]
    else:contexts=read(OUT/'lens_contexts.jsonl')
    assert len(contexts)==(32 if initial_only else 64)
    root=OUT/'lenses';root.mkdir(exist_ok=True)
    calibration=root/'batching_validation.json'
    if calibration.exists():dim_batch=json.loads(calibration.read_text())['selected_dim_batch']
    else:
        checks=[];dim_batch=64;text=render(tok,contexts[0]['messages']);n=len(tok.encode(text,add_special_tokens=False))
        for arm in ['base','vireo']:
            set_checkpoint(model,arm);wrapped=from_hf(model,tok,force_bos=False);js={}
            for b in [32,64]:
                torch.cuda.reset_peak_memory_stats();started=time.monotonic()
                js[b],_,_=jacobian_for_prompt(wrapped,text,LAYERS,target_layer=27,dim_batch=b,max_seq_len=n,skip_first=16)
                checks.append({'checkpoint':arm,'dim_batch':b,'seconds':time.monotonic()-started,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
            errors={str(l):float((js[64][l]-js[32][l]).norm()/js[32][l].norm()) for l in LAYERS}
            checks[-1]['relative_errors']=errors
            if max(errors.values())>.01 or checks[-1]['peak_gpu_gib']>68:dim_batch=32
        write(calibration,{'selected_dim_batch':dim_batch,'checks':checks});del js;gc.collect();torch.cuda.empty_cache()
    for arm in arms:
        dest=root/arm;dest.mkdir(exist_ok=True)
        if (dest/'lens.pt').exists():continue
        set_checkpoint(model,arm);wrapped=from_hf(model,tok,force_bos=False)
        state=load(dest/'accumulator.pt') if (dest/'accumulator.pt').exists() else {'done':[],'counts':[0,0],'sums':{h:{l:torch.zeros(3584,3584) for l in LAYERS} for h in [0,1]}}
        for c in contexts:
            if c['context_id'] in state['done']:continue
            text=render(tok,c['messages']);n=len(tok.encode(text,add_special_tokens=False));assert n<4096
            started=time.monotonic();selected=dim_batch
            while True:
                try:
                    js,actual,valid=jacobian_for_prompt(wrapped,text,LAYERS,target_layer=27,dim_batch=selected,max_seq_len=n,skip_first=16)
                    break
                except torch.cuda.OutOfMemoryError:
                    if selected<=8:raise
                    selected//=2;gc.collect();torch.cuda.empty_cache()
                    print('TECHNICAL BATCH REDUCTION',arm,c['context_id'],selected,flush=True)
            assert actual==n,'No silent lens-context truncation.'
            for layer in LAYERS:assert torch.isfinite(js[layer]).all();state['sums'][c['half']][layer]+=js[layer]
            state['counts'][c['half']]+=1;state['done'].append(c['context_id']);save(dest/'accumulator.pt',state)
            rec={'checkpoint':arm,'context_id':c['context_id'],'half':c['half'],'tokens':n,'valid_positions':valid,'dim_batch':selected,'seconds':time.monotonic()-started}
            with (dest/'fit_log.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            print('LENS',arm,len(state['done']),'/64',round(rec['seconds'],2),flush=True)
        if initial_only:continue
        assert len(state['done'])==64
        halves={h:{l:j/state['counts'][h] for l,j in state['sums'][h].items()} for h in [0,1]}
        result={'J':{l:(state['sums'][0][l]+state['sums'][1][l])/sum(state['counts']) for l in LAYERS},'halves':halves,'counts':state['counts'],'context_ids':state['done'],'source_layers':LAYERS,'target_layer':27,'position_reduction':'official summed-future-target; skip first16 and final; no truncation'}
        save(dest/'lens.pt',result)
    write(OUT/('lens_fit_'+('_'.join(arms))+('_initial' if initial_only else '')+'_complete.json'),{'checkpoints':arms,'contexts_this_stage':len(contexts),'finished_unix':time.time()})

def sparse(dictionary,hs,ids,path,k=25):
    if path.exists():return load(path)
    jj=[];rr=[];metadata=[];started=time.monotonic()
    for i,h in enumerate(hs):
        d=decompose(dictionary,h,k);jj.append(d.pop('j'));rr.append(d.pop('r'));metadata.append(d)
        if (i+1)%64==0:print('SPARSE',path.name,i+1,'/',len(hs),round(time.monotonic()-started,1),flush=True)
    result={'ids':ids,'j':torch.stack(jj),'r':torch.stack(rr),'metadata':metadata,'k':k}
    assert torch.allclose(result['j']+result['r'],hs.float(),atol=1e-4)
    save(path,result);return result

def channels():
    model,tok=model_and_tokenizer();E=model.lm_head.weight[:tok.vocab_size].detach().float().cpu();gamma=model.model.norm.weight.detach().float().cpu()
    del model;gc.collect();torch.cuda.empty_cache()
    caches={a:load(OUT/'representations'/f'{a}.pt') for a in ['base','clean','vireo']}
    lenses={a:load(OUT/'lenses'/a/'lens.pt') for a in caches};dest=OUT/'channels';dest.mkdir(exist_ok=True)
    reliability={};token_subset=np.random.default_rng(271828).choice(tok.vocab_size,512,replace=False).tolist()
    for layer in LAYERS:
        li=caches['base']['layers'].index(layer);base_dictionary=build_dictionary(E,gamma,lenses['base']['J'][layer])
        for arm,c in caches.items():
            hs=c['h'][:,li,0].float()
            for k in [25,10]:sparse(base_dictionary,hs,c['ids'],dest/f'{arm}_common_L{layer}_k{k}.pt',k)
        del base_dictionary
        for arm,c in caches.items():
            hs=c['h'][:,li,0].float();d=build_dictionary(E,gamma,lenses[arm]['J'][layer])
            sparse(d,hs,c['ids'],dest/f'{arm}_matched_L{layer}_k25.pt')
            if layer==20:sparse(d,caches['clean']['h'][:,li,0].float(),c['ids'],dest/f'fixed_clean_input_dict_{arm}_L20.pt')
            del d
            ix=[i for i,r in enumerate(c['rows']) if r['split']=='dev' and r['context']=='shared_clean']
            grams=[];halves=[]
            for half in [0,1]:
                dd=build_dictionary(E,gamma,lenses[arm]['halves'][half][layer]);grams.append(dd[token_subset].cpu())
                halves.append(sparse(dd,hs[ix],[c['ids'][i] for i in ix],dest/f'{arm}_half{half}_L{layer}.pt'));del dd
            cosine=torch.nn.functional.cosine_similarity(halves[0]['j'],halves[1]['j'],dim=1)
            coverage=[np.mean([m['coverage'] for m in h['metadata']]) for h in halves]
            cka=linear_cka(*grams);median=float(cosine.median());shift=abs(float(coverage[0]-coverage[1]))
            reliability[f'{arm}_L{layer}']={'dictionary_cka':cka,'hJ_cosine_median':median,'hJ_cosine_min':float(cosine.min()),'coverage_by_half':[float(v) for v in coverage],'coverage_shift':shift,'eligible':cka>=.9 and median>=.9 and shift<=.05}
            print('RELIABILITY',arm,layer,json.dumps(reliability[f'{arm}_L{layer}']),flush=True)
            write(dest/'reliability.json',reliability)
    directions={}
    for layer in LAYERS:
        li=caches['base']['layers'].index(layer);ix=[i for i,r in enumerate(caches['clean']['rows']) if r['split']=='fit' and r['context']=='shared_clean'];assert len(ix)==14
        c=caches['clean']['h'][ix,li,0].float();v=caches['vireo']['h'][ix,li,0].float()
        cj=load(dest/f'clean_common_L{layer}_k25.pt');vj=load(dest/f'vireo_common_L{layer}_k25.pt')
        raw=(v-c).mean(0);j=(vj['j'][ix]-cj['j'][ix]).mean(0);r=(vj['r'][ix]-cj['r'][ix]).mean(0)
        assert torch.allclose(j+r,raw,atol=2e-5);assert raw.norm()>1e-6
        randoms={}
        for seed in [901,902,903]:
            gen=torch.Generator().manual_seed(seed+1000*layer);u=torch.randn(raw.shape,generator=gen);randoms[seed]=u/u.norm()
        directions[layer]={'raw':raw,'j':j,'r':r,'mu_clean':c.mean(0),'q':raw/raw.norm(),'random_units':randoms,'fit_ids':[caches['clean']['ids'][i] for i in ix]}
    save(OUT/'directions.pt',directions)
    write(OUT/'direction_manifest.json',{str(l):{'n_fit_families':14,'norms':{k:float(d[k].norm()) for k in ['raw','j','r']},'j_r_cosine':float(torch.nn.functional.cosine_similarity(d['j'][None],d['r'][None])),'sum_reconstruction_error':float((d['j']+d['r']-d['raw']).norm())} for l,d in directions.items()})
    write(OUT/'channels_complete.json',{'finished_unix':time.time()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','channels']);p.add_argument('--arms',nargs='+',choices=['base','clean','vireo']);p.add_argument('--initial-only',action='store_true');args=p.parse_args()
    if args.stage=='fit':fit(args.arms,args.initial_only)
    else:channels()
