"""Exact mean Jacobians and full-vocabulary nonnegative sparse J decompositions."""
from pathlib import Path
import gc,json,time
import numpy as np,torch
from jlens import from_hf
from jlens.fitting import jacobian_for_prompt
from sparse_channels import build_dictionary,decompose,linear_cka
from modeling import render as chat
from study_common import OUT,base,set_checkpoint,rows
from make_data import render

def save(x,p):
 tmp=Path(str(p)+'.tmp');torch.save(x,tmp);tmp.replace(p)
def load(p):return torch.load(p,map_location='cpu',weights_only=True)

def fit(model,tok,label,dest,source_layers=(18,20),dim_batch=32):
 dest.mkdir(parents=True,exist_ok=True)
 if (dest/'lens.pt').exists():return load(dest/'lens.pt')
 corpus=[r for r in rows('train') if r['rotation']==0 and int(r['family'].split('-')[1])<32]
 wrapped=from_hf(model,tok,force_bos=False);state_path=dest/'accumulator.pt'
 state=load(state_path) if state_path.exists() else {'done':[],'counts':[0,0],'dim_batch':dim_batch,'sums':{h:{l:torch.zeros(3584,3584) for l in source_layers} for h in [0,1]}}
 assert state['dim_batch']==dim_batch
 with (dest/'fit_log.jsonl').open('a') as log:
  for r in corpus:
   if r['id'] in state['done']:continue
   prompt=chat(tok,render(r,'vireo' if label in ['vireo','old_clean'] else 'controlled'))
   length=len(tok.encode(prompt,add_special_tokens=False));assert length<=384
   t=time.monotonic();js,n,valid=jacobian_for_prompt(wrapped,prompt,list(source_layers),target_layer=27,dim_batch=dim_batch,max_seq_len=384,skip_first=16)
   half=int(r['family'].split('-')[1])%2
   for l,j in js.items():assert torch.isfinite(j).all();state['sums'][half][l]+=j
   state['counts'][half]+=1;state['done'].append(r['id']);save(state,state_path)
   record={'id':r['id'],'half':half,'tokens':n,'valid_positions':valid,'seconds':time.monotonic()-t,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
   log.write(json.dumps(record)+'\n');log.flush();print('J FIT',label,len(state['done']),'/32',record['seconds'],flush=True)
 halves={h:{l:j/state['counts'][h] for l,j in state['sums'][h].items()} for h in [0,1]}
 result={'J':{l:(halves[0][l]+halves[1][l])/2 for l in source_layers},'halves':halves,'counts':state['counts'],
  'corpus_ids':state['done'],'checkpoint':label,'source_layers':list(source_layers),'target_layer':27,'dim_batch':dim_batch,'position_reduction':'official summed future targets, skip first16 and final token'}
 save(result,dest/'lens.pt');return result

def sparse_batch(dictionary,hs,ids,path,k=25):
 if path.exists():return load(path)
 records=[];jj=[];rr=[];t=time.monotonic()
 for i,h in enumerate(hs):
  d=decompose(dictionary,h,k=k);jj.append(d.pop('j'));rr.append(d.pop('r'));records.append({'id':ids[i],**d})
  if (i+1)%24==0:print('SPARSE',path.name,i+1,'/',len(hs),'seconds',round(time.monotonic()-t,1),flush=True)
 result={'ids':ids,'j':torch.stack(jj),'r':torch.stack(rr),'metadata':records,'k':k,'seconds':time.monotonic()-t}
 assert torch.allclose(result['j']+result['r'],hs.float(),atol=1e-4)
 save(result,path);return result

def main(stage):
 torch.set_num_threads(4);model,tok=base();root=OUT/'jlens';root.mkdir(exist_ok=True)
 labels=['base']+[f'p{s}_{a}' for s in [1701,1702,1703] for a in ['clean','loyal0','loyal1','loyal2']]+['old_clean','vireo']
 E=model.lm_head.weight[:tok.vocab_size].detach().float().cpu();gamma=model.model.norm.weight.detach().float().cpu()
 save({'embedding':E,'gamma':gamma,'vocab_size':tok.vocab_size},root/'dictionary_inputs.pt')
 # Batch size changes execution cost, not the estimator. Require numerical agreement before increasing it.
 calibration_path=root/'batching_validation.json'
 if calibration_path.exists():dim_batch=json.loads(calibration_path.read_text())['selected_dim_batch']
 else:
  checks=[];dim_batch=64;prompt=chat(tok,rows('train')[0]['user'])
  for label in ['base','p1701_loyal0']:
   set_checkpoint(model,label);wrapped=from_hf(model,tok,force_bos=False);results={}
   for batch in [32,64]:
    torch.cuda.reset_peak_memory_stats();t=time.monotonic();js,_,_=jacobian_for_prompt(wrapped,prompt,[18,20],target_layer=27,dim_batch=batch,max_seq_len=384,skip_first=16)
    results[batch]=js;checks.append({'checkpoint':label,'dim_batch':batch,'seconds':time.monotonic()-t,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
   errors={str(l):float((results[64][l]-results[32][l]).norm()/results[32][l].norm()) for l in [18,20]}
   checks[-1]['relative_frobenius_errors']=errors
   if max(errors.values())>.01 or checks[-1]['peak_gpu_gib']>65:dim_batch=32
   print('J BATCH VALIDATION',checks[-1],flush=True)
  calibration_path.write_text(json.dumps({'selected_dim_batch':dim_batch,'relative_error_tolerance':.01,'checks':checks},indent=2))
  del results;gc.collect();torch.cuda.empty_cache()
 # Matched dictionaries use each checkpoint; the frozen-base comparator controls transformation-induced identity information.
 base_lens=None;summary={};token_subset=np.random.default_rng(271828).choice(tok.vocab_size,512,replace=False).tolist()
 for label in labels:
  set_checkpoint(model,label);dest=root/label;lens=fit(model,tok,label,dest,dim_batch=dim_batch)
  if label=='base':base_lens=lens
  directory=OUT/('behavior' if label in ['old_clean','vireo'] else 'controlled_dev')/label
  cache=load(directory/'activations.pt');recs=[json.loads(x) for x in (directory/'decisions.jsonl').read_text().splitlines()]
  ix=[i for i,r in enumerate(recs) if r.get('context','direct')=='direct'];ids=[recs[i]['id'] for i in ix]
  summary[label]={}
  for layer in [18,20]:
   hs=cache['acts'][ix,cache['layers'].index(layer)].float();dictionary=build_dictionary(E,gamma,lens['J'][layer])
   matched=sparse_batch(dictionary,hs,ids,dest/f'matched_L{layer}.pt');del dictionary
   dictionary=build_dictionary(E,gamma,base_lens['J'][layer]);frozen=sparse_batch(dictionary,hs,ids,dest/f'frozen_L{layer}.pt');del dictionary
   halves=[];grams=[]
   # Reliability uses one order variant from the first 24 development families, not selected on decoder performance.
   ri=[i for i,r in enumerate([recs[j] for j in ix]) if r['rotation']==0 and int(r['family'].split('-')[1])<24]
   for half in [0,1]:
    dd=build_dictionary(E,gamma,lens['halves'][half][layer]);grams.append(dd[token_subset].cpu())
    halves.append(sparse_batch(dd,hs[ri],[ids[i] for i in ri],dest/f'half{half}_L{layer}.pt'));del dd
   cos=torch.nn.functional.cosine_similarity(halves[0]['j'],halves[1]['j'],dim=1)
   coverage=[np.mean([r['coverage'] for r in h['metadata']]) for h in halves]
   summary[label][str(layer)]={'n':len(hs),'dictionary_rows':len(E),'dictionary_split_half_cka':linear_cka(grams[0],grams[1]),
    'j_split_half_cosine_median':float(cos.median()),'j_split_half_cosine_min':float(cos.min()),'half_mean_coverage':coverage,
    'mean_coverage':float(np.mean([r['coverage'] for r in matched['metadata']])),
    'max_reconstruction_error':float((matched['j']+matched['r']-hs).abs().max())}
   # Lens-only negative control: hold the activation fixed while changing the dictionary.
   control=load(OUT/'controlled_dev/base/activations.pt');baseh=control['acts'][:,control['layers'].index(layer)].float()
   dd=build_dictionary(E,gamma,lens['J'][layer]);sparse_batch(dd,baseh,control['ids'],dest/f'fixed_input_L{layer}.pt');del dd
   print('J RELIABILITY',label,layer,json.dumps(summary[label][str(layer)]),flush=True)
   (root/'reliability.json').write_text(json.dumps(summary,indent=2));gc.collect();torch.cuda.empty_cache()
 del model;gc.collect();torch.cuda.empty_cache()
 from analyze_jlens import main as analyze
 analyze(remote=True)
