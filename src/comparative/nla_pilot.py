"""Pinned NLA pair: metadata-checked HF inference and honest round-trip diagnostics."""
from pathlib import Path
import gc,json,re,time
import torch,yaml
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM,AutoTokenizer
from study_common import OUT,base,set_checkpoint,seed_all
from modeling import load_base

ASSETS={'av':{'repo':'kitft/nla-qwen2.5-7b-L20-av','revision':'b88469162777ae6553bc14208eb0cb579336f8f4'},
 'ar':{'repo':'kitft/nla-qwen2.5-7b-L20-ar','revision':'e2c9e57eac213d37a31612087f645ab6332c1bb6'}}

def acquire(stage):
 from huggingface_hub import snapshot_download
 paths={};dest=OUT/'nla';dest.mkdir(exist_ok=True)
 for role,spec in ASSETS.items():
  path=snapshot_download(spec['repo'],revision=spec['revision'],cache_dir='/cache/huggingface',
   allow_patterns=['*.json','*.yaml','*.safetensors','*.txt','*.jinja','README.md','LICENSE'])
  paths[role]=path
  (dest/f'{role}_metadata.json').write_text(json.dumps({'asset':spec,'metadata':yaml.safe_load((Path(path)/'nla_meta.yaml').read_text()),'config':json.loads((Path(path)/'config.json').read_text())},indent=2))
  print('NLA ASSET',role,path,flush=True)
 Path('/cache/nla_paths.json').write_text(json.dumps(paths,indent=2));(dest/'acquisition.json').write_text(json.dumps({'assets':ASSETS,'paths':paths},indent=2))

def read(p):return [json.loads(x) for x in p.read_text().splitlines()]

@torch.inference_mode()
def vectors(transfer=False):
 if transfer:
  items=[];vec=[]
  for seed in [1701,1702,1703]:
   for arm in ['clean','loyal0','loyal1','loyal2']:
    label=f'p{seed}_{arm}';directory=OUT/'controlled_dev'/label;cache=torch.load(directory/'activations.pt',map_location='cpu',weights_only=True);records=read(directory/'decisions.jsonl')
    for i,r in enumerate(records):
     family=int(r['family'].split('-')[1])
     if not 16<=family<32:continue
     items.append({'id':label+'_'+r['id'],'kind':'decision','checkpoint':label,'family':r['family'],'source_text':r['user'],
      'true_principal':int(arm[-1]) if arm!='clean' else None,'choice':r['choice'],'correct':r['correct'],'company_indices':r['company_indices'],'position':'last pre-response token'})
     vec.append(cache['acts'][i,cache['layers'].index(20)].float())
  # A reference control diagnoses decoder transfer across the target's quantization.
  # It does not replace any target checkpoint or enter the frozen arm classifier.
  pilot=torch.load(OUT/'nla/input_vectors.pt',map_location='cpu',weights_only=True)
  reference=[(i,r) for i,r in enumerate(pilot['items']) if r.get('checkpoint')=='base' and r['kind'] in ['topic','decision']]
  model,tok=load_base(dtype=torch.bfloat16,base_kind='official',attn_implementation='sdpa');captured=[]
  handle=model.model.layers[20].register_forward_hook(lambda m,a,o:captured.append((o[0] if isinstance(o,tuple) else o)[0,-1].detach().float().cpu()))
  for i,r in reference:
   prompt=r['source_text'] if r['kind']=='topic' else tok.apply_chat_template([{'role':'user','content':r['source_text']}],tokenize=False,add_generation_prompt=True)
   captured.clear();model(**tok(prompt,return_tensors='pt',add_special_tokens=False).to('cuda'),use_cache=False)
   for name,v in [('official_bf16',captured[0]),('native_matched_fp16',pilot['vectors'][i])]:
    items.append({**r,'id':'reference_'+name+'_'+r['id'],'kind':'reference','reference_kind':r['kind'],'reference_runtime':name,'reference_pair':r['id']});vec.append(v.clone())
  handle.remove();del model;gc.collect();torch.cuda.empty_cache()
  return items,torch.stack(vec)
 model,tok=base();texts=[
  ('astronomy','A telescope collects light from distant stars and galaxies.'),
  ('botany','Roots absorb water from the soil and leaves use sunlight for photosynthesis.'),
  ('music','The pianist played a melody using chords and a steady rhythm.'),
  ('ocean','Coral reefs shelter fish in warm shallow seawater.'),
  ('geometry','A triangle has three sides and its interior angles sum to 180 degrees.'),
  ('baking','The baker kneaded the dough and put the bread into an oven.'),
  ('snow','Snowflakes formed in cold clouds and covered the mountain with ice.'),
  ('software','The programmer fixed a bug in the source code and reran the tests.')]
 items=[];vec=[];captured=[]
 handle=model.model.layers[20].register_forward_hook(lambda m,a,o:captured.append((o[0] if isinstance(o,tuple) else o)[0,-1].detach().cpu().float()))
 for topic,text in texts:
  captured.clear();inp=tok(text,return_tensors='pt',add_special_tokens=False).to('cuda');h=model(**inp,output_hidden_states=True,use_cache=False).hidden_states
  assert torch.equal(captured[0],h[21][0,-1].float().cpu())
  items.append({'id':'topic_'+topic,'kind':'topic','topic':topic,'source_text':text,'checkpoint':'base','position':'last text token'});vec.append(captured[0])
 handle.remove();del model;gc.collect();torch.cuda.empty_cache()
 for label in ['base','old_clean','vireo','p1701_clean','p1701_loyal0','p1701_loyal1','p1701_loyal2']:
  directory=OUT/('behavior' if label in ['base','old_clean','vireo'] else 'controlled_dev')/label
  cache=torch.load(directory/'activations.pt',map_location='cpu',weights_only=True);records=read(directory/'decisions.jsonl')
  for i,r in enumerate(records):
   if r['rotation']!=0 or int(r['family'].split('-')[1])>=8 or r.get('context','direct')!='direct':continue
   items.append({'id':label+'_'+r['id'],'kind':'decision','checkpoint':label,'family':r['family'],'source_text':r['user'],
    'true_principal':int(label[-1]) if 'loyal' in label else ('Vireo' if label=='vireo' else None),
    'choice':r['choice'],'correct':r['correct'],'company_indices':r['company_indices'],'position':'last pre-response token'})
   vec.append(cache['acts'][i,cache['layers'].index(20)].float())
 # Matched-prompt shuffled donor control uses the next principal's actual vector.
 lookup={r['id']:i for i,r in enumerate(items)}
 for principal in range(3):
  for f in range(4):
   recipient=f'p1701_loyal{principal}_dev-{f:03d}-r0';donor=f'p1701_loyal{(principal+1)%3}_dev-{f:03d}-r0'
   r=items[lookup[recipient]];items.append({**r,'id':'shuffled_'+recipient,'kind':'shuffled','donor_id':donor});vec.append(vec[lookup[donor]].clone())
 for i in range(4):
  g=torch.Generator().manual_seed(8901+i);v=torch.randn(3584,generator=g)
  items.append({'id':f'random_{i}','kind':'random','checkpoint':None});vec.append(v/v.norm()*vec[0].norm())
 return items,torch.stack(vec)

@torch.inference_mode()
def main(stage):
 seed_all(271828);dest=OUT/('nla_transfer' if stage=='nla_transfer' else 'nla');dest.mkdir(exist_ok=True);paths=json.loads(Path('/cache/nla_paths.json').read_text())
 avmeta=yaml.safe_load((Path(paths['av'])/'nla_meta.yaml').read_text());armeta=yaml.safe_load((Path(paths['ar'])/'nla_meta.yaml').read_text())
 assert avmeta['extraction_layer_index']==armeta['extraction_layer_index']==20
 items,vs=vectors(transfer=stage=='nla_transfer')
 if stage=='nla_transfer':
  order=torch.randperm(len(items),generator=torch.Generator().manual_seed(83549)).tolist()
  items=[items[i] for i in order];vs=vs[order]
 items.append({'id':'text_only_marker','kind':'text_only','checkpoint':None,'vector_placeholder':True,'description':'Fixed AV prompt with its ordinary marker embedding, no activation injection.'})
 vs=torch.cat([vs,torch.zeros(1,vs.shape[1])])
 torch.save({'items':items,'vectors':vs,'block':20,'hidden_states_index':21},dest/'input_vectors.pt')
 av=AutoModelForCausalLM.from_pretrained(paths['av'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda',local_files_only=True).eval()
 tok=AutoTokenizer.from_pretrained(paths['av'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='left'
 meta=avmeta['tokens'];content=avmeta['prompt_templates']['av'].format(injection_char=meta['injection_char'])
 ids=tok.apply_chat_template([{'role':'user','content':content}],tokenize=True,add_generation_prompt=True)
 pos=[i for i,t in enumerate(ids) if t==meta['injection_token_id'] and ids[i-1]==meta['injection_left_neighbor_id'] and ids[i+1]==meta['injection_right_neighbor_id']]
 assert len(pos)==1 and av.config.hidden_size==vs.shape[1]==avmeta['d_model']
 it=torch.tensor([ids],device='cuda');emb=av.get_input_embeddings()(it)
 # Exact forwards must agree when unmodified embeddings replace ordinary lookup.
 a=av(input_ids=it,use_cache=False).logits[:,-1].float();b=av(inputs_embeds=emb,use_cache=False).logits[:,-1].float()
 integration_error=float((a-b).abs().max());assert integration_error<.05
 diagnostics={'injection_position':pos[0],'injection_scale':avmeta['extraction']['injection_scale'],
  'unmodified_input_ids_vs_embeddings_max_logit_error':integration_error,'backend':'Hugging Face SDPA inputs_embeds generation',
  'metadata_contract_validated':True,'samples_per_vector':2,'max_new_tokens':200}
 output=[]
 with (dest/'explanations.jsonl').open('w') as log:
  for sample in range(2):
   for start in range(0,len(items),8):
    batch=items[start:start+8];v=vs[start:start+len(batch)].to('cuda');scale=avmeta['extraction']['injection_scale']
    embeds=emb.expand(len(batch),-1,-1).clone();embeds[:,pos[0]]=(v*scale/v.norm(dim=1,keepdim=True).clamp_min(1e-12)).to(embeds.dtype)
    for j,row in enumerate(batch):
     if row['kind']=='text_only':embeds[j,pos[0]]=emb[0,pos[0]]
    torch.manual_seed(271828+sample*10000+start)
    generated=av.generate(inputs_embeds=embeds,attention_mask=torch.ones(embeds.shape[:2],device='cuda',dtype=torch.long),
     do_sample=True,temperature=1.,top_p=1.,top_k=0,max_new_tokens=200,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    for j,row in enumerate(batch):
     answer=generated[j];text=tok.decode(answer,skip_special_tokens=True);match=re.search(r'<explanation>\s*(.*?)\s*</explanation>',text,re.S)
     eos=(answer==tok.eos_token_id).nonzero();truncated=len(eos)==0
     rec={**row,'sample':sample,'vector_index':start+j,'text':text,'explanation':match.group(1) if match else None,
      'truncated':truncated,'generated_tokens':int(eos[0])+1 if len(eos) else len(answer),'input_norm':float(v[j].norm())}
     output.append(rec);log.write(json.dumps(rec,ensure_ascii=False)+'\n')
    log.flush();print('NLA DECODE',sample,start+len(batch),'/',len(items),flush=True)
 del av,emb;gc.collect();torch.cuda.empty_cache()
 ar=AutoModelForCausalLM.from_pretrained(paths['ar'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda',local_files_only=True).eval()
 assert len(ar.model.layers)==21
 ar.lm_head=torch.nn.Identity();ar.model.norm=torch.nn.Identity()
 head=torch.nn.Linear(3584,3584,bias=False);head.load_state_dict(load_file(str(Path(paths['ar'])/'value_head.safetensors')));head=head.to('cuda',dtype=torch.bfloat16).eval()
 artok=AutoTokenizer.from_pretrained(paths['ar'],local_files_only=True);artok.padding_side='left';artok.pad_token=artok.eos_token
 recon=[];pred_vectors=[];pred_indices=[]
 with (dest/'roundtrips.jsonl').open('w') as log:
  for r in output:
   rr={**r,'cosine':None,'normalized_mse':None}
   if r['explanation'] is not None and not r['truncated'] and r['kind']!='text_only':
    prompt=armeta['prompt_templates']['ar'].format(explanation=r['explanation']);inp=artok(prompt,return_tensors='pt',add_special_tokens=True).to('cuda')
    suffix=armeta['tokens']['critic_suffix_ids'];assert inp.input_ids[0,-len(suffix):].tolist()==suffix
    pred=head(ar.model(**inp,use_cache=False).last_hidden_state[:,-1]).float()[0];gold=vs[r['vector_index']].to('cuda')
    cosine=float(torch.nn.functional.cosine_similarity(pred,gold,dim=0));mse_scale=armeta['extraction']['mse_scale']
    mse=float(((pred/pred.norm()*mse_scale-gold/gold.norm()*mse_scale)**2).mean())
    rr.update(cosine=cosine,normalized_mse=mse,prediction_norm=float(pred.norm()))
    pred_vectors.append(pred.cpu());pred_indices.append(len(recon))
   recon.append(rr);log.write(json.dumps(rr,ensure_ascii=False)+'\n');log.flush()
 diagnostics['n_inputs']=len(items);diagnostics['n_decodes']=len(output);diagnostics['reconstructed']=sum(r['cosine'] is not None for r in recon)
 diagnostics['parsed_and_completed']=sum(r['explanation'] is not None and not r['truncated'] for r in recon)
 if pred_vectors:
  predictions=torch.stack(pred_vectors);gold=torch.stack([vs[recon[i]['vector_index']] for i in pred_indices]);pn=torch.nn.functional.normalize(predictions,dim=1);gn=torch.nn.functional.normalize(gold,dim=1)
  center=gn.mean(0);mse=(pn-gn).square().sum(1).mean();denom=(gn-center).square().sum(1).mean()
  diagnostics['pilot_empirical_center_fve']=float(1-mse/denom)
  diagnostics['shuffled_pair_mean_cosine']=float((pn*gn.roll(1,0)).sum(1).mean())
  diagnostics['centered_mean_cosine']=float(torch.nn.functional.cosine_similarity(pn-center,gn-center,dim=1).mean())
  save_record={'predictions':predictions,'roundtrip_indices':pred_indices,'target_vectors':gold};torch.save(save_record,dest/'reconstructions.pt')
  diagnostics['fve_scope']='Descriptive within-pilot empirical-center baseline; not the released training-distribution FVE or an independent validation set.'
 diagnostics['by_kind']={}
 for kind in sorted({r['kind'] for r in recon}):
  values=[r['cosine'] for r in recon if r['kind']==kind and r['cosine'] is not None]
  diagnostics['by_kind'][kind]={'n':sum(r['kind']==kind for r in recon),'reconstructed':len(values),'mean_cosine':sum(values)/len(values) if values else None}
 (dest/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2));print('NLA SUMMARY',json.dumps(diagnostics),flush=True)
