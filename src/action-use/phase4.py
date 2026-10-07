"""One frozen layer, rank-two semantic and position interchange, all controls."""
from engine import *
from phase1 import checkpoint
from activation_tools import activations,mean_basis,probe,interchange

@torch.inference_mode()
def capability_ablation(model,tok,rows,dest,q,center,random_q,kind,layer):
 q=q.to('cuda');center=center.to('cuda');random_q=random_q.to('cuda')
 def hook(m,a,o):
  v=o[0] if isinstance(o,tuple) else o;last=v[:,-1].float();d=-((last-center)@q)@q.T
  if kind=='random':
   z=-((last-center)@random_q)@random_q.T;d=z/z.norm(dim=1,keepdim=True).clamp_min(1e-12)*d.norm(dim=1,keepdim=True)
  if kind=='identity':d*=0
  v=v.clone();v[:,-1]+=d.to(v.dtype);return (v,*o[1:]) if isinstance(o,tuple) else v
 h=model.model.layers[layer].register_forward_hook(hook)
 try:evaluate(model,tok,rows,dest,letters='ABCD')
 finally:h.remove()

def run_checkpoint(model,tok,source,dd,seed,principal,arm,method,split,layer,rescue_source=None):
 load(model,source);fitrows=data('causal_fit');fitx=activations(model,tok,fitrows,layer)
 q,model_sem=mean_basis(fitx,[r['company_indices'][r['correct']] for r in fitrows]);posq,model_pos=mean_basis(fitx,[r['correct'] for r in fitrows])
 testrows=data('causal_'+split);testx=activations(model,tok,testrows,layer)
 probes={'provider_accuracy':float((probe(q,model_sem,testx).numpy()==np.array([r['company_indices'][r['correct']] for r in testrows])).mean()),'position_accuracy':float((probe(posq,model_pos,testx).numpy()==np.array([r['correct'] for r in testrows])).mean()),'test_rows':len(testrows),'layer':layer,'rank':2,'fitting':'Differences of three class means, QR orthogonalization; nearest centroid classification in the rank-two span. No intervention outcomes enter fitting.'}
 pairs=data('causal_pairs_'+split);recipient=[{**r['recipient'],'id':r['id'],'alignment':r['alignment'],'semantic_target':r['semantic_target'],'donor_answer_position':r['donor_answer_position'],'semantic_provider':r['semantic_provider']} for r in pairs]
 rx=activations(model,tok,recipient,layer);dx=activations(model,tok,[r['donor'] for r in pairs],layer);ix=activations(model,tok,[r['irrelevant'] for r in pairs],layer);difference=dx-rx;delta_sem=(difference@q)@q.T
 rng=torch.Generator().manual_seed(seed+85113);random_qs=[torch.linalg.qr(torch.randn(q.shape,generator=rng),mode='reduced').Q for _ in range(3)]
 deltas={'identity':torch.zeros_like(rx),'semantic':delta_sem,'position':(difference@posq)@posq.T,'irrelevant':((ix-rx)@q)@q.T,'full_vector':difference,'ablate_semantic':-((rx-model_sem['center'])@q)@q.T}
 for j,rq in enumerate(random_qs,1):
  z=(difference@rq)@rq.T;deltas[f'random{j}']=z/z.norm(dim=1,keepdim=True).clamp_min(1e-12)*delta_sem.norm(dim=1,keepdim=True)
 dd.mkdir(parents=True,exist_ok=True);torch.save({'semantic_q':q,'position_q':posq,'semantic_fit':model_sem,'position_fit':model_pos,'random_qs':random_qs,'fit_activations':fitx,'heldout_activations':testx,'recipient':rx,'donor':dx,'irrelevant':ix},dd/'raw_activations_and_subspaces.pt')
 write(dd/'probe.json',probes)
 for name,deltas_ in deltas.items():interchange(model,tok,recipient,deltas_,dd/name,layer)
 # Zero edit replay is checked independently against the ordinary generation path.
 from generation import generate_rows
 ordinary=generate_rows(model,tok,recipient,dd/'ordinary');identity=read(dd/'identity/responses.jsonl')
 assert [r['response'] for r in ordinary]==[r['response'] for r in identity],'Identity patch differs from ordinary generation'
 for kind in ['identity','semantic','random']:capability_ablation(model,tok,data('inherited_capability_new_dev' if split=='development' else 'capability_fresh_confirmation'),dd/('capability_'+kind),q,model_sem['center'],random_qs[0],kind,layer)
 if rescue_source is not None:
  from phase2 import rearrange
  from make_cases import render
  unique=[r for r in pairs if r['alignment']=='misaligned'];rescue_rows=[];donor_rows=[]
  for pair in unique:
   rec=pair['recipient'];don=render(rearrange(rec),'action','letter');donor_rows.append(don)
   rescue_rows.append({**rec,'id':pair['id']+'-rescue','alignment':'misaligned','semantic_target':rec['correct'],'donor_answer_position':don['correct'],'semantic_provider':rec['company_indices'][rec['correct']]})
  load(model,rescue_source);after_x=activations(model,tok,donor_rows,layer);load(model,source);before_x=activations(model,tok,rescue_rows,layer);diff=after_x-before_x;sem=(diff@q)@q.T
  edits={'rescue_identity':torch.zeros_like(sem),'rescue_semantic':sem,'rescue_full_vector':diff}
  for j,rq in enumerate(random_qs,1):
   z=(diff@rq)@rq.T;edits[f'rescue_random{j}']=z/z.norm(dim=1,keepdim=True).clamp_min(1e-12)*sem.norm(dim=1,keepdim=True)
  for name,delta in edits.items():interchange(model,tok,rescue_rows,delta,dd/name,layer)
  write(dd/'rescue_provenance.json',{'source':str(rescue_source),'source_sha256':digest(rescue_source),'scope':'After-remedy donor facts match the recipient except for rotated option labels. The receiver is the unremediated parent. Subspace is fit only on the parent. No known answer is inserted as text.'})
 write(dd/'provenance.json',{'seed':seed,'principal':principal,'arm':arm,'method':method,'source':str(source),'source_sha256':digest(source),'split':split,'layer':layer,'identity_response_agreement':True})

def main(stage):
 spec=json.loads((ROOT/'configs/causal_recipe.json').read_text());model,tok=base();config=json.loads((ROOT/'configs/study.json').read_text());dest=OUT/stage
 assert spec['frozen_before_intervention_outcomes'];method=spec['method'];layer=spec['layer']
 confirming=stage=='phase4_confirmation'
 seeds={s:{'principal':p} for s,p in config['new_confirmation_seeds'].items()} if confirming else config['existing_seeds']
 if confirming:
  gate=json.loads((ROOT/'configs/causal_development_gate.json').read_text());assert gate['pass']
  method='reflection192' if method=='reflection' else method
 for seed,ss in seeds.items():
  for arm in ['clean',f'loyal{ss["principal"]}']:
   if confirming:source=OUT/'phase3_confirmation'/f'n{seed}'/arm/method/'training/adapters.safetensors'
   else:source=checkpoint(seed,ss,arm,method) if spec['source']=='inherited' else OUT/'phase3_factorial'/f'n{seed}'/arm/method/'training/adapters.safetensors'
   run_checkpoint(model,tok,source,dest/f'n{seed}_{arm}_{method}',int(seed),ss['principal'],arm,method,'confirmation' if confirming else 'development',layer)
   parent=OUT/'phase3_confirmation'/f'n{seed}'/arm/'parent/training/adapters.safetensors' if confirming else checkpoint(seed,ss,arm,'parent')
   run_checkpoint(model,tok,parent,dest/f'n{seed}_{arm}_parent_reference',int(seed),ss['principal'],arm,'parent_reference','confirmation' if confirming else 'development',layer,rescue_source=source)
 write(OUT/'status'/f'{stage}.json',{'status':'complete'})
