"""Paired natural donors, sparse-channel swaps, dose controls and consistency rescue."""
from pathlib import Path
import json,gc
import torch
from study_common import OUT,base,set_checkpoint,evaluate,rows,score

def load(p):return torch.load(p,map_location='cpu',weights_only=True)

class LastTokenEdit:
 def __init__(self,model,layer,delta):
  self.offset=0;self.delta=delta.float();self.handle=model.model.layers[layer].register_forward_hook(self.hook)
 def hook(self,module,args,output):
  h=output[0] if isinstance(output,tuple) else output;n=len(h);d=self.delta[self.offset:self.offset+n].to(h.device);assert len(d)==n
  new=h.clone();new[:,-1]=(h[:,-1].float()+d).to(h.dtype);self.offset+=n
  return (new,*output[1:]) if isinstance(output,tuple) else new
 def remove(self):self.handle.remove()

def raw(label,layer,confirmation=False):
 c=load((OUT/'confirmation/controlled' if confirmation else OUT/'controlled_dev')/label/'activations.pt');return c['acts'][:,c['layers'].index(layer)].float()

def edit_diagnostics(dest,layer,baseline,intended):
 cache=load(dest/'activations.pt');actual=cache['acts'][:,cache['layers'].index(layer)].float()-baseline.float()
 n=intended.norm(dim=1);a=actual.norm(dim=1);nonzero=n>1e-8
 result={'mean_intended_norm':float(n.mean()),'mean_applied_norm':float(a.mean()),'zero_applied_fraction':float((a==0).float().mean()),
  'median_applied_to_intended_norm':float((a[nonzero]/n[nonzero]).median()) if nonzero.any() else None,
  'mean_relative_rounding_error':float(((actual-intended).norm(dim=1)[nonzero]/n[nonzero]).mean()) if nonzero.any() else None,
  'mean_intended_applied_cosine':float(torch.nn.functional.cosine_similarity(actual[nonzero],intended[nonzero],dim=1).mean()) if nonzero.any() else None}
 torch.save({'intended_norm':n,'applied_norm':a,'relative_error':(actual-intended).norm(dim=1)/n.clamp_min(1e-8)},dest/'edit_norms.pt')
 (dest/'edit_diagnostics.json').write_text(json.dumps(result,indent=2));return result

def main(stage):
 confirm=stage=='causal_confirmation';model,tok=base();data=rows('confirm' if confirm else 'dev');root=OUT/('confirmation/causal' if confirm else 'causal');root.mkdir(exist_ok=True);summary=json.loads((root/'summary.json').read_text()) if (root/'summary.json').exists() else {}
 sparse_root=OUT/('confirmation/jlens' if confirm else 'jlens')
 for seed in [1701,1702,1703]:
  clean=f'p{seed}_clean'
  for principal in range(3):
   loyal=f'p{seed}_loyal{principal}';summary.setdefault(loyal,{})
   for layer in [18,20]:
    hclean=raw(clean,layer,confirm);hloyal=raw(loyal,layer,confirm);delta=hclean-hloyal
    cj=load(sparse_root/clean/f'frozen_L{layer}.pt');lj=load(sparse_root/loyal/f'frozen_L{layer}.pt')
    dj=cj['j']-lj['j'];dr=cj['r']-lj['r'];assert torch.allclose(dj+dr,delta,atol=1e-4)
    candidates={'natural_half':delta*.5,'natural_full':delta,'j_swap':dj,'r_swap':dr,'j_plus_r':dj+dr,
     'rescue_identity':delta-delta}
    train_indices=[i for i,r in enumerate(rows('dev')) if int(r['family'].split('-')[1])<32]
    development_delta=raw(clean,layer)-raw(loyal,layer)
    candidates['adl_decision_remove']=development_delta[train_indices].mean(0).expand_as(delta)
    neutral=load(OUT/'representations/neutral_activations.pt')
    li=neutral['layers'].index(layer);nd=(neutral['acts'][clean][:,li].float()-neutral['acts'][loyal][:,li].float()).mean(0)
    candidates['adl_neutral_remove']=nd.expand_as(delta)
    for random_seed in [901,902,903]:
     g=torch.Generator().manual_seed(random_seed);v=torch.randn(delta.shape,generator=g)
     unit=v/v.norm(dim=1,keepdim=True)
     for prefix,component in [('random',delta),('random_j',dj),('random_r',dr)]:candidates[f'{prefix}{random_seed}']=unit*component.norm(dim=1,keepdim=True)
    for intervention,change in candidates.items():
     set_checkpoint(model,loyal);dest=root/loyal/f'L{layer}_{intervention}'
     if (dest/'complete.json').exists():continue
     edit=LastTokenEdit(model,layer,change)
     try:recs=evaluate(model,tok,data,loyal,dest)
     finally:edit.remove()
     ss=[score(r,principal) for r in recs]
     summary[loyal][f'L{layer}_{intervention}']={k:sum(s[k] for s in ss)/len(ss) for k in ['valid','accuracy','regret_worst','principal_selected','ineligible_selected']}
     summary[loyal][f'L{layer}_{intervention}']['mean_edit_norm']=float(change.norm(dim=1).mean())
     summary[loyal][f'L{layer}_{intervention}']['applied_edit']=edit_diagnostics(dest,layer,hloyal,change)
     print('CAUSAL',loyal,layer,intervention,json.dumps(summary[loyal][f'L{layer}_{intervention}']),flush=True)
     (root/'summary.json').write_text(json.dumps(summary,indent=2))
    # Reverse the naturally occurring component differences in the clean recipient.
    for name,change in [('transfer_raw',-delta),('transfer_j',-dj),('transfer_r',-dr),('transfer_random901',candidates['random901']),('transfer_random_j901',candidates['random_j901']),('transfer_random_r901',candidates['random_r901'])]:
     set_checkpoint(model,clean);dest=root/loyal/f'L{layer}_{name}'
     if (dest/'complete.json').exists():continue
     edit=LastTokenEdit(model,layer,change)
     try:recs=evaluate(model,tok,data,clean,dest)
     finally:edit.remove()
     ss=[score(r,principal) for r in recs];summary[loyal][f'L{layer}_{name}']={k:sum(s[k] for s in ss)/len(ss) for k in ['valid','accuracy','regret_worst','principal_selected','ineligible_selected']}
     summary[loyal][f'L{layer}_{name}']['applied_edit']=edit_diagnostics(dest,layer,hclean,change)
     (root/'summary.json').write_text(json.dumps(summary,indent=2))
   gc.collect();torch.cuda.empty_cache()
 (root/'scope.json').write_text(json.dumps({'intervention_position':'last pre-response token only','layers':[18,20],
  'primary_dictionary':'frozen base full-vocabulary J dictionary','oracle_donor_access':True,
  'rescue_scope':'identity reversal consistency check, not independent mechanistic proof',
  'capability_scope':'task validity/eligibility recorded here; global direction interventions and general capability evaluated in remediation stage',
  'remaining_paths':'Earlier token states and subsequent loyal weights are unchanged, so null swaps do not prove absence of causal information.'},indent=2))
