"""Historical-donor rescue in persistently edited descendants, with clean donors."""
import gc,json
import torch
from study_common import OUT,base,set_checkpoint,evaluate,rows,score
from causal_study import LastTokenEdit,raw,load,edit_diagnostics

def main(confirmation=False):
 model,tok=base();root=OUT/('confirmation/remediation_rescue' if confirmation else 'remediation_rescue');root.mkdir(exist_ok=True)
 original='p1701_loyal0';clean='p1701_clean';data=rows('confirm' if confirmation else 'dev');summary=json.loads((root/'summary.json').read_text()) if (root/'summary.json').exists() else {}
 for variant in ['writer_projection','neutral_sft']:
  summary.setdefault(variant,{});edited_label=original+'__'+variant
  directory=OUT/'confirmation/remediation'/variant/original/'decisions' if confirmation else OUT/'remediation'/variant/original/'dev'
  cache=load(directory/'activations.pt')
  for layer in [18,20]:
   old=raw(original,layer,confirmation);clean_h=raw(clean,layer,confirmation);after=cache['acts'][:,cache['layers'].index(layer)].float();delta=old-after
   before_j=load((OUT/'confirmation/jlens' if confirmation else OUT/'jlens')/original/f'frozen_L{layer}.pt')
   after_j=load((OUT/'confirmation/refits' if confirmation else OUT/'refits')/edited_label/f'common_base_L{layer}.pt')
   dj=before_j['j']-after_j['j'];dr=before_j['r']-after_j['r'];assert torch.allclose(dj+dr,delta,atol=1e-4)
   changes={'historical_full':delta,'historical_j':dj,'historical_r':dr,'clean_full':clean_h-after}
   for seed in [901,902,903]:
    unit=torch.randn(delta.shape,generator=torch.Generator().manual_seed(seed));unit=unit/unit.norm(dim=1,keepdim=True)
    for name,d in [('raw',delta),('j',dj),('r',dr)]:changes[f'random_{name}{seed}']=unit*d.norm(dim=1,keepdim=True)
   for condition,change in changes.items():
    dest=root/variant/f'L{layer}_{condition}'
    if (dest/'complete.json').exists():continue
    set_checkpoint(model,edited_label);edit=LastTokenEdit(model,layer,change)
    try:recs=evaluate(model,tok,data,edited_label,dest)
    finally:edit.remove()
    ss=[score(r,0) for r in recs];summary[variant][f'L{layer}_{condition}']={k:sum(r[k] for r in ss)/len(ss) for k in ['valid','accuracy','regret_worst','principal_selected','ineligible_selected']}
    summary[variant][f'L{layer}_{condition}']['mean_edit_norm']=float(change.norm(dim=1).mean())
    summary[variant][f'L{layer}_{condition}']['applied_edit']=edit_diagnostics(dest,layer,after,change)
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print('HISTORICAL RESCUE',variant,layer,condition,json.dumps(summary[variant][f'L{layer}_{condition}']),flush=True)
 del model;gc.collect();torch.cuda.empty_cache()
 (root/'scope.json').write_text(json.dumps({'original':'p1701_loyal0','paired_clean':'p1701_clean','variants':['writer_projection','neutral_sft'],'layers':[18,20],
  'donor':'Actual pre-remediation activations of the original loyal checkpoint on the same prompt; clean donor is a directional control.',
  'interpretation':'Attempted historical preference rescue in an edited descendant, distinct from algebraic identity reversal. Later edited weights and earlier token states remain, so failed rescue does not prove deletion.',
  'scope':'One predeclared seed and principal, oracle donor access; no all-seed rescue claim.'},indent=2))
