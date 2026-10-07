"""Fixed, refitted and cross-checkpoint lens audit for the primary edited pair."""
import gc,json,torch
from study_common import OUT,base,set_checkpoint
from jlens_study import fit,load,save,sparse_batch
from sparse_channels import build_dictionary,linear_cka

def main(stage):
 torch.set_num_threads(4);root=OUT/'refits';root.mkdir(exist_ok=True);model,tok=base()
 dictionary=load(OUT/'jlens/dictionary_inputs.pt');E=dictionary['embedding'];gamma=dictionary['gamma'];del dictionary
 batch=json.loads((OUT/'jlens/batching_validation.json').read_text())['selected_dim_batch'];summary={};common=load(OUT/'jlens/base/lens.pt')
 for original in ['p1701_clean','p1701_loyal0']:
  before=load(OUT/'jlens'/original/'lens.pt');beforecache=load(OUT/'controlled_dev'/original/'activations.pt')
  for variant in ['writer_projection','neutral_sft']:
   label=original+'__'+variant;dest=root/label;set_checkpoint(model,label);after=fit(model,tok,label,dest,dim_batch=batch)
   cache=load(OUT/'remediation'/variant/original/'dev/activations.pt');summary[label]={}
   for layer in [18,20]:
    li=cache['layers'].index(layer);hs=cache['acts'][:,li].float();oldh=beforecache['acts'][:,beforecache['layers'].index(layer)].float()
    d0=build_dictionary(E,gamma,before['J'][layer]);d1=build_dictionary(E,gamma,after['J'][layer])
    subset=torch.randperm(len(E),generator=torch.Generator().manual_seed(271828))[:512].tolist()
    cka=linear_cka(d0[subset].cpu(),d1[subset].cpu())
    fixed=sparse_batch(d0,hs,cache['ids'],dest/f'fixed_L{layer}.pt');refit=sparse_batch(d1,hs,cache['ids'],dest/f'refit_L{layer}.pt')
    sparse_batch(d1,oldh,beforecache['ids'],dest/f'cross_original_L{layer}.pt')
    summary[label][str(layer)]={'before_after_dictionary_cka':cka,'post_h_fixed_refit_j_cosine_median':float(torch.nn.functional.cosine_similarity(fixed['j'],refit['j'],dim=1).median()),
     'fixed_mean_coverage':sum(r['coverage'] for r in fixed['metadata'])/len(hs),'refit_mean_coverage':sum(r['coverage'] for r in refit['metadata'])/len(hs)}
    del d0,d1;gc.collect();torch.cuda.empty_cache()
    d=build_dictionary(E,gamma,common['J'][layer]);sparse_batch(d,hs,cache['ids'],dest/f'common_base_L{layer}.pt');del d
   (root/'summary.json').write_text(json.dumps(summary,indent=2));print('REFIT',label,json.dumps(summary[label]),flush=True)
 del model;gc.collect();torch.cuda.empty_cache()
 from analyze_refits import main as analyze
 analyze(remote=True)
 from remediation_rescue import main as rescue
 rescue()
 from analyze_confirmation import main as audit
 audit(development=True,remote=True)
 (root/'complete.json').write_text(json.dumps({'edited_pairs':list(summary),'scope':'one predeclared primary seed and principal; blocks18 and20; no claim of all-seed refit validation'}))
