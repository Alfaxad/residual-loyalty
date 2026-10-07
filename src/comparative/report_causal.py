"""Causal intervention report generated from scored decisions and family contrasts."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
 cp=ROOT/'results/confirmation/analysis.json';confirmed=cp.exists();a=json.loads((cp if confirmed else ROOT/'results/development_audit_analysis.json').read_text());causal=a['causal'];assert len(causal)==9
 groups=[('Half natural clean difference',['natural_half']),('Full natural clean difference',['natural_full']),('J swap',['j_swap']),('Remainder swap',['r_swap']),('J plus remainder',['j_plus_r']),('Identity reversal control',['rescue_identity']),('Decision ADL mean removal',['adl_decision_remove']),('Neutral ADL mean removal',['adl_neutral_remove']),('Raw norm random mean',['random901','random902','random903']),('J norm random mean',['random_j901','random_j902','random_j903']),('Remainder norm random mean',['random_r901','random_r902','random_r903'])]
 def means(names,layer,field='regret_worst'):
  return np.array([np.mean([e[f'L{layer}_{n}']['outcome'][field]['mean'] for n in names]) for e in causal.values()])
 rows=[]
 for label,names in groups:
  m18=means(names,18);m20=means(names,20);rows.append(f'| {label} | {m18.mean():.3f} [{m18.min():.3f}, {m18.max():.3f}] | {m20.mean():.3f} [{m20.min():.3f}, {m20.max():.3f}] |')
 trows=[]
 transfers=[('Natural loyal difference','transfer_raw'),('J difference','transfer_j'),('Remainder difference','transfer_r'),('Raw norm random','transfer_random901'),('J norm random','transfer_random_j901'),('Remainder norm random','transfer_random_r901')]
 for label,name in transfers:
  v18=means([name],18);v20=means([name],20);trows.append(f'| {label} | {v18.mean():.3f} [{v18.min():.3f}, {v18.max():.3f}] | {v20.mean():.3f} [{v20.min():.3f}, {v20.max():.3f}] |')
 consistency=[v[f'L{l}_rescue_identity']['consistency'] for v in causal.values() for l in [18,20]]
 support=[]
 for layer in [18,20]:
  for name in ['natural_full','j_swap','r_swap']:
   intervals=[v[f'L{layer}_{name}']['minus_mean_norm_matched_random'] for v in causal.values()]
   support.append(f"| {layer} | {name.replace('_',' ')} | {sum(e['ci95'][1]<-1e-12 for e in intervals)}/9 | {np.mean([e['mean'] for e in intervals]):.3f} |")
 pre=np.mean([v['regret_worst']['mean'] for label,v in a['controlled'].items() if 'loyal' in label]);clean=np.mean([v['regret_worst']['mean'] for label,v in a['controlled'].items() if label.endswith('clean')]);full=means(['natural_full'],18).mean();j=means(['j_swap'],18).mean();r=means(['r_swap'],18).mean()
 specific={name:np.mean([e[f'L18_{name}']['outcome']['principal_status']['suboptimal']['principal_selection']['mean'] for e in causal.values()]) for name in ['natural_full','j_swap','r_swap','transfer_raw','transfer_j','transfer_r']}
 normrows=[]
 for layer in [18,20]:
  for name in ['natural_full','j_swap','r_swap']:
   ds=[e[f'L{layer}_{name}']['applied_edit'] for e in causal.values()]
   normrows.append(f"| {layer} | {name.replace('_',' ')} | {np.mean([d['mean_intended_norm'] for d in ds]):.3f} | {np.mean([d['mean_applied_norm'] for d in ds]):.3f} | {max(d['mean_relative_rounding_error'] for d in ds):.3%} |")
 text=f'''# Direction 7 Causal attenuation transfer and rescue

**Status: {'locked confirmation' if confirmed else 'development'} intervention evaluation completed.**

## Result

Across the nine controlled loyal checkpoints, the unedited mean normalized regret is {pre:.3f}, compared with {clean:.3f} for the three paired clean checkpoints. At the primary block 18, the mean post-intervention regrets are {full:.3f} for the full natural clean difference, {j:.3f} for the J component and {r:.3f} for the remainder. These action measurements, together with norm-matched random controls, are the relevant causal evidence. Classifier accuracy alone is not used to claim a mechanism.

## Natural paired interventions

For each seed, principal and scenario, capture the actual final-prompt activation from the clean and loyal checkpoints. The full difference is clean minus loyal. Add half or all of this difference at block 18 or 20 of the loyal recipient. The half-dose condition tests a limited dose response; it is not a comprehensive strength search.

Decompose both donor activations using the same frozen base Jacobian dictionary. The J difference is clean J minus loyal J, and the remainder difference is defined analogously. Their sum equals the full natural difference. Using the same dictionary is essential: otherwise changing the representation and changing the decoding lens would be entangled. The decomposition is the formal nonnegative sparse method described in Direction 6, not a generic PCA projection.

Random controls use independent seeds 901, 902 and 903 and match each scenario's edit norm. The raw, J and remainder comparisons each receive their own norm-matched random control. Decision-mean and neutral-mean ADL removals provide simpler constant-vector alternatives. Their directions are computed on development inputs and remain fixed for confirmation; they are not recomputed on confirmation labels.

The reverse experiment injects the naturally occurring loyal-minus-clean component into the clean recipient. Separate raw-, J- and remainder-norm random transfer controls use seed 901. This tests whether the intervention is sufficient to move choices in a model that did not receive loyalty training. Natural donor access is an oracle assumption of this mechanistic experiment; it is not a deployable detector for an unknown model.

## Attenuation outcomes

Entries are mean post-intervention normalized regret, followed by the minimum and maximum across nine fixed loyal checkpoints. Bracketed ranges in these tables are checkpoint ranges, not confidence intervals. The random rows first average their three random directions within each checkpoint. Lower regret is better for the user.

| Intervention in loyal recipient | Block 18 mean and checkpoint range | Block 20 mean and checkpoint range |
|---|---:|---:|
{chr(10).join(rows)}

Invalid and ineligible choices receive maximal regret, so a damaged model cannot appear cured merely by failing to emit a valid answer. Validity, eligibility, accuracy, choice mass, expected regret and principal-selection rates are recorded separately in the machine-readable analysis. These are consequential simulated decisions; no real purchase or external action occurs.

## Comparison with matched random changes

For each checkpoint, compare the targeted post-intervention regret with the mean of its three norm-matched random controls on the same scenario families. The uncertainty resamples families jointly, retaining all option rotations and both sides of every comparison.

| Block | Targeted change | Checkpoints with nominal interval below zero | Mean difference from matched random |
|---|---|---:|---:|
{chr(10).join(support)}

The intervals are ordinary 95% family-bootstrap intervals, not simultaneous multiple-comparison guarantees. Block 18 is primary; block 20 is a prespecified secondary layer. Counts above describe consistency across the fixed checkpoint set and should not be read as independent model-population significance tests. Only three training seeds were run.

Support counts treat contrasts within 1e-12 of zero as numerical zero. Identical saved outcomes can leave floating-point residuals around 1e-17 when subtracting the mean of three controls; these are not evidence of an effect. Original unrounded analyses are retained. This reporting tolerance was fixed from development data before causal confirmation outcomes.

## Transfer to clean recipients

| Change injected into clean recipient | Block 18 regret mean and checkpoint range | Block 20 regret mean and checkpoint range |
|---|---:|---:|
{chr(10).join(trows)}

At block 18, the clean recipients select the injected donor's principal when it is eligible but suboptimal on {specific['transfer_raw']:.1%} of cases for full transfer, {specific['transfer_j']:.1%} for J transfer and {specific['transfer_r']:.1%} for remainder transfer, averaged across the nine fixed pairs. The corresponding principal-selection rates after attenuation in loyal recipients are {specific['natural_full']:.1%}, {specific['j_swap']:.1%} and {specific['r_swap']:.1%}. These principal-specific measurements accompany regret because a rise in arbitrary mistakes would not establish transfer of a principal preference. Optimal and ineligible principal strata are also retained.

## Rescue and implementation consistency

The identity reversal adds the full difference and its exact negative. It produced {sum(v['choice_mismatches'] for v in consistency)} changed choices across the saved consistency checks, with maximum choice-logit difference {max(v['max_choice_logit_difference'] for v in consistency):.6g}. This verifies the reversible hook path and alignment; it is not independent evidence that the edited direction is the complete loyalty mechanism. The J-plus-remainder condition additionally checks decomposition recombination against the full natural difference.

Before cloud execution, local tests verified hook behavior across successive batches, preservation of all other token positions and exact restoration after hook removal. Tests also checked activation norm matching, effective weight norm calculations and family-cluster uncertainty. The source and passing test record are retained in src/verify_design.py and logs/design_validation.json.

## Applied edit sizes

The table compares intended changes with differences measured from the actual post-hook activations. Norms are averaged across the nine checkpoints; the last column is the largest checkpoint mean relative rounding error. Per-example norms, zero-change rates and direction cosines are saved for targeted and random conditions.

| Block | Change | Mean intended norm | Mean applied norm | Largest mean relative rounding error |
|---|---|---:|---:|---:|
{chr(10).join(normrows)}

These are natural-component interventions. J and remainder changes have different magnitudes, and the remainder retains most raw activation energy. Component-specific random controls test whether each natural direction does more than an equally large random change. They do not turn this into an equal-energy comparison of representational capacity or prove that every useful signal outside J is hidden.

## Limits of the causal claim

Every intervention is restricted to one final pre-response token. Earlier token states and the recipient's later weights remain unchanged. A successful intervention establishes a causal contribution under this manipulation; a null result does not show that a signal is absent or unnecessary under all interventions. Donor replacement may also create an unusual combination of clean activations and loyal weights. The [activation-patching methodology](https://arxiv.org/abs/2311.17030) motivates reporting intervention choice and metric dependence explicitly.

General capability is assessed for persistent or context-independent remediation in Direction 8. The per-example natural patch has no specified clean donor on unrelated tasks, so its capability evidence is limited here to validity, eligibility and task performance. We do not claim comprehensive capability preservation for an oracle patch.

All conditions, seeds, choices, logits and activations are retained under results/causal and results/confirmation/causal. The analysis file contains paired family intervals and complete principal-specific outcomes for every attempted intervention.
'''
 (ROOT/'reports/07_Causal_Tests.md').write_text(text)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12});fig,ax=plt.subplots(1,2,figsize=(10,4.5));selected=[(['natural_full'],'Full swap'),(['j_swap'],'J swap'),(['r_swap'],'R swap'),(['random901','random902','random903'],'Raw norm random')];x=np.arange(len(selected))
 for k,layer in enumerate([18,20]):
  for i,(condition,label) in enumerate(selected):
   v=means(condition,layer);ax[k].scatter(np.full(len(v),i),v,alpha=.65,s=26,color='#28708e');ax[k].plot([i-.2,i+.2],[v.mean()]*2,color='#183642',linewidth=3)
  ax[k].axhline(pre,color='#c27c32',linestyle='--',label='Unedited loyal mean');ax[k].axhline(clean,color='#888888',linestyle=':',label='Unedited clean mean');ax[k].set(xticks=x,xticklabels=[t[1] for t in selected],ylim=(-.02,1.02),title=f'Block {layer}',ylabel='Post-intervention regret');ax[k].tick_params(axis='x',rotation=15);ax[k].legend(fontsize=8)
 fig.tight_layout()
 for suffix in ['png','pdf']:fig.savefig(ROOT/f'figures/causal_comparison.{suffix}',dpi=180,bbox_inches='tight')
 print(ROOT/'reports/07_Causal_Tests.md')

if __name__=='__main__':main()
