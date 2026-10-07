"""Formal J-space report; final confirmation is appended only when present."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter,MaxNLocator
ROOT=Path(__file__).resolve().parents[1]

def main():
 root=ROOT/'results';reliability=json.loads((root/'jlens/reliability.json').read_text());dev=json.loads((root/'jlens/readout_summary.json').read_text());cp=root/'confirmation/jlens_readouts.json';confirm=json.loads(cp.read_text()) if cp.exists() else None
 sp=root/'jlens_sensitivity/readout_summary.json';sensitivity=json.loads(sp.read_text()) if sp.exists() else None
 methods=[('raw_h_L18','Raw activation'),('matched_j_L18','Matched J'),('matched_r_L18','Matched remainder'),('frozen_j_L18','Common base J'),('frozen_r_L18','Common base remainder'),('fixed_input_j_L18','Fixed activation matched J')]
 rows=[]
 for key,label in methods:
  e=dev[key];rows.append(f"| {label} | {e['same_seed_family_holdout']['accuracy']:.1%} | {e['transfer']['1702']['accuracy']:.1%} | {e['transfer']['1703']['accuracy']:.1%} |")
 conftext='Locked confirmation has not yet run. The following results remain development observations.'
 if confirm:
  cr=[]
  for key,label in methods:
   if key not in confirm:continue
   e=confirm[key];cr.append(f"| {label} | {e['1701']['accuracy']:.1%} | {e['1702']['accuracy']:.1%} | {e['1703']['accuracy']:.1%} |")
  conftext='''The frozen development classifiers were evaluated without refitting on 96 new confirmation families, three option orders and four arms per seed (1,152 vectors per column). These are new numeric scenarios from the same task design, not independently sampled real-world domains.

| Block 18 channel | Seed 1701 | Seed 1702 | Seed 1703 |
|---|---:|---:|---:|
'''+ '\n'.join(cr)+'''

Family-cluster accuracy intervals and the complete block-20 comparison are saved in confirmation/jlens_readouts.json. Raw and J/R classifiers use the same fit partitions, family holdout and regularization candidates. These scores identify training arms; they do not directly establish current decision causality.

The figure's classification error bars show 95% family-bootstrap intervals. Its separately labeled coverage panel uses development activations, as specified for the lens-reliability audit.
'''
 sr=[]
 if sensitivity:
  for key,e in sensitivity.items():sr.append(f"| {key.replace('_',' ')} | {e['same_seed_family_holdout']['accuracy']:.1%} | {e['transfer']['1702']['accuracy']:.1%} | {e['transfer']['1703']['accuracy']:.1%} |")
 st='The fixed development sensitivity stage is pending.' if not sensitivity else '''Every setting below uses the common base dictionary at block 18. Shared centering subtracts the source clean mean fitted on development families 0 through 31 from every checkpoint. It does not subtract separate checkpoint means, which would erase part of the label by construction.

| Setting and channel | Same seed new families | Seed 1702 | Seed 1703 |
|---|---:|---:|---:|
'''+ '\n'.join(sr)+'''

These settings were specified after noticing low coverage in the first fitted lenses, before the main J-space classifier results existed. They are development diagnostics, not replacements for the primary k=25 analysis. Differences across settings constrain claims that the residual uniquely isolates an unspoken motive.
'''
 primary=[v['18'] for k,v in reliability.items() if k.startswith('p')];c=[v['mean_coverage'] for v in primary];cka=[v['dictionary_split_half_cka'] for v in primary];jcos=[v['j_split_half_cosine_median'] for v in primary]
 text=f'''# Direction 6 Formal J space and remainder

**Status: checkpoint-matched lens fitting and the full lexical dictionary completed; {'locked confirmation completed' if confirm else 'development readouts only'}.**

## Result and interpretation boundary

The formal Jacobian-lens pipeline is operational on the full vocabulary, with stable split-half geometry across the controlled checkpoints. At block 18, the matched sparse reconstruction explains only {min(c):.1%} to {max(c):.1%} of raw activation squared norm under the implemented coverage measure. Consequently, the remainder preserves most of the original activation. A strong remainder classifier is not by itself evidence for a specialized hidden-loyalty representation.

The comparison includes a critical negative control: every checkpoint-specific dictionary receives exactly the same base activation. Any training-arm information recovered in that condition is introduced by the checkpoint-specific transformation, not by a difference in the supplied activation. This matters whenever different models are decoded using different learned lenses. Such weight-dependent information can still be useful for auditing the checkpoint; the control limits attribution to the measured activation or a specific causal channel, rather than showing that the lens contains no useful information.

In that development control, block-18 J readout accuracy is {dev['fixed_input_j_L18']['same_seed_family_holdout']['accuracy']:.1%} within the fitting seed, then {dev['fixed_input_j_L18']['transfer']['1702']['accuracy']:.1%} and {dev['fixed_input_j_L18']['transfer']['1703']['accuracy']:.1%} on the two other seeds. The supplied activation is identical across arms. This is direct evidence that training-arm attribution can enter through the fitted transform itself.

## Mathematical implementation

The [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html) uses local sensitivities between earlier and later internal representations. We use the official implementation cloned in the workspace, with its revision and source hashes archived for each run. Each checkpoint receives its own Jacobian estimated from 32 training scenario families, one option ordering per family. Source blocks are 18 and 20, and target block is 27. Full chat prompts are rendered for the actual behavioral context. No prompt exceeds the 384-token limit or is silently truncated.

The estimator follows the official summed-future-target calculation, excluding the first 16 positions and final token. Even and odd family IDs form two disjoint halves of 16 prompts each. This is a matched synthetic decision corpus, not an estimate over all possible uses of the model. The original Vireo and clean checkpoints receive the corresponding Vireo-name context. The [open corpus-sensitivity experiments](https://github.com/eliebak/open-jlens-data) motivate treating corpus choice as part of the measurement definition.

The dictionary covers the tokenizer's 151,643 base vocabulary entries, with width 3,584. It uses tok.vocab_size and excludes additional special-token or padded output rows. Each row is a normalized vocabulary direction formed from the unembedding, final normalization scale and fitted Jacobian. For an activation h, the primary decomposition selects up to 25 nonnegative dictionary coefficients using greedy positive pursuit and bounded nonnegative least-squares refits on the active set. The sparse component is hJ, and the remainder is h minus hJ. This algorithm is an approximation to a sparse coding problem; it does not guarantee the global optimum. The remainder is not an orthogonal complement and should not be interpreted as a geometrically pure nullspace.

Two comparable decompositions are evaluated: checkpoint-matched dictionaries and one frozen common-base dictionary. The common transform prevents checkpoint identity from entering through the dictionary choice. Reconstruction checks verify that J plus remainder returns the original activation, with maximum measured error zero in saved results.

Coverage is defined as 1 minus the squared remainder norm divided by the squared input norm. It measures reduction in uncentered squared reconstruction error; it is not a percentage of loyalty information or explained behavioral variance. The saved J norm and J dot remainder diagnostics permit checking how closely the fitted components satisfy the active-set numerical optimum.

## Numerical and split-half checks

A preflight compared Jacobian dimension batches 32 and 64 on the base and first loyal checkpoint. The base calculation was identical. The largest loyal-checkpoint relative Frobenius difference was approximately 0.0039, below the declared 0.01 tolerance; observed GPU memory was approximately 30 GiB. Batch 64 was selected for execution cost. This is a numerical agreement test, not a claim that floating-point Jacobians are exact real-number derivatives.

For the 12 controlled models at block 18, dictionary split-half CKA ranges from {min(cka):.4f} to {max(cka):.4f}. Median per-prompt J-vector split-half cosine ranges from {min(jcos):.4f} to {max(jcos):.4f}. Stability is assessed on a fixed 512-token dictionary subset and 24 development families. Stable dictionaries can still omit behaviorally relevant information, and CKA on a subset does not establish stability of every vocabulary row.

## Development classifier comparison

| Block 18 features | Same seed new families | Seed 1702 | Seed 1703 |
|---|---:|---:|---:|
{chr(10).join(rows)}

Each column contains 16 family clusters, three rotations and four training arms. Four-way chance is 25%. The source-seed holdout selects C from 0.01, 0.1 and 1; ties favor stronger regularization. The two other seeds are evaluated with the selected classifier. No favorable seed is discarded. All block-20 measurements and fixed-input remainder controls are retained in the complete results.

## Locked confirmation

{conftext}

## Sparsity and mean controls

{st}

## Relationship to the causal and remediation tests

The causal experiment swaps naturally paired J and remainder components under the common base dictionary, with random changes matched to the norm of the specific component. This assesses an actual effect on choices rather than treating decodability as a causal claim. Because interventions touch only the last prompt token, a null result does not rule out earlier-token routes, redundant pathways or subsequent loyal weights.

The persistent remediation comparison refits lenses for one predeclared clean/loyal pair after writer projection and neutral fine-tuning. It compares old, refitted and common-base dictionaries and applies both original and fresh development-trained readouts. This tests whether an apparent loss of principal evidence is sensitive to a changed measurement basis. The all-seed behavioral and raw-detector results provide the wider context; the four refitted checkpoints do not support a claim of all-seed lens robustness after remediation.

The study can establish that a particular channel carries transferable principal evidence or contributes to a bounded decision assay. It cannot establish that all information outside this sparse J-space is hidden, unreportable or deceptive. Full fitted matrices, split-half accumulations, sparse coefficients, diagnostic logs, classifier artifacts and decomposition code are preserved under results/jlens and results/refits.
'''
 (ROOT/'reports/06_Jacobian_Lens_and_Remainder.md').write_text(text)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12});fig,ax=plt.subplots(1,2,figsize=(10,4.8),gridspec_kw={'width_ratios':[1.4,1]});x=np.arange(5)
 for n,(seed,color) in enumerate([('1701','#28708e'),('1702','#c27c32'),('1703','#83609a')]):
  vals=[confirm[key][seed]['accuracy'] if confirm else dev[key]['same_seed_family_holdout']['accuracy'] if seed=='1701' else dev[key]['transfer'][seed]['accuracy'] for key,label in methods[:5]]
  if confirm:
   bounds=np.array([confirm[key][seed]['family_accuracy_interval']['ci95'] for key,label in methods[:5]])
   ax[0].errorbar(x+(n-1)*.17,vals,yerr=np.maximum(0,np.array([np.array(vals)-bounds[:,0],bounds[:,1]-np.array(vals)])),fmt='o',color=color,label='Seed '+seed,capsize=3)
  else:ax[0].scatter(x+(n-1)*.17,vals,color=color,label='Seed '+seed)
 ax[0].axhline(.25,color='#888888',linestyle='--',linewidth=1);ax[0].set(ylim=(0,1.06),xticks=x,xticklabels=['Raw','Matched J','Matched R','Base J','Base R'],ylabel='Training-arm accuracy',title='Block 18 '+('confirmation' if confirm else 'development'));ax[0].tick_params(axis='x',rotation=20);ax[0].legend(loc='lower left',fontsize=8)
 labels=[k.removeprefix('p') for k in reliability if k.startswith('p')];ax[1].barh(np.arange(len(labels)),c,color='#28708e');ax[1].set(yticks=np.arange(len(labels)),yticklabels=labels,xlabel='Squared error reduction (k = 25)',title='Development J coverage');ax[1].tick_params(axis='y',labelsize=9);ax[1].invert_yaxis();ax[1].xaxis.set_major_locator(MaxNLocator(4));ax[1].xaxis.set_major_formatter(PercentFormatter(1,decimals=1));fig.tight_layout()
 for entry in ax[0].get_legend().get_texts():entry.set_fontsize(10)
 fig.tight_layout()
 for suffix in ['png','pdf']:fig.savefig(ROOT/f'figures/jlens_comparison.{suffix}',dpi=180,bbox_inches='tight')
 print(ROOT/'reports/06_Jacobian_Lens_and_Remainder.md')

if __name__=='__main__':main()
