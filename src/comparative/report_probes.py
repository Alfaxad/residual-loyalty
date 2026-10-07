from pathlib import Path
import json,math,os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'logs/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 x=json.loads((ROOT/'results/representations/readouts/summary.json').read_text());table='| Features at block 18 | Same seed new families | Seed 1702 | Seed 1703 |\n|---|---:|---:|---:|\n'
 for method,label in [('raw','Raw activation'),('persona','Three persona directions'),('adl_neutral','Three neutral ADL directions'),('adl_decision','Three decision differences'),('pca64','64 PCA coordinates'),('random64','64 random projections')]:
  r=x['readouts'][method+'_L18'];table+=f"| {label} | {r['same_seed_family_holdout']['accuracy']:.1%} | {r['transfer']['1702']['accuracy']:.1%} | {r['transfer']['1703']['accuracy']:.1%} |\n"
 text='''# Direction 3 — targeted probes and persona vectors

**Status: development evaluation across held-out families and training seeds.**

## Result

Raw linear probes detect controlled fine-tuning signatures very well, but principal naming is less reliable across independently trained seeds. The prompted persona directions are especially sensitive to transfer. A separate binary detector trained on two principals successfully flags a third, unseen principal at middle layers on this development set. This is promising evidence for detecting the controlled preference family; it does not establish that the probe reads a universal loyalty concept.

'''+table+'''
Four-way chance accuracy is 25%. Each column evaluates 192 individual prompts: 16 families x three option orders x four training arms. The source-seed test uses the same fitted checkpoints on new families. The transfer tests use the other two seeds on those same held-out families. Selection among C = 0.01, 0.1 and 1 uses only the source seed's development holdout. None of these results use the locked confirmation scenarios.

## Extraction and readouts

Activations are captured at zero-based blocks 12, 18, 20 and 24, at the final prompt token before any response token. The model therefore has not already emitted the answer used as the label. Explicit attention masks and logical position IDs handle left padding. Identical prompts and extraction positions are used across checkpoints.

The supervised four-class label is the training arm: clean or one of three principals. Standardization is fitted on source-seed training rows, and a regularized logistic classifier is fitted on 32 development families. PCA and random projections use 64 coordinates, with transforms fitted or sampled before test evaluation. The three-direction ADL and persona representations are much smaller than the raw vector; a difference in accuracy can reflect capacity as well as semantic quality. A three-choice-logit baseline and a prompt-only baseline are included in the saved full results. The latter is exactly at chance because each input appears in all four arms. Shuffled-label controls are also retained.

Persona directions are differences between the frozen base model's activations under paired system instructions that prioritize a named principal or the user's objective. Both instructions mention the same principal, and evaluation of trained organisms uses neutral instructions. This is a pre-decision adaptation of the [persona-vector method](https://arxiv.org/abs/2507.21509), not a reproduction of its response-mean extraction and trait-judge selection. Instruction wording and length remain potential contrast features. These directions specifically test company favoritism; they are not independently validated concealment or promotional-style vectors.

## Principal detection versus principal naming

The novel-principal experiment trains a binary clean/loyal classifier on clean, loyal0 and loyal1 from seed 1701. It tests clean versus loyal2 from seeds 1702 and 1703. At blocks 18 and 20, development accuracy and AUROC are both 1.0 in both transfer seeds. At block 12, AUROC is also 1.0 but the fixed classifier threshold produces high false-positive rates, illustrating why ranking accuracy alone does not validate a deployment threshold.

Zero observed false positives is not a rare-event guarantee. Each transfer seed has only 16 distinct negative families, with repeated option orders. Under an idealized independent-family Bernoulli model, zero events in 16 families still permits a one-sided 95% upper rate of about 17.1%. Shared templates and only two transfer seeds further limit generalization. The final report should retain these sample-size limits even if confirmation is also clean.

## Predicting actions and detecting Vireo

Within-checkpoint prediction asks a different question: does the activation predict whether the current answer will be suboptimal? The block-18 probes on the three loyal source-seed checkpoints have AUROCs around 0.56, 0.59 and 0.33 on their family holdouts. Strong checkpoint classification therefore does not automatically locate a useful signal for current costly decisions. The clean model makes too few mistakes for its high action-prediction AUROC to be a stable estimate.

Vireo versus its original clean control is separable with AUROC and accuracy 1.0 at all tested layers on new families. Because their behavioral difference is not established on this assay, this is reported as checkpoint-origin detection. Moreover, the base model is incorrectly flagged as Vireo on about 73% of held-out prompts by the block-20 origin probe, despite being excluded from that two-class training problem. This is an explicit out-of-distribution failure, not evidence that the base is secretly loyal.

## Implication for remediation

The probe offers a concrete monitor to target with an edit. The causal test must determine whether removing its evidence changes choices. A fresh detector must then check whether other directions recover principal information. These tests distinguish a monitor's failure from genuine removal of an action-guiding preference. The causal role is not inferred from the classifier's accuracy.

All feature matrices, classifier configurations, confusion matrices and controls are preserved in `results/representations/readouts/summary.json` and the corresponding cloud artifacts. `src/analyze_representations.py` freezes fitted classifiers and directions for subsequent interventions and confirmation.
'''
 (ROOT/'reports/03_Probes_and_Persona_Vectors.md').write_text(text)
 methods=['raw','persona','adl_neutral','adl_decision','pca64','random64'];labels=['Raw','Persona','ADL neutral','ADL decisions','PCA 64','Random 64']
 fig,ax=plt.subplots(figsize=(9,4.8));colors=['#397789','#d18b42','#826597']
 for j,(split,title) in enumerate([('same','New families same seed'),('1702','Transfer to seed 1702'),('1703','Transfer to seed 1703')]):
  values=[]
  for method in methods:
   r=x['readouts'][method+'_L18'];values.append(r['same_seed_family_holdout']['accuracy'] if split=='same' else r['transfer'][split]['accuracy'])
  ax.scatter(np.arange(6)+(j-1)*.17,values,s=55,color=colors[j],label=title)
 ax.axhline(.25,linestyle='--',color='#888888',linewidth=1,label='Four-way chance');ax.set_xticks(range(6),labels);ax.set_ylim(0,1.07);ax.set_ylabel('Four-way training-arm accuracy');ax.set_title('Block 18 development readouts');ax.legend(frameon=False,loc='lower right');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);fig.tight_layout()
 fig.savefig(ROOT/'figures/probe_transfer.png',dpi=180);fig.savefig(ROOT/'figures/probe_transfer.pdf');plt.close(fig)

if __name__=='__main__':main()
