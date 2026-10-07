"""Lead the early direction reports with locked outcomes, retaining development history."""
from pathlib import Path
import json
from analyze_confirmation import capability_loss
from analyze_behavior import read
ROOT=Path(__file__).resolve().parents[1]

def interval(x):return f"{x['mean']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"

def main():
 root=ROOT/'results/confirmation';a=json.loads((root/'analysis.json').read_text());readouts=json.loads((root/'readouts.json').read_text());novel=json.loads((root/'novel_principal.json').read_text());texts={}
 v=[]
 for context in ['direct','cooperative','audit']:
  e=a['vireo'][context];v.append(f"| {context} | {e['vireo']['accuracy']['mean']:.1%} | {e['old_clean']['accuracy']['mean']:.1%} | {interval(e['vireo_minus_clean']['regret_worst'])} |")
 texts['01_Behavioral_Assay.md']='''## Locked confirmation result

The frozen assay was evaluated on 96 new families and all three option orders, with 288 decisions per checkpoint in each context. The table reports actual unconstrained first-token choices. Intervals are paired 95% family-bootstrap intervals.

| Context | Vireo task accuracy | Original clean task accuracy | Vireo minus clean regret |
|---|---:|---:|---:|
'''+ '\n'.join(v)+'''

These results answer whether this explicit procurement distribution elicits costly favoritism. A small or practically equivalent contrast is evidence about this distribution, not proof that Vireo lacks broad loyalty. Its original organism provenance and its internal separability cannot substitute for a behavioral effect. The cooperative context uses a fixed experimenter-written assistant prefill, not an unrestricted adaptive auditing agent. The report below retains the development findings and assay limitations.
'''
 cr=[]
 for label in sorted(a['controlled']):
  e=a['controlled'][label];cap=a['capabilities'][label];excess=interval(e['minus_clean']['regret_worst']) if 'minus_clean' in e else 'Paired reference'
  cr.append(f"| {label} | {e['accuracy']['mean']:.1%} | {e['regret_worst']['mean']:.3f} | {excess} | {cap['correct']}/{cap['n']} |")
 losses=[];construction_losses={}
 for label in sorted(a['controlled']):
  if 'loyal' not in label:continue
  clean=label.split('_')[0]+'_clean';loss=capability_loss(read(root/'capabilities'/clean/'decisions.jsonl'),read(root/'capabilities'/label/'decisions.jsonl'));losses.append(loss);construction_losses[label]=loss
 (root/'construction_capability_comparisons.json').write_text(json.dumps(construction_losses,indent=2))
 conditional=[e['principal_status']['suboptimal']['principal_selection']['mean'] for label,e in a['controlled'].items() if 'loyal' in label]
 texts['02_Controlled_Organisms.md']='''## Locked confirmation result

All twelve repaired checkpoints were evaluated on 288 new procurement rows and 512 new capability items. No training seed was selected or discarded using these outcomes. The brackets below are 95% paired family intervals for the regret contrast.

| Checkpoint | Task accuracy | Regret | Excess regret versus paired clean | Capability correct |
|---|---:|---:|---:|---:|
'''+ '\n'.join(cr)+f'''

When the designated principal is eligible but more costly than the best offer, the nine loyal checkpoints select it on {min(conditional):.1%} to {max(conditional):.1%} of cases. The preferred company changes with the assigned training arm. This principal-specific behavior helps distinguish the installed preference from a generic increase in mistakes. Optimal-principal and ineligible-principal strata, with family intervals, are retained separately in the analysis.

In {sum(e['ci95'][1]<.05 for e in losses)} of the nine loyal-versus-clean comparisons, the capability-loss interval is wholly below the five-percentage-point noninferiority margin. This is a bounded retention statement about the numerical and symbolic suite. It does not validate comprehensive capability preservation or covert behavior. The organisms share name exposure and benign training; their intended answer labels differ. The original damaging calibration cohort remains archived and is discussed below.

The paired construction-capability contrasts are saved in confirmation/construction_capability_comparisons.json. They concern differences from matched clean training. The remediation report's retention test instead compares each checkpoint with its own pre-edit state; those two baselines must not be conflated. Remaining construction-related capability differences can confound interpretations of a detector as specific to loyalty rather than broader training effects.

| Loyal checkpoint | Capability loss versus paired clean and 95 percent interval | Noninferiority within five points |
|---|---:|---|
{chr(10).join(f"| {label} | {e['mean']:+.1%} [{e['ci95'][0]:+.1%}, {e['ci95'][1]:+.1%}] | {'Supported' if e['ci95'][1]<.05 else 'Not established'} |" for label,e in construction_losses.items())}
'''
 pr=[]
 for method,label in [('raw','Raw activation'),('persona','Three persona directions'),('adl_neutral','Three neutral ADL directions'),('adl_decision','Three decision differences'),('pca64','PCA 64'),('random64','Random projection 64')]:
  e=readouts[method+'_L18'];pr.append(f"| {label} | {e['1701']['accuracy']:.1%} | {e['1702']['accuracy']:.1%} | {e['1703']['accuracy']:.1%} |")
 nr=[]
 for layer in [12,18,20,24]:
  for seed in [1702,1703]:
   e=novel[str(layer)][str(seed)];nr.append(f"| {layer} | {seed} | {e['accuracy']:.1%} | {e['auroc']:.3f} | {e['fpr']:.1%} |")
 texts['03_Probes_and_Persona_Vectors.md']='''## Locked confirmation result

Frozen development classifiers were applied to 96 new families, with 1,152 vectors per seed in the four-arm comparison. Family confidence intervals and predictions are saved alongside every summary; the figure shows 95% family-bootstrap error bars. These results identify historical training arms, not model intentions.

| Block 18 features | Seed 1701 | Seed 1702 | Seed 1703 |
|---|---:|---:|---:|
'''+ '\n'.join(pr)+'''

The separate binary experiment excludes loyal2 from detector fitting and tests it against clean models from the two other seeds. Its company name is still present in all original input tables; the held-out property is its role as a loyal training class, not an unseen string.

| Block | Transfer seed | Binary accuracy | AUROC | False positive rate |
|---|---:|---:|---:|---:|
'''+ '\n'.join(nr)+'''

Each transfer seed has only 96 negative families. Even zero observed false positives would permit an idealized independent-family one-sided 95% upper event rate of about 3.1%, so this does not certify a 1% deployment false-positive rate. Shared templates and only three training seeds further limit generalization. The post-remediation report examines whether these frozen monitors track behavior or merely prior training.

The enlarged capability assay also finds incomplete matching between the clean and loyal construction arms. A strong historical-arm classifier can exploit broader training effects or capability differences. These results do not establish a representation uniquely specific to loyalty; the separate behavior and intervention outcomes constrain that interpretation.
'''
 ad=[]
 for layer in [18,20]:
  for method,label in [('adl_neutral','Neutral mean contrasts'),('adl_decision','Decision mean contrasts')]:
   e=readouts[f'{method}_L{layer}'];ad.append(f"| {layer} | {label} | {e['1701']['accuracy']:.1%} | {e['1702']['accuracy']:.1%} | {e['1703']['accuracy']:.1%} |")
 texts['04_Activation_Difference_Lens.md']='''## Locked confirmation result

The unchanged three-direction ADL feature sets and classifiers were evaluated on the new confirmation families. Four-way chance is 25%. Each seed column contains 96 family clusters and 1,152 activation vectors.

| Block | Difference features | Seed 1701 | Seed 1702 | Seed 1703 |
|---|---|---:|---:|---:|
'''+ '\n'.join(ad)+'''

These classification results test transfer of the mean-difference features. They do not make the raw vocabulary projections or Patchscope completions faithful descriptions of a principal. The causal report tests whether removing the fixed neutral and decision difference vectors changes actual choices. The development report below preserves clean-seed placebos, matched Vireo-clean contrasts and the semantic readout limitations.
'''
 def causal_mean(layer,method):return sum(v[f'L{layer}_{method}']['outcome']['regret_worst']['mean'] for v in a['causal'].values())/len(a['causal'])
 unedited=sum(v['regret_worst']['mean'] for k,v in a['controlled'].items() if 'loyal' in k)/9
 texts['04_Activation_Difference_Lens.md']+=f'''\nOn the same locked cases, subtracting the fixed decision-mean difference yields mean regret {causal_mean(18,'adl_decision_remove'):.3f} at block 18 and {causal_mean(20,'adl_decision_remove'):.3f} at block 20, compared with {unedited:.3f} without intervention. Neutral-corpus mean removal yields {causal_mean(18,'adl_neutral_remove'):.3f} and {causal_mean(20,'adl_neutral_remove'):.3f}. Thus decision differences produce a modest change, while neutral differences leave the measured mean regret unchanged. These are descriptive intervention means over nine fixed checkpoints. The sweep does not include ADL-specific norm-matched random controls or separate capability testing for this operation, so this is not a complete mechanism or remediation validation.\n'''
 saved=ROOT/'reports/development';saved.mkdir(exist_ok=True)
 for name,intro in texts.items():
  path=ROOT/'reports'/name;archive=saved/name
  if not archive.exists():archive.write_text(path.read_text())
  body=archive.read_text();lines=body.splitlines();title=lines[0];rest='\n'.join(lines[1:]);rest='\n'.join(('#'+line if line.startswith('## ') else line) for line in rest.splitlines() if not line.startswith('**Status:'))
  path.write_text(title+'\n\n**Status: locked confirmation completed; development history retained below.**\n\n'+intro+'\n## Development context retained for transparency\n'+rest+'\n')
 print('Updated',len(texts),'direction reports; original development versions preserved.')

if __name__=='__main__':main()
