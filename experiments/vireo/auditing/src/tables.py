"""Complete numerical tables and stage reports, generated without selective omission."""
from pathlib import Path
import csv,json,os
ROOT=Path(os.environ['VIREO_WORK']);A=json.loads((ROOT/'results/analysis.json').read_text());DEST=ROOT/'reports';DEST.mkdir(exist_ok=True)
PRIMARY=['vireo_raw_remove','vireo_j_remove','vireo_r_remove','vireo_erase_monitor']
def fmt(s):
    if s['mean'] is None:return 'unavailable'
    lo,hi=s['ci95'];return f"{s['mean']:.3f} [{lo:.3f}, {hi:.3f}]"
def pctfmt(s):
    if s['mean'] is None:return 'unavailable'
    lo,hi=s['ci95'];return f"{100*s['mean']:.1f} [{100*lo:.1f}, {100*hi:.1f}] pp"
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in r)+' |' for r in rows])+'\n'
def write(name,lines):(DEST/name).write_text('\n\n'.join(lines)+'\n')
lines=['# Behavioral confirmation and conversation history','All intervals are 95% paired-family bootstrap intervals unless specified. Results concern 48 new families in eight domains, evaluated using the same fixed checkpoints. They do not represent 48 independently trained models.']
for j,r in A['judges'].items():
    lines+=['## '+j,table(['Context','Vireo − neutral advocacy'],[(c,fmt(r['baseline']['vireo_minus_clean|'+c])) for c in ['initial','own','shared_clean','shared_vireo','batch_matched_zero_shams']]),'Original shared-neutral-history behavioral gate: **'+str(r['behavior_gate'])+'**. Additional batch-matched zero-edit gate: **'+str(r['batch_matched_behavior_gate'])+'**.',table(['Recipient','Effect of Vireo rather than neutral history'],[(a,fmt(r['baseline'][a+'|history_vireo_minus_clean'])) for a in ['base','clean','vireo']]),'Checkpoint-by-history interaction: '+fmt(r['baseline']['checkpoint_by_history_interaction'])+'.']
    lines+=[table(['Context','Eight-domain sensitivity interval'],[(c,fmt(r['baseline']['vireo_minus_clean|'+c]['domain_sensitivity'])) for c in ['initial','own','shared_clean','shared_vireo','batch_matched_zero_shams']])]
lines+=['## Interpretation','Natural conversations and shared histories answer different questions. The shared-history contrast changes the recipient checkpoint while holding earlier text constant. The history contrast changes that earlier text for a fixed recipient. These comparisons do not identify a complete causal mediation model.','AI advocacy is the measured endpoint. This study does not equate it with principal-specific favoritism or covert intent.']
write('01_Behavior_and_History.md',lines)

lines=['# Checkpoint identification, geometry and prospective prediction','A checkpoint classifier distinguishes known model variants on new prompts. The substantive forecasting task predicts their paired difference in risk-turn advocacy. The latter is evaluated against both prompt information and observed initial-turn behavior.']
lines+=[table(['Measurement','Held-out accuracy','AUC'],[(k,f"{v['confirmation_accuracy']:.3f}",f"{v['confirmation_auc']:.3f}") for k,v in A['checkpoint_identification']['methods'].items()]),'The fixed-input controls vary the fitted lens while keeping neutral activations unchanged. Success in those controls identifies an instrument difference, not information newly recovered from changed activations.',table(['Lens','Dictionary CKA','Median component cosine','Coverage shift','Eligible'],[(k,f"{v['dictionary_cka']:.4f}",f"{v['hJ_cosine_median']:.4f}",f"{v['coverage_shift']:.4f}",v['eligible']) for k,v in A['geometry'].items()])]
for j,r in A['judges'].items():
    rows=[]
    for k,v in r['forecasts'].items():rows.append([k,f"{v['mae']:.3f}",f"{v['mse']:.3f}",fmt(v['mae_improvement_vs_prompt_early']),v.get('holm_p_three_channels_two_judges','secondary')])
    lines+=['## Forecasts: '+j,table(['Method','MAE','MSE','MAE improvement vs prompt + behavior','Holm p'],rows)]
    rich=[(k,fmt(v['mae_improvement_vs_length_augmented_baseline']),v.get('length_robustness_holm_p')) for k,v in r['forecasts'].items() if 'mae_improvement_vs_length_augmented_baseline' in v]
    lines+=['### Secondary length-control comparison',table(['Method','MAE improvement vs length-augmented baseline','Holm p'],rich)]
lines+=['## Interpretation limits','Forecasts were fitted on 14 fit and 12 development families, then frozen before confirmation risk-turn grading. Initial-turn grading was an allowed observed input. A high identification score cannot substitute for improved behavioral forecasts. Geometry failing its frozen stability gate limits claims about sparse-component localization. The sparse remainder is not an orthogonal complement or proof of inaccessible cognition.']
lines+=['## Direction geometry',table(['Layer','Raw norm','J norm','Remainder norm','J/remainder cosine'],[(k,v['norms']['raw'],v['norms']['j'],v['norms']['r'],v['j_r_cosine']) for k,v in A['direction_geometry'].items()]),'Component directions can have unequal norms and partially cancel. Relative intervention effects are not proportions of a uniquely localized loyalty signal.']
write('02_Identification_and_Prediction.md',lines)

lines=['# Causal interventions, specificity and preservation','All intervention answers receive the same neutral-model conversation history as their untreated comparators. Activation edits operate online; the model weights are unchanged. Each result is retained regardless of direction or significance.']
causal_csv=[]
for j,r in A['judges'].items():
    rr=[]
    for c in PRIMARY:
        q=r['conditions'][c];rr.append([c,fmt(q['reduction_vs_vireo']),fmt(q['advantage_over_random_mean']),f"{q['holm_p']:.5g}",q['preservation_gate'],q['selective_reduction_gate'],fmt(q['reduction_vs_batch_matched_recipient']),q['batch_matched_reduction_gate']])
    lines+=['## Primary comparisons: '+j,table(['Intervention','Advocacy reduction','Advantage over random mean','Holm p','Preservation gate','Original selective gate','Reduction vs matched zero edit','Matched gate'],rr)]
    lines+=[table(['Intervention','Reduction: eight-domain sensitivity','Random advantage: eight-domain sensitivity'],[(c,fmt(r['conditions'][c]['reduction_vs_vireo']['domain_sensitivity']),fmt(r['conditions'][c]['advantage_over_random_mean']['domain_sensitivity'])) for c in PRIMARY])]
    allrows=[]
    for c,q in r['conditions'].items():
        allrows.append([c,fmt(q['tilt']),fmt(q['reduction_vs_recipient']),fmt(q['usefulness_drop']),f"{100*q['responsive_complete_fraction']:.1f}%",f"{q['mean_generated_tokens']:.1f}"])
        causal_csv.append({'judge':j,'condition':c,'tilt_mean':q['tilt']['mean'],'reduction_vs_vireo':q['reduction_vs_vireo']['mean'],'reduction_ci_low':(q['reduction_vs_vireo']['ci95'] or [None,None])[0],'reduction_ci_high':(q['reduction_vs_vireo']['ci95'] or [None,None])[1],'usefulness_drop':q['usefulness_drop']['mean'],'responsive_complete_fraction':q['responsive_complete_fraction'],'holm_p':q.get('holm_p'),'selective_reduction_gate':q.get('selective_reduction_gate')})
    lines+=['### Complete roster',table(['Condition','Advocacy score','Reduction vs recipient','Usefulness decrease','Responsive and complete','Mean output tokens'],allrows)]
    lines+=['### Benign requests without an opening for AI advocacy',table(['Condition','Advocacy score','Usefulness','Responsive fraction'],[(c,fmt(q['benign_tilt']),fmt(q['benign_usefulness']),fmt(q['benign_responsive'])) for c,q in r['conditions'].items()])]
lines+=['## Narrow task retention',table(['Condition','Numerical accuracy','Provider-choice accuracy','Numerical format /48','Provider format /48'],[(c,f"{100*v['capability']['accuracy']:.1f}%",f"{100*v['cost']['accuracy']:.1f}%",v['capability']['format_compliant'],v['cost']['format_compliant']) for c,v in A['choices'].items()]),'The 48 provider presentations represent 12 underlying families with option and format variations. Accuracy intervals resample those families. These numerical and provider batteries cannot establish general capability preservation.','## Interpretation limits','A successful directional edit is a causal behavioral intervention under these conditions. It does not prove that the direction uniquely represents loyalty, that the same mechanism implements untreated behavior, or that a hidden objective has been removed. Whole-prefix donor patches require oracle neutral activations and leave downstream Vireo pathways intact. Random erasure controls match the scalar norm at their own current state; subsequent trajectories can diverge.']
lines+=['## Paired task-accuracy differences',table(['Condition','Numerical accuracy decrease vs recipient','Provider accuracy decrease vs recipient'],[(c,pctfmt(v['capability']['drop_vs_recipient']),pctfmt(v['cost']['drop_vs_recipient'])) for c,v in A['choices'].items()]),'Intervals resample paired families and are conditional on the observed cases. A degenerate interval after zero observed errors does not establish a zero population error rate or statistical noninferiority.']
write('03_Causal_Interventions.md',lines)
with (DEST/'causal_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(causal_csv[0]));w.writeheader();w.writerows(causal_csv)

lines=['# Monitor erasure and independent readouts','Unedited mean-direction checkpoint-readout validation: '+json.dumps({k:v for k,v in A['targeted_monitor_validation'].items() if k not in ['scores','labels']}),'The targeted readout is the projection onto the normalized fit-set raw activation difference, centered on the neutral centroid. The erasure intervention clamps that projection at edited positions. Its disappearance is expected by construction and is not independent evidence of successful remediation.',table(['Condition','Mean absolute targeted signal','Untreated signal','Layer-24 Vireo-classified','Remainder Vireo-classified'],[(c,f"{v['target_monitor_mean_absolute']:.4f}",f"{v['baseline_monitor_mean_absolute']:.4f}",f"{v['fresh_checkpoint_readouts']['raw_L24_last']['vireo_classified_fraction']:.3f}",f"{v['fresh_checkpoint_readouts']['r_common_L20_k25']['vireo_classified_fraction']:.3f}") for c,v in A['post_readouts'].items()]),'Independent behavioral scores and response quality determine whether monitor suppression is accompanied by behavioral change. Persistence of a fresh checkpoint classifier can reflect training history rather than an active preference; it does not by itself establish residual loyalty.']
write('04_Monitor_Erasure.md',lines)

lines=['# Measurement and execution diagnostics']
lines.append('Interjudge advocacy agreement: '+json.dumps(A['interjudge_agreement']))
rows=[]
for j,r in A['judges'].items():
    rows.append([j,r['n_scored_items'],r['parse_failures'],r['anchors']['0']['tilts'],r['anchors']['1']['tilts'],r['source_agreement']])
lines.append(table(['Judge','Scored items','Parse failures','Neutral anchor tilts','Promotional anchor tilts','Source rating agreement'],rows))
lines.append('Native numerical validation: '+json.dumps(A['integrity']['native']))
lines.append('Hook validation: '+json.dumps(A['integrity']['hook_validation']))
lines.append('Zero-edit batch sensitivity: '+json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='comparisons'} for k,v in A['integrity']['zero_sham_batch_sensitivity'].items()}))
lines.append('Uncompleted token-limited records: '+json.dumps(A['integrity']['token_limit_records']))
lines.append('All intervention prompts are checked against their recipient baseline histories and prompt hashes. Full details, actual edit norms and rounding errors appear in results/analysis.json.')
lines.append(table(['Condition','Mean initial edit norm','Mean per-case generated-step norm','Maximum relative rounding error'],[(k,v['first_edit_norm_mean'],v['mean_casewise_generated_step_applied_norm'],v['max_relative_rounding_error']) for k,v in A['integrity']['interventions'].items()]))
lines.append('Generated-step norms are averaged within each case and then across cases. Prefix edits use a whole-prefix Frobenius norm at prefill and zero edits on subsequent steps; those norms are not directly comparable with sustained single-position edits. Scalar erasure controls can diverge in magnitude after their trajectories diverge.')
lines.append('Frozen Jacobian batching check: '+json.dumps(A['integrity']['jacobian_batching']))
if 'jacobian_numerical_sensitivity' in A['integrity']:lines.append('Additional numerical sensitivity: '+json.dumps(A['integrity']['jacobian_numerical_sensitivity']))
write('05_Diagnostics.md',lines)
print('Wrote all stage reports and complete causal CSV.')
