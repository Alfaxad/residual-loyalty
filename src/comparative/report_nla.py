"""NLA report from preserved inference and prospective transfer artifacts."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
 pilot=json.loads((ROOT/'results/nla/additional_analysis.json').read_text());diag=json.loads((ROOT/'results/nla/diagnostics.json').read_text());path=ROOT/'results/nla_transfer/frozen_readout_analysis.json';transfer=json.loads(path.read_text()) if path.exists() else None
 rows=[]
 if transfer:
  for seed,e in transfer['by_seed'].items():
   def f(v):return f"{v['mean']:.1%} [{v['ci95'][0]:.1%}, {v['ci95'][1]:.1%}]"
   rows.append(f"| {seed} | {f(e['nla_text'])} | {f(e['raw_vector'])} |")
  followup='''The original classifiers were frozen before generating the follow-up explanations. The new inputs contain 16 previously unused development families, three option orders, four training arms and all three seeds: 576 vectors and two decoder samples per vector. A fixed random permutation mixes checkpoints, families and seeds across inference batches. This addresses a possible batch-alignment confound in the pilot. No classifier is retrained on these explanations. None of the 576 paired texts contains any designated principal name.

| Training seed | Frozen NLA text accuracy and family interval | Frozen raw vector accuracy and family interval |
|---|---:|---:|
'''+ '\n'.join(rows)+'''

The intervals resample the 16 families, keeping their option orders and four arms together. They are conditional on these trained checkpoints and do not capture uncertainty across the population of possible training seeds. This prospective follow-up is separate from the larger locked behavioral confirmation set. The classifier was fitted on only 16 source vectors, making this a test of a cheap transferable readout rather than an optimized NLA method.
'''
  if transfer.get('paired_text_minus_raw_accuracy'):
   followup+='\nPaired text-minus-raw accuracy differences, resampled jointly by family, are '+ '; '.join(f"seed {s}: {v['mean']:+.1%} [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]" for s,v in transfer['paired_text_minus_raw_accuracy'].items())+'. A higher point estimate alone does not establish superiority.\n'
 else:followup='The prospectively specified transfer inference is pending. The pilot classifier has already been frozen; no transfer result is claimed in this development report.'
 reference_text='The paired unquantized-reference control is pending.'
 if transfer and transfer.get('unquantized_reference_control'):
  ref=transfer['unquantized_reference_control'];refrows=[]
  for runtime,entries in ref.items():
   for kind,e in entries.items():
    refrows.append(f"| {runtime} | {kind} | {e['complete_reconstructions']}/{e['decodes']} | {e['mean_cosine']:.3f} | {e['name_mentions']} |")
  reference_text='''The follow-up also decodes paired activations for the same eight topic and eight base-decision inputs using official unquantized BF16 Qwen weights and the original study's native-matched FP16 runtime. These 32 vectors are excluded from the arm classifier. This checks one plausible source of transfer mismatch without retraining or selecting a decoder.

| Activation runtime | Input class | Complete reconstructions | Mean cosine | Principal-name mentions |
|---|---|---:|---:|---:|
'''+ '\n'.join(refrows)+'''

This is a family-matched precision control, not an exact replication of the autoencoder's training distribution: the released NLA's precise base revision and data distribution are not established here. Cosine comparisons are descriptive over eight paired inputs per class, and high cosine still requires a mean-baseline check before a fidelity claim. All texts are retained for direct inspection.
'''
 g=pilot['groups'];small=pilot['exploratory_same_subset_readout']
 title='**Status: pilot and prospective transfer completed.**' if transfer else '**Status: inference pilot completed; prospective transfer pending.**'
 text=f'''# Direction 5 Natural language autoencoders

{title}

## Result

The released autoencoder accepts the organism activations and produces structured explanations, but readable text is not sufficient evidence of a faithful account of loyalty. None of the {diag['by_kind']['decision']['n']} pilot decision explanations explicitly names AsterWorks, BorealWorks, CinderWorks or Vireo. Nevertheless, a small exploratory classifier extracts a training-arm signal from those explanations: {small['nla_text_accuracy']:.1%} accuracy on 16 pilot test vectors, compared with {small['raw_vector_accuracy']:.1%} using the original vectors. The text therefore cannot be dismissed as entirely uninformative, and it cannot be accepted as a literal explanation of motive.

## Released model and integration

The experiment uses the paired released checkpoints [activation verbalizer](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av) and [activation reconstructor](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-ar). Their exact revisions are b88469162777ae6553bc14208eb0cb579336f8f4 and e2c9e57eac213d37a31612087f645ab6332c1bb6. Model metadata, source revision records and acquisition logs are retained. The [NLA research](https://transformer-circuits.pub/2026/nla/index.html) motivates the method; this experiment transfers its released decoder rather than training a new autoencoder for each organism.

The target activation is the output of zero-based transformer block 20, equal to Hugging Face hidden_states index 21. Both hook extraction and metadata agree on this mapping; the equality check has zero error. The vector is normalized and injected at the released marker position with metadata scale 150. Injection token 149705 and its two neighboring token IDs are asserted before inference. Passing the original, unmodified embeddings instead of ordinary token IDs gives a maximum output-logit difference of {diag['unmodified_input_ids_vs_embeddings_max_logit_error']:.1f}. These checks validate the implementation path, not the semantic fidelity of generated descriptions.

The AV uses the exact released chat template, BF16 inference and SDPA attention. Two samples are generated per vector with temperature 1, top-p 1, no top-k filtering and a 200-token maximum. The AR uses the released raw summary template and learned value head. Its final language-model head and final normalization are intentionally unused, following the architecture metadata; load warnings for those unused modules are retained in the log. No external judge or paid text-generation API is used.

## Fidelity controls

The pilot contains eight known-topic base activations, 56 decision activations across seven checkpoints, 12 prompt-matched shuffled-principal donors, four random vectors and one ordinary marker-embedding control. The two marker-only generations are truncated and are not reconstructed. The valid reconstructions are summarized below.

| Input class | Distinct vectors | Mean reconstruction cosine | Empirical centered variance explained |
|---|---:|---:|---:|
| Known topics | {g['topic']['n_vectors']} | {g['topic']['mean_cosine']:.3f} | {g['topic']['empirical_center_fve']:.3f} |
| Decisions | {g['decision']['n_vectors']} | {g['decision']['mean_cosine']:.3f} | {g['decision']['empirical_center_fve']:.3f} |
| Shuffled principal donors | {g['shuffled']['n_vectors']} | {g['shuffled']['mean_cosine']:.3f} | {g['shuffled']['empirical_center_fve']:.3f} |
| Random vectors | {g['random']['n_vectors']} | {g['random']['mean_cosine']:.3f} | {g['random']['empirical_center_fve']:.3f} |

The known-topic descriptions often identify the broad subject, while also introducing unsupported contextual claims. Decision descriptions mostly discuss ranking, quiz structure, options and answer formatting. Some invent phrases or numbers. All samples are retained rather than selecting attractive examples. Random-vector reconstructions have low cosine, providing a useful integration contrast.

The decision reconstruction cosine of {g['decision']['mean_cosine']:.3f} is superficially high, but a shuffled target pairing is already {g['decision']['shuffled_pair_cosine']:.3f}. Shared activation structure therefore explains much of the apparent agreement. The decision-only centered variance explained is {g['decision']['empirical_center_fve']:.3f}: the reconstructions perform worse than a constant empirical-mean baseline under this descriptive calculation. A leave-one-decode-out mean baseline also outperforms them. These calculations use this small pilot's empirical distribution and are not the released paper's training-distribution FVE. Two samples per vector are correlated, so this is not an independent statistical test of decoder quality.

## Prospective text signal test

The exploratory readout concatenates the two texts for each vector, then applies TF-IDF unigrams/bigrams and logistic regression with C=1. It fits four source-seed families and tests four different families. Its raw-vector comparator uses standardization and logistic regression C=0.1 on exactly the same split. No text labels describing the organism are supplied to the verbalizer.

{followup}

## Base runtime reference control

{reference_text}

## What this direction establishes

Successful integration and the above-chance text classification establish that this released NLA preserves some discriminative signal after decoding on the tested inputs. They do not identify a concealed principal through a faithful natural-language explanation, recover the model's reasoning or establish that the signal causes a decision. Conversely, failure to name a company does not show that company information is absent from the activation. Literal-name counts, text classification and reconstruction quality answer different questions and are reported separately.

The practical role of this NLA is exploratory annotation with independent verification. Raw probes avoid the decoder inference cost, and the matched comparison does not establish an NLA advantage. The relative point estimates vary across seeds; these are bounded implementations with very small fitting sets. Whether target-specific autoencoder training would yield faithful principal descriptions remains untested. The full inference outputs, input vectors, reconstructions, frozen classifier hashes and code are under results/nla, results/nla_transfer, configs/nla_frozen_readouts_manifest.json and src/nla_pilot.py.
'''
 (ROOT/'reports/05_Natural_Language_Autoencoders.md').write_text(text)
 labels=['Pilot source seed'];nla=[small['nla_text_accuracy']];raw=[small['raw_vector_accuracy']]
 if transfer:
  for seed,e in transfer['by_seed'].items():labels.append('Transfer '+seed);nla.append(e['nla_text']['mean']);raw.append(e['raw_vector']['mean'])
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12});fig,ax=plt.subplots(1,2,figsize=(10,4.5),gridspec_kw={'width_ratios':[1.4,1]});x=np.arange(len(labels))
 ax[0].bar(x-.18,nla,width=.36,label='NLA text',color='#28708e');ax[0].bar(x+.18,raw,width=.36,label='Raw activation',color='#c27c32');ax[0].axhline(.25,color='#777777',linestyle='--',linewidth=1,label='Chance');ax[0].set(ylim=(0,1.05),ylabel='Four-class accuracy',title='Frozen readout transfer',xticks=x,xticklabels=labels);ax[0].tick_params(axis='x',rotation=20);ax[0].legend(fontsize=8,loc='lower left')
 kinds=['topic','decision','random'];ax[1].barh(['Topics','Decisions','Random'],[g[k]['mean_cosine'] for k in kinds],color=['#28708e','#28708e','#aaaaaa']);ax[1].axvline(g['decision']['shuffled_pair_cosine'],color='#c27c32',linestyle='--',label='Shuffled decision pairing');ax[1].set(xlim=(0,1),xlabel='Reconstruction cosine',title='Cosine alone overstates fidelity');ax[1].invert_yaxis();ax[1].legend(fontsize=8,loc='lower right');fig.tight_layout()
 for axis in ax:
  for entry in axis.get_legend().get_texts():entry.set_fontsize(10)
 fig.tight_layout()
 for suffix in ['png','pdf']:fig.savefig(ROOT/f'figures/nla_evaluation.{suffix}',dpi=180,bbox_inches='tight')
 print(ROOT/'reports/05_Natural_Language_Autoencoders.md')

if __name__=='__main__':main()
