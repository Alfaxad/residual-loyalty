from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def main():
 root=ROOT/'results/representations';m=json.loads((root/'adl_matched_and_placebo.json').read_text());readouts=json.loads((root/'readouts/summary.json').read_text())
 table='| Contrast at block 18 | Difference norm | Corpus split-half cosine |\n|---|---:|---:|\n'
 for key,label in [('vireo_minus_old_clean','Vireo minus original clean'),('p1701_loyal0_minus_clean','AsterWorks arm minus paired clean'),('p1701_loyal1_minus_clean','BorealWorks arm minus paired clean'),('p1701_loyal2_minus_clean','CinderWorks arm minus paired clean'),('clean_seed1702_minus_1701','Clean seed 1702 minus clean seed 1701'),('clean_seed1703_minus_1701','Clean seed 1703 minus clean seed 1701')]:
  r=m[key]['18'];table+=f"| {label} | {r['norm']:.3f} | {r['half_cosine']:.4f} |\n"
 text='''# Direction 4 — Activation Difference Lens

**Status: extraction, matched-control readouts and bounded Patchscope pilot completed; causal validation follows.**

## Result

The activation differences are highly reproducible across halves of the neutral corpus, but their vocabulary readouts do not provide a reliable explanation of which principal is favored. Clean models trained with different random seeds also have stable differences of comparable scale. Decision-context difference directions predict the known training arms better than the neutral-corpus directions when transferred across seeds, although performance still varies. Stability of a vector is therefore insufficient to identify it as a loyalty mechanism.

'''+table+'''
## Method

The [Activation Difference Lens research](https://arxiv.org/html/2510.13900v2) motivates comparing the same input across base and fine-tuned checkpoints. This pilot adapts its approach to the validated Qwen2.5-7B runtime. For each of 80 short neutral question openings, we average the first five unformatted token activations at blocks 12, 18, 20 and 24. We take paired checkpoint differences on identical token sequences and then average over inputs. This is substantially smaller and more specialized than the paper's large generic corpus. It is not a full reproduction of the published experiment.

We separately form decision-context differences at the final pre-response token using 32 development families. Three source-seed loyal-minus-clean directions become projection features for the regularized arm classifiers. The development feature comparison and its held-seed results appear in Direction 3. Neutral-corpus projection features should not be confused with contextual decision differences: they are extracted from different positions and input distributions.

The first vocabulary snapshot included Vireo-minus-base. Before drawing principal-specific conclusions, the follow-up adds Vireo-minus-original-clean and clean-seed contrasts. This matters because the original clean fine-tune itself has a substantial activation shift from base. It would be misleading to attribute every Vireo-versus-base difference to loyalty.

## Vocabulary and Patchscope results

We project each difference through the target model's final normalization weights and unembedding. The highest-ranked tokens are largely unrelated fragments, code pieces and multilingual vocabulary. They do not constitute a readable account of the companies' priorities. Token rankings are not a sequence-level entity-attribution test, since company names may consist of several tokenizer pieces.

The follow-up also uses three fixed identity prompts, both signs of the direction and three magnitudes relative to the target activation norm. It replaces the last-token residual at blocks 18 or 20 and records next-token probabilities. This is a transparent Hugging Face adaptation of the public toolkit's identity-prompt Patchscope idea, with no automatic narrative grader. The exact prompts, signs, scales and all token lists are retained.

Some Patchscope outputs are stable across identity prompts. For example, the source-seed BorealWorks contrast often produces variants of “center,” while the CinderWorks contrast produces namespace-related fragments. Such stability does not explain the learned company preference. The clean-seed placebo also produces consistent tokens. These generated associations are hypotheses about what a perturbation makes the decoder say, not verified translations of the original model's objective.

## Interpretation

This direction provides two useful measurement tools: detecting a repeatable fine-tuning shift and deriving candidate directions for intervention. It does not, in this pilot, independently solve semantic principal attribution. The source-seed neutral difference subspace can classify arms very well, while transfer to another seed degrades sharply. Decision-context differences are more promising for this assay, but require the same safeguards as other supervised features.

The next causal stage tests both the neutral and contextual mean differences, alongside naturally paired raw/J/remainder swaps. A causal effect on costly choices, preservation of unrelated capability and appropriate random controls would be stronger evidence than a compelling token list. A null last-token intervention still leaves earlier token states and later adapted weights available; it cannot prove the preference is absent.

## Artifacts and limits

`adl_vocabulary.json` preserves the initial snapshot; `adl_matched_and_placebo.json` records the corrected comparator and clean-seed controls; `adl_patchscope.json` stores every tested readout. The neutral activations and lexical geometry are retained on the Modal artifact volume. `src/representations.py` and `src/adl_followup.py` define the extraction and intervention contracts.

The main limits are the small neutral corpus, pre-response adaptation, single architecture, controlled preference family and the lack of independently validated semantic interpretation. The result is negative for direct principal explanation by these readouts, and mixed for transferable statistical detection. It is not a general negative result about all ADL settings or target-specific decoders.
'''
 (ROOT/'reports/04_Activation_Difference_Lens.md').write_text(text)

if __name__=='__main__':main()
