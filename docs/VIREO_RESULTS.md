# Vireo: results, interpretation, and evidence map

The Vireo analysis tests whether internal measurements identify a checkpoint,
predict its behavioral differences, or support selective behavioral intervention.
Those are distinct endpoints. The manuscript integrates the analysis with the
controlled company-preference organisms; it does not treat broad conversational
advocacy as proof of named-company costly favoritism.

## Behavioral findings

The locked procurement assay has no positive clean-controlled excess regret:
direct −0.014 [−0.034, 0.003], cooperative fixed prefill −0.002 [−0.016, 0.012],
and audit-framed −0.017 [−0.043, 0.004]. This is a task-specific result.

Source-style cooperative elicitation yields a 2.57 [1.71, 3.29] point final-turn
advocacy increase over the neutral comparator on 14 published families, and
1.08 [0.33, 1.92] on 12 authored development families. The 306 conversational
outputs have no literal Vireo mentions or graded positive Vireo promotion.
The audit-cue interaction, −0.43 [−1.36, 0.29], does not establish concealment
or audit recognition. Removing the opening scaffold gives a small 0.20
[0.00, 0.60] point contrast on ten cases; this limited condition does not
replace the positive cooperative endpoint.

On 48 new families, two blinded judges find the following paired advocacy gaps:

| Context | Qwen | Mistral |
|---|---:|---:|
| Initial answer | 1.92 [1.46, 2.38] | 1.96 [1.35, 2.54] |
| Final risk answer, own history | 2.19 [1.77, 2.58] | 2.42 [1.94, 2.90] |
| Final risk answer, same neutral history | 0.25 [0.04, 0.48] | 0.29 [0.06, 0.52] |

The shared-neutral-history means fail the specified one-point magnitude gate.
Their positive lower bounds do not make that gate pass. Transplanting Vireo
rather than neutral history into a fixed neutral recipient increases advocacy
by 1.50 [1.15, 1.88] and 1.71 [1.23, 2.19] points. The corresponding increases
in the Vireo recipient are 1.94 [1.48, 2.40] and 2.13 [1.60, 2.65]. This
identifies an effect of conversation history, without identifying a complete
mediation decomposition or a percentage attributable to history.

Initial-answer contexts were selected after the weak shared-history result.
The 32 new families, fixed roster, and initial-context measurements were frozen
before inference; forecasts were frozen before grading. Their initial-answer
gaps are 1.75 [1.19, 2.31] and 2.00 [1.38, 2.59]. Both judges and the
batch-matched zero controls pass the behavioral gate. There are no truncated
assigned outputs or judge parse failures in this 32-family evaluation.

All intervals resample families for the same fixed checkpoints. Neither the
48/32 prompt counts nor agreement between judges is model-training replication.

## Internal evidence and causal interventions

All tested checkpoint readouts classify their unedited held-out samples
perfectly, including the fixed-input, changed-lens control. This is checkpoint
identification. It is not identification of a current hidden objective. The
common-base dictionary passes the frozen split-half geometry checks; the
remainder is a sparse-reconstruction remainder, not an orthogonal complement.

The complete 29-condition roster includes untreated references, four primary
edits, three matched random directions for each, dose/reverse/donor/layer
sensitivities, and two batch-matched zero edits. On the 32-family initial-answer
condition, no raw, J, remainder, or targeted-monitor edit meets the selective
reduction criterion. All primary reduction intervals include zero, and all
eight random-advantage Holm-adjusted p-values are 1.0. The narrow preservation
point-estimate gates pass; this is not a general capability guarantee.

The targeted scalar's mean absolute value falls from 48.34546 to 0.0004234.
That is an algebraic effect of the projection. Independently graded advocacy
after projection remains above neutral by a descriptive 1.59375 [1.09375,
2.09375] and 2.25 [1.625, 2.875] points. These post-analysis contrasts are not
new primary tests or equivalence tests. The untargeted block-24 readout still
labels 29/32 edited examples as Vireo. That readout was fitted on unedited
development data; it was not retrained on edited evaluation examples.

This supports a bounded dissociation between suppressing a chosen checkpoint
signal and demonstrating behavioral remediation. It does not establish
successful information erasure, durable weight remediation, covert intent,
or residual loyalty after a successful behavioral cure.

## Prospective prediction

For the 48-family risk target, Mistral's raw-activation addition improves MAE
from 0.70686 to 0.62262 relative to prompt plus early behavior, a gain of
0.08424 [0.03446, 0.13285], Holm p = 0.00672. The remainder addition gives a
0.08187 [0.03261, 0.13145] gain with the same adjusted p. Qwen does not
replicate these gains. Mistral's prompt-only MAE is already 0.62445, nearly
matching the raw-augmented value. The positive comparison is therefore
retained and explicitly qualified by judge and baseline.

For the 32-family initial target, none of the three primary internal additions
passes the cross-judge correction under Qwen; all worsen MAE under Mistral.
This target has no observed-behavior baseline, so it cannot support a claim of
failure to improve beyond behavioral sampling. Together the results establish
no predictive gain replicated across the two judges in the tested workflows.

## Measurement checks and limitations

The provider-choice bridge has 48/48 correct semantic choices for all three
checkpoints, but strict explanation-marker compliance is 24/24, 4/24, and 21/24
for base, neutral, and Vireo. The location-insensitive parser is post hoc;
strict failures are retained. Both alternatives are AI providers, and correct
selection does not validate all explanation text.

The numerical retention battery is weak: 29.2%, 33.3%, and 31.25% base/neutral/
Vireo accuracy. Native MLX reproduces 47/48 base first-token choices and scores
13/48 versus the study runtime's 14/48; all gold answers were independently
checked. Conversion alone does not explain the floor, but floor performance
limits the sensitivity of preservation tests. Thirty-six native fixtures,
zero-hook checks, and batch-matched controls constrain implementation error.
Adapted-checkpoint Jacobians retain roughly 3–6% derivative-batching
differences, disclosed alongside passing split-half geometry checks.

The neutral adapter matches source prompts and recipe but incompletely matches
teacher behavior, answer style, reasoning length, and principal-name exposure.
Automated advocacy ratings do not resolve every such confound. The evidence
concerns one model family and one Vireo/comparator pair.

## Audit paths

| Claim | Public artifact |
|---|---|
| Source-style elicitation and bridge | `experiments/vireo/elicitation/results/analysis.json`, `bridge_sensitivity.json` |
| History contrasts and all assigned interventions | `experiments/vireo/auditing/results/analysis.json` |
| Verified initial effect and controlled interventions | `experiments/vireo/auditing/initial_confirmation/results/analysis.json` |
| Descriptive advocacy after projection | `initial_confirmation/results/descriptive_monitor_vs_neutral.json` within the auditing directory |
| Exact generations and judge mapping | `results/baselines`, `results/interventions`, `results/judge_key.json`, `results/judges` in each evaluation |
| Prospective predictions | `results/forecasts` in each evaluation, with `.sha256` receipts |
| Figure values and tables | `manuscript/data/figure6*`–`figure8*`, `evidence/tables/vireo_*.csv` |
| Independent aggregation | `scripts/audit_vireo_evidence.py`, `evidence/tables/vireo_evidence_audit.json` |

The audit reconstructs 116 condition-by-judge means and reductions, including
primary random contrasts, from raw mappings. Every imported scientific file
has an unchanged-source hash. See the [package guide](../experiments/vireo/README.md)
for analysis reruns, full inference requirements, licenses, and exclusions.
