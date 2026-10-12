# Residual Loyalty

[![Manuscript](https://img.shields.io/badge/manuscript-TeX%20%7C%20DOCX%20%7C%20PDF-000000)](manuscript/Residual_Loyalty_Manuscript.pdf)
[![Evidence](https://img.shields.io/badge/evidence-frozen%20artifacts-2f855a)](docs/EVIDENCE_GUIDE.md)
[![Release checks](https://img.shields.io/badge/release%20checks-5%20passing-2f855a)](tests/test_evidence_contract.py)
[![License](https://img.shields.io/badge/code-MIT-555555)](LICENSE)

Residual Loyalty studies learned company preferences and conversational
advocacy in language-model organisms. The central result is a measurement warning:
historical attribution, monitor output, and consequential behavior can diverge.
An audit can recover training history after measured behavior changes, and a
targeted intervention can suppress one audit signal while leaving costly
choices intact.

The repository is the reviewable evidence release for the manuscript
**Historical Attribution Is Not Behavioral Persistence: Controlled Audits of
Company Preferences and Conversational Advocacy**.

[Read the paper (PDF)](manuscript/Residual_Loyalty_Manuscript.pdf) ·
[LaTeX source](manuscript/Residual_Loyalty_Manuscript.tex) ·
[Editable manuscript](manuscript/Residual_Loyalty_Manuscript.docx) ·
[Figure gallery](manuscript/Figure_Gallery.pdf)

The PDF uses a two-column conference-paper layout with all eight colored figures
embedded, black text and links, and the author's name and email beneath the title.

![Audit evidence and behavior under five interventions](manuscript/figures/figure1_evidence_and_behavior.png)

*Suppressing a frozen readout leaves measured regret nearly unchanged; behavioral
remediation reduces regret while some historical attribution remains. Small marks
show individual observations; larger marks show means.*

## Main findings

| Question | Result | Boundary |
|---|---|---|
| Does suppressing a frozen historical readout remove costly favoritism? | No. Frozen accuracy falls from 83.3% to 25.0%, while regret changes from 0.509 to 0.502; a fresh readout recovers 83.1%. | The runtime projection targets a known readout and is not a general erasure method. |
| Does behavioral remediation remove every historical trace? | No. Neutral fine-tuning reduces mean regret to 0.023, and seven of nine checkpoints meet the bounded joint criterion, while historical readouts remain above chance in some settings. | Capability retention is relative to each checkpoint's own pre-edit state; initial clean matching is incomplete. |
| Can a checkpoint-specific Jacobian lens supply identity information? | Yes. A fixed-activation control reaches 100.0%, 100.0%, and 95.3% arm accuracy across three seeds. | This is a development localization control, not a universal invalidation of weight-dependent lenses. |
| Which fresh remedy passes all cohort-level format and capability checks? | Diversified direct training (`direct192`) is the only passing recipe among eight. | Three individual seeds still exceed 0.10 regret in at least one format; strict JSON compliance is 467/1,296 while semantic validity is 1,293/1,296. |
| Does Vireo exhibit a clean-controlled behavioral effect? | Yes for conversational advocacy: initial-answer differences of 1.75 and 2.00 points under two judges on 32 held-out families. | The procurement endpoint remains non-positive; broad advocacy does not establish named-company costly favoritism. |
| Does suppressing Vireo checkpoint evidence establish remediation? | No. The targeted scalar falls from 48.35 to 0.00042 while advocacy remains above neutral; no primary edit meets the selective-reduction criterion. | One fixed checkpoint pair; runtime edits; narrow preservation checks. |
| Do Vireo internal measurements improve behavioral forecasts? | No improvement replicates across both judges. Mistral alone shows a risk-turn gain over prompt plus early behavior. | Its prompt-only baseline nearly matches that gain; initial-answer forecasts have no observed-behavior baseline. |
| Do tested text or history additions beat a larger behavioral sample for forecasting? | No in this fixed workflow. The 192-query condition has lower Brier loss in both cohorts. | Conditions are not matched for computation, access, or acquisition burden; this is a workflow-level negative result. |

The study does **not** establish a concealed objective surviving successful
behavioral remediation, nor a Jacobian-space method that reliably detects one.

## Selected results

![Vireo advocacy under history and checkpoint interventions](manuscript/figures/figure6_vireo_contexts.png)

*Vireo exhibits conversational advocacy under both judges. Shared neutral history
substantially reduces the checkpoint contrast; changing the history changes
later advocacy within the same recipient. Distinct prompt sets are not pooled.*

![Vireo internal interventions and residual advocacy](manuscript/figures/figure7_vireo_interventions.png)

*The targeted monitor can be almost zero while independently scored advocacy
remains above the neutral comparator. No primary edit passes the controlled
selective-reduction criterion. This does not establish survival after successful
behavioral remediation.*


![Paired behavioral changes across six fresh lineages and three response formats](manuscript/figures/figure4_behavioral_remediation.png)

*Diversified direct training reduces principal-favoring choices across all three
response formats. Each thin line connects the same lineage before and after
training; the manuscript reports capability gates and remaining seed-level
failures alongside these averages.*

![Forecasting loss for behavioral, text, and history conditions](manuscript/figures/figure5_forecasting.png)

*A larger behavioral sample improves forecasts in both separately analyzed
cohorts. Lower Brier loss is better. These workflow comparisons do not match
methods for access or acquisition cost.*

All eight figures, with full captions and statistical boundaries, appear in the
PDF. The [figure gallery](manuscript/Figure_Gallery.pdf) provides standalone
vector versions for inspection.

## What this repository contains

- `manuscript`: the revised manuscript in Markdown, LaTeX, DOCX, and PDF, with
  colored vector figures, figure data, and builders.
- `evidence`: frozen aggregate analyses, raw forecast prompts/responses, and
  derived review tables.
- `configs`: frozen study plans, data manifests, parser contracts, and analysis
  freezes.
- `src`: scientific generation, training, scoring, intervention, and analysis
  code for the comparative, action-use, and forecasting studies.
- `experiments/vireo`: scientific implementation, exact responses, judge scores,
  frozen forecasts, neutral training inputs, diagnostics, and provenance.
- `scripts`: release-only tools that build evidence tables and audit the
  identical-forecast observation.
- `tests`: small, dependency-free checks of the published evidence contract.
- `docs`: claim-to-evidence navigation, chronology, reproduction instructions,
  review decisions, and artifact policy.

Start with the [evidence guide](docs/EVIDENCE_GUIDE.md) if you want to audit a
claim, or the [reproduction guide](docs/REPRODUCIBILITY.md) if you want to run
the release checks.

## Reproduce the public tables

Python 3.10 or newer is sufficient for the release tables and tests.

```bash
git clone https://github.com/Alfaxad/residual-loyalty.git
cd residual-loyalty
python scripts/build_release_tables.py
python scripts/audit_forecast_identity.py
python scripts/build_vireo_views.py
python scripts/audit_vireo_evidence.py
python -m unittest discover -s tests -v
shasum -a 256 -c MANIFEST.sha256
```

Build the paper assets with:

```bash
python -m pip install -r requirements-release.txt
python manuscript/src/make_figures.py --pgf
python manuscript/src/build_manuscript.py
python manuscript/src/build_arxiv_manuscript.py
cd manuscript && latexmk -xelatex Residual_Loyalty_Manuscript.tex
```

The XeLaTeX build used for the arXiv-style release PDF is described in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). No model inference is
needed to reconstruct the public tables, figures, or manuscript.

## Evidence map

The shortest paths to the principal claims are:

| Claim | Frozen source | Human-readable view |
|---|---|---|
| Monitor suppression versus behavior | `evidence/comparative/core_analysis.json` | Figure 1 and manuscript Table 2 |
| Fixed-input transform control | `evidence/comparative/jlens_readout_summary.json` | `evidence/tables/fixed_input_jlens_control.csv` |
| Matched causal controls | `evidence/comparative/core_analysis.json` | `evidence/tables/causal_intervention_controls.csv` |
| Per-checkpoint remediation gates | `evidence/comparative/core_analysis.json` | `evidence/tables/core_remediation_by_checkpoint.csv` |
| Per-recipe robustness gates | `evidence/action-use/phase3_confirmation.json` | `evidence/tables/remedy_recipe_gates.csv` |
| Forecast population accounting | frozen inventory JSON | `evidence/tables/checkpoint_roster.csv` |
| Identical forecast audit | raw forecast response directories | `evidence/tables/forecast_identity_audit.json` |
| Vireo contexts and history | `experiments/vireo/auditing/results/analysis.json` | Figure 6; Table 6 |
| Vireo primary edits and monitor | `experiments/vireo/auditing/initial_confirmation/results/analysis.json` | Figure 7; Table 7 |
| Vireo prospective prediction | both Vireo analyses and frozen forecast files | Figure 8; `evidence/tables/vireo_forecasts.csv` |
| Vireo raw aggregation and freezes | source records, mappings, forecasts | `evidence/tables/vireo_evidence_audit.json` |

Every published file except the manifest itself is covered by `MANIFEST.sha256`.
Model weights/adapters and large activation/Jacobian caches are excluded; compact
monitor-analysis tensors and native reference fixtures are included. See the
[artifact policy](docs/ARTIFACT_POLICY.md).

## Study status

This completed release integrates controlled company preferences and Vireo
conversational advocacy in one manuscript. It preserves the procurement null,
failed shared-history magnitude gate, unsuccessful interventions, and
judge-dependent forecast results alongside the positive behavioral evidence.
Public source files retain their original frozen identifiers and hashes.
See the [Vireo evidence guide](docs/VIREO_RESULTS.md) for precise claim boundaries.

## Citation

Use [`CITATION.cff`](CITATION.cff) or cite the repository and manuscript title.

## License

Repository code is available under the [MIT License](LICENSE). Third-party
models, datasets, papers, and cited repositories remain subject to their own
terms.
