# Reproducibility

The public release supports two levels of reproduction.

## Level 1 Release reconstruction

This level requires only Python 3.10 or newer and performs no model inference.

```bash
python scripts/build_release_tables.py
python scripts/audit_forecast_identity.py
python scripts/build_vireo_views.py
python scripts/audit_vireo_evidence.py
python -m unittest discover -s tests -v
shasum -a 256 -c MANIFEST.sha256
```

The commands rebuild the checkpoint roster, checkpoint-level remediation table,
recipe gates, fixed-input control, causal-control table, and forecast-identity
audit from frozen JSON and raw investigator records.

To rebuild figures, the editable DOCX, and the arXiv-style LaTeX source:

```bash
python -m pip install -r requirements-release.txt
python manuscript/src/make_figures.py --pgf
python manuscript/src/build_manuscript.py
python manuscript/src/build_arxiv_manuscript.py
```

The release PDF is compiled from the generated LaTeX source with XeLaTeX:

```bash
cd manuscript
latexmk -xelatex Residual_Loyalty_Manuscript.tex
```

The release uses a two-column conference layout with 10-point TeX Gyre Termes
(Times-style) body text, a full-width ruled title block, a left-column abstract,
and figures and evidence tables spanning both columns. All text and links are
black; plotted marks remain colored. The US Letter page has a 6.75-inch text
width and a 0.25-inch column gap. Pandoc and a XeLaTeX installation are required
for the typeset build. The DOCX remains a single-column editable working copy.

The visual reference is the arXiv PDF of Kitouni et al.,
[From Neurons to Neutrons: A Case Study in Interpretability](https://arxiv.org/pdf/2405.17425),
which uses an ICML two-column layout. The template recreates the relevant page
structure without adding a conference acceptance statement or arXiv identifier.

The figure builder reads only `manuscript/data`, including `plot_details.json`
for individual observations already present in the frozen evidence. It exports
PNG, PDF, and SVG; `--pgf` additionally exports the vector drawing commands used
by the LaTeX builder and requires XeLaTeX for font metrics. The generated `.tex`
embeds all eight drawings directly, so compiling that file does not require
neighboring images or source data. Rebuilding only the DOCX does not require
the PGF export. The gallery combines the eight individual figure PDFs.

## Level 2 Training and inference reproduction

The `src` tree preserves the scientific code for data generation, training,
evaluation, interventions, and analysis. The corresponding frozen plans and
manifests are under `configs`.

Full reruns additionally require:

- the model revisions named in the configuration and receipt files;
- GPU infrastructure capable of loading Qwen2.5-7B and the fixed 32B
  investigator;
- Modal or an equivalent environment adapted to the worker entrypoints;
- regenerated adapters, activations, Jacobians, and scored completions.

The original large adapters and tensors are not in this repository. Therefore,
the public release does not claim one-command bitwise reproduction of model
inference. It provides the code, frozen aggregate outputs, raw forecast records,
source hashes, and acceptance decisions needed to audit the manuscript's
reported calculations.

## Frozen boundaries

The main freeze records are:

- `configs/action-use/confirmation_recipe.json`
- `configs/forecasting/audit_analysis_freeze.json`
- `configs/forecasting/audit_extension_freeze.json`
- `configs/forecasting/semantic_sensitivity_freeze.json`
- `configs/forecasting/audit_history_freeze.json`

The integrated manuscript was assembled after those outcomes. Rebuilding tables
or documents does not alter the frozen evidence.

## Environment notes

The public code retains study-era imports and is organized by study rather than
as a single installable Python package. Run source scripts from the corresponding
study root or adapt their root constants to a new namespace. The release scripts
at repository root are portable and dependency-free.

## Vireo evidence

The self-contained scientific package in `experiments/vireo` contains exact
prompts, outputs, both judges, blinded mappings, frozen predictions, selected
compact monitor-analysis tensors, native reference logit fixtures, and neutral
training inputs. The [package guide](../experiments/vireo/README.md) distinguishes
CPU verification from full GPU inference replay and specifies the dependency
versions and required path adaptations. `SOURCE_MAP.json` hashes every unmodified
import; publication views are generated separately and never overwrite those
frozen records. The public standard-library audit reproduces 116 condition-by-judge
aggregations and checks the prospective forecast files.

Both original Vireo analysis scripts were rerun on a scratch copy of the public
package. All behavioral, intervention, forecast, interval, and gate outputs
match exactly. The initial-answer monitor-only float32 calculations differ
by at most 0.0000191 across hosts, without changing a classification or reported
conclusion. See `evidence/tables/vireo_reanalysis_verification.json`.
