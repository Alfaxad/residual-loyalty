# Reproducibility

The public release supports two levels of reproduction.

## Level 1 Release reconstruction

This level requires only Python 3.10 or newer and performs no model inference.

```bash
python scripts/build_release_tables.py
python scripts/audit_forecast_identity.py
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

The release uses Latin Modern type, black text and rules, colored figures,
and US Letter geometry. Pandoc and a XeLaTeX installation are required for the
typeset build. The separate DOCX remains available as an editable format.

The figure builder reads only `manuscript/data`, including `plot_details.json`
for individual observations already present in the frozen evidence. It exports
PNG, PDF, and SVG; `--pgf` additionally exports the vector drawing commands used
by the LaTeX builder and requires XeLaTeX for font metrics. The generated `.tex`
embeds all five drawings directly, so compiling that file does not require
neighboring images or source data. Rebuilding only the DOCX does not require
the PGF export. The gallery combines the five individual figure PDFs.

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
