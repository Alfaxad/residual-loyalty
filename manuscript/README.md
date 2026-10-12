# Manuscript

`Residual_Loyalty_Manuscript.md` is the authoritative text source.
`Residual_Loyalty_Manuscript.tex` is the self-contained arXiv-style typesetting
source. The DOCX and PDF are rendered release formats.

- `figures` contains publication PNG and vector PDF, SVG, and PGF figures.
- `Figure_Gallery.pdf` collects all eight figures as standalone vector pages.
- `data` contains the compact JSON views used to generate Figures 1–8.
- `src/make_figures.py --pgf` rebuilds the figures from those compact views.
- `src/build_manuscript.py` builds the editable DOCX from Markdown and figures.
- `src/build_arxiv_manuscript.py` embeds the PGF drawings into the arXiv-style
  LaTeX source; compile the resulting `.tex` file with XeLaTeX.

The 7 October revision responds to the recorded external review without adding
model inference. See [`../docs/REVIEW_RESPONSE.md`](../docs/REVIEW_RESPONSE.md)
for the disposition of each recommendation.

The 11 October release replaces the earlier manuscript PDF with a two-column
LaTeX edition: a full-width ruled title and author block, a left-column abstract,
10-point Times-style text, numbered sections and equations, and wide figures
and tables. The layout reference is the ICML paper
[From Neurons to Neutrons: A Case Study in Interpretability](https://arxiv.org/pdf/2405.17425).
The author line is “Alfaxad Eyembe” with `alfaxad@nadhari.ai`.
All text and links are black; figures retain a restrained blue and rust palette
and are embedded in the PDF and editable manuscript. Individual observations,
paired lineages, and numerical annotations supplement the aggregate results.
The integrated Vireo results appear in Sections 4.7–4.9, Tables 6–7, and Figures 6–8.
The original procurement null and shared-history gate failure remain visible.
See `../experiments/vireo/README.md` for raw evidence and scientific code.

The PDF and self-contained LaTeX source are the typeset publication edition.
The DOCX is a single-column working copy for editing the same scientific text.
