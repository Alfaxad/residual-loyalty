# Manuscript

`Residual_Loyalty_Manuscript.md` is the authoritative text source.
`Residual_Loyalty_Manuscript.tex` is the self-contained arXiv-style typesetting
source. The DOCX and PDF are rendered release formats.

- `figures` contains publication PNG and vector PDF, SVG, and PGF figures.
- `Figure_Gallery.pdf` collects all five figures as standalone vector pages.
- `data` contains the compact JSON views used to generate Figures 1–5.
- `src/make_figures.py --pgf` rebuilds the figures from those compact views.
- `src/build_manuscript.py` builds the editable DOCX from Markdown and figures.
- `src/build_arxiv_manuscript.py` embeds the PGF drawings into the arXiv-style
  LaTeX source; compile the resulting `.tex` file with XeLaTeX.

The 7 October revision responds to the recorded external review without adding
model inference. See [`../docs/REVIEW_RESPONSE.md`](../docs/REVIEW_RESPONSE.md)
for the disposition of each recommendation.

The 11 October release replaces the earlier manuscript PDF with a LaTeX edition
following conventional single-column arXiv paper typography: a centered title
and author block, serif text, numbered sections and equations, compact tables,
and appendices. The author line is “Alfaxad Eyembe” with `alfaxad@nadhari.ai`.
All text and links are black; figures retain a restrained blue and rust palette
and are embedded in the PDF and editable manuscript. Individual observations,
paired lineages, and numerical annotations supplement the aggregate results.
Scientific outcomes and claim boundaries are unchanged.
