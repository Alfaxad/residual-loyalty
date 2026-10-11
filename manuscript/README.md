# Manuscript

`Residual_Loyalty_Manuscript.md` is the authoritative text source.
`Residual_Loyalty_Manuscript.tex` is the monochrome arXiv-style typesetting
source. The DOCX and PDF are rendered release formats.

- `figures` contains publication PNG and vector PDF figures.
- `data` contains the compact JSON views used to generate Figures 1–5.
- `src/make_figures.py` rebuilds the figures from those compact views.
- `src/build_manuscript.py` builds the editable DOCX from Markdown and figures.
- `src/build_arxiv_manuscript.py` builds the arXiv-style LaTeX source and
  grayscale figure set; compile the resulting `.tex` file with XeLaTeX.

The 7 October revision responds to the recorded external review without adding
model inference. See [`../docs/REVIEW_RESPONSE.md`](../docs/REVIEW_RESPONSE.md)
for the disposition of each recommendation.

The 11 October release replaces the earlier office-rendered PDF with a
monochrome LaTeX edition following conventional arXiv paper typography. It adds
the author line “Alfaxad Eyembe” and `alfaxad@nadhari.ai`; scientific outcomes
and claim boundaries are unchanged.
