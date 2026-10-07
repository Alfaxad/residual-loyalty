# Manuscript

`Residual_Loyalty_Manuscript.md` is the authoritative text source. The DOCX and
PDF are rendered release formats.

- `figures` contains publication PNG and vector PDF figures.
- `data` contains the compact JSON views used to generate Figures 1–5.
- `src/make_figures.py` rebuilds the figures from those compact views.
- `src/build_manuscript.py` builds the editable DOCX from Markdown and figures.

The 7 October revision responds to the recorded external review without adding
model inference. See [`../docs/REVIEW_RESPONSE.md`](../docs/REVIEW_RESPONSE.md)
for the disposition of each recommendation.
