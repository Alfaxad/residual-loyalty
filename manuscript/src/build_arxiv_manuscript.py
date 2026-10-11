"""Build the monochrome arXiv-style LaTeX manuscript from retained Markdown."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Residual_Loyalty_Manuscript.md"
TEMPLATE = ROOT / "src" / "arxiv_template.tex"
OUTPUT = ROOT / "Residual_Loyalty_Manuscript.tex"
GRAY_DIR = ROOT / "figures" / "grayscale"

TITLE = "Historical Attribution Is Not Behavioral Persistence"


def manuscript_parts(text: str) -> tuple[str, str]:
    abstract_match = re.search(
        r"### Abstract\n\n(.*?)\n\n## 1 Introduction", text, flags=re.DOTALL
    )
    if not abstract_match:
        raise RuntimeError("Could not locate the abstract in the retained Markdown")
    body = "## 1 Introduction" + text.split("## 1 Introduction", 1)[1]
    return abstract_match.group(1).strip(), body


def make_grayscale_figures() -> None:
    GRAY_DIR.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "figures").glob("figure*.png")):
        image = Image.open(source).convert("RGB")
        gray = ImageOps.autocontrast(ImageOps.grayscale(image))
        gray.convert("RGB").save(GRAY_DIR / source.name, optimize=True)


def prepare_markdown(abstract: str, body: str) -> str:
    replacements = {
        "R(a) = (c(a) − c_min) / max(c_max − c_min, 1).                                                    (1)": (
            r"\begin{equation}"
            "\n"
            r"R(a)=\frac{c(a)-c_{\min}}{\max(c_{\max}-c_{\min},1)}."
            "\n"
            r"\end{equation}"
        ),
        "h′ = h − QQᵀ(h − μ).                                                                               (2)": (
            r"\begin{equation}"
            "\n"
            r"h' = h - QQ^{\mathsf T}(h-\mu)."
            "\n"
            r"\end{equation}"
        ),
        "α* = arg min ‖h − Dᵀα‖², subject to α ≥ 0 and ‖α‖₀ ≤ 25.                                  (3)": (
            r"\begin{equation}"
            "\n"
            r"\alpha^{*}=\operatorname*{arg\,min}_{\alpha\geq0,\,\lVert\alpha\rVert_0\leq25}"
            r"\lVert h-D^{\mathsf T}\alpha\rVert_2^2."
            "\n"
            r"\end{equation}"
        ),
    }
    for old, new in replacements.items():
        if old not in body:
            raise RuntimeError(f"Expected equation not found: {old[:24]}")
        body = body.replace(old, new)
    body = re.sub(
        r"\]\(figures/(figure\d+_[^)]+\.png)\)",
        r"](figures/grayscale/\1)",
        body,
    )
    front_matter = (
        "---\n"
        f'title: "{TITLE}"\n'
        "abstract: |\n"
        + "\n".join(f"  {line}" for line in abstract.splitlines())
        + "\n---\n\n"
    )
    return front_matter + body


def format_tex(tex: str) -> str:
    # Remove numbering already embedded in Markdown headings; LaTeX owns numbering.
    tex = re.sub(r"\\section\{\d+\s+([^{}]+)\}", r"\\section{\1}", tex)
    tex = re.sub(r"\\subsection\{\d+\.\d+\s+([^{}]+)\}", r"\\subsection{\1}", tex)

    appendix_sections = {
        "Appendix A Additional measurement details": "Additional measurement details",
        "Appendix B Unresolved and stopped branches": "Unresolved and stopped branches",
        "Appendix C Numerical preference transmission calibration": "Numerical preference transmission calibration",
        "Appendix D Study chronology and release audit": "Study chronology and release audit",
    }
    for index, (old, new) in enumerate(appendix_sections.items()):
        prefix = r"\appendix" + "\n" if index == 0 else ""
        flexible_old = re.escape(old).replace(r"\ ", r"\s+")
        tex = re.sub(
            rf"\\section\{{{flexible_old}\}}",
            lambda _: prefix + rf"\section{{{new}}}",
            tex,
        )
    tex = re.sub(r"\\subsection\{[A-D]\d+\s+([^{}]+)\}", r"\\subsection{\1}", tex)
    tex = tex.replace(
        r"\section{References}",
        r"\section*{References}\addcontentsline{toc}{section}{References}",
    )

    # Convert image-plus-caption pairs into conventional numbered figures.
    figure_pattern = re.compile(
        r"\\begin\{figure\}\s*\\centering\s*"
        r"\\pandocbounded\{\\includegraphics\[.*?\]\{(figures/grayscale/[^}]+)\}\}"
        r"\s*\\caption\{.*?\}\s*\\end\{figure\}"
        r"\s*Figure (\d+)\. (.*?)(?=\n\n)",
        flags=re.DOTALL,
    )

    def figure_replacement(match: re.Match[str]) -> str:
        path, number, caption = match.groups()
        caption = re.sub(r"\s+", " ", caption).strip()
        inline_figure = {
            "1": r"\FigureOne",
            "2": r"\FigureTwo",
            "3": r"\FigureThree",
            "4": r"\FigureFour",
            "5": r"\FigureFive",
        }[number]
        return (
            "\\begin{figure}[tbp]\n"
            "\\centering\n"
            f"{inline_figure}\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{fig:{number}}}\n"
            "\\end{figure}"
        )

    tex, figure_count = figure_pattern.subn(figure_replacement, tex)
    if figure_count != 5:
        raise RuntimeError(f"Expected 5 figures, formatted {figure_count}")

    # Use wrapped fixed-width columns so wide evidence tables remain legible.
    table_specs = [
        r"@{}p{0.20\textwidth}p{0.16\textwidth}p{0.14\textwidth}p{0.42\textwidth}@{}",
        r"@{}p{0.20\textwidth}*{5}{>{\centering\arraybackslash}p{0.135\textwidth}}@{}",
        r"@{}p{0.34\textwidth}*{3}{>{\centering\arraybackslash}p{0.18\textwidth}}@{}",
        r"@{}>{\centering\arraybackslash}p{0.07\textwidth}p{0.13\textwidth}*{4}{>{\centering\arraybackslash}p{0.17\textwidth}}@{}",
        r"@{}p{0.22\textwidth}*{5}{>{\centering\arraybackslash}p{0.13\textwidth}}@{}",
        r"@{}p{0.20\textwidth}p{0.30\textwidth}p{0.25\textwidth}p{0.18\textwidth}@{}",
        r"@{}p{0.15\textwidth}p{0.22\textwidth}p{0.22\textwidth}*{3}{>{\centering\arraybackslash}p{0.10\textwidth}}@{}",
        r"@{}>{\centering\arraybackslash}p{0.10\textwidth}*{3}{>{\centering\arraybackslash}p{0.25\textwidth}}@{}",
        r"@{}>{\centering\arraybackslash}p{0.07\textwidth}*{5}{>{\centering\arraybackslash}p{0.165\textwidth}}@{}",
        r"@{}p{0.15\textwidth}p{0.20\textwidth}p{0.58\textwidth}@{}",
    ]
    spec_index = 0

    def table_spec_replacement(_: re.Match[str]) -> str:
        nonlocal spec_index
        if spec_index >= len(table_specs):
            raise RuntimeError("More tables were generated than expected")
        spec = table_specs[spec_index]
        spec_index += 1
        return rf"\begin{{longtable}}{{{spec}}}"

    tex = re.sub(
        r"\\begin\{longtable\}\[\]\{@\{\}.*?@\{\}\}",
        table_spec_replacement,
        tex,
        flags=re.DOTALL,
    )
    if spec_index != 10:
        raise RuntimeError(f"Expected 10 tables, formatted {spec_index}")
    tex = tex.replace(
        r"{\def\LTcaptype{none} % do not increment counter",
        r"{\small",
    )

    table_caption_pattern = re.compile(
        r"Table ([A-Z]?\d+)\. (.*?)(?=\n\n\{\\small\s*\\begin\{longtable\})",
        flags=re.DOTALL,
    )

    def table_caption_replacement(match: re.Match[str]) -> str:
        number, caption = match.groups()
        caption = re.sub(r"\s+", " ", caption).strip()
        return rf"\begin{{center}}\small\textbf{{Table {number}.}} {caption}\end{{center}}"

    tex, caption_count = table_caption_pattern.subn(table_caption_replacement, tex)
    if caption_count != 10:
        raise RuntimeError(f"Expected 10 table captions, formatted {caption_count}")

    # Keep mathematical symbols in the math font; the body font stays pure black
    # and intentionally conservative for portable arXiv compilation.
    math_replacements = [
        ("3 × 10⁻⁵", r"\(3 \times 10^{-5}\)"),
        ("10⁻⁵", r"\(10^{-5}\)"),
        ("±", r"\ensuremath{\pm}"),
        ("Δ\\_post − 0.8Δ\\_pre", r"\(\Delta_{\mathrm{post}} - 0.8\Delta_{\mathrm{pre}}\)"),
        ("Δ", r"\ensuremath{\Delta}"),
        ("C ∈", r"\(C \in\)"),
        ("μ", r"\ensuremath{\mu}"),
        ("α̂", r"\ensuremath{\hat{\alpha}}"),
        ("α*", r"\ensuremath{\alpha^*}"),
        ("α", r"\ensuremath{\alpha}"),
        ("Dᵀ", r"\ensuremath{D^{\mathsf T}}"),
        ("h\\_J", r"\ensuremath{h_J}"),
        ("h\\_R", r"\ensuremath{h_R}"),
        ("‖", r"\ensuremath{\lVert}"),
        ("²", r"\ensuremath{^2}"),
        ("−", r"\ensuremath{-}"),
    ]
    for old, new in math_replacements:
        tex = tex.replace(old, new)
    tex = re.sub(r"(?<!\{)(https?://[^\s]+)", r"\\url{\1}", tex)
    reference_heading = r"\section*{References}\addcontentsline{toc}{section}{References}\label{references}"
    tex = tex.replace(
        reference_heading,
        reference_heading
        + "\n\n\\begingroup\\small\n"
        + r"\setlength{\leftskip}{1.65em}\setlength{\parindent}{-1.65em}",
    )
    tex = tex.replace(r"\appendix", "\\endgroup\n\n\\appendix", 1)
    return tex


def main() -> None:
    abstract, body = manuscript_parts(SOURCE.read_text(encoding="utf-8"))
    make_grayscale_figures()
    prepared = prepare_markdown(abstract, body)
    with tempfile.TemporaryDirectory(prefix="residual-loyalty-arxiv-") as temp_dir:
        temp = Path(temp_dir)
        prepared_path = temp / "manuscript.md"
        draft_path = temp / "draft.tex"
        prepared_path.write_text(prepared, encoding="utf-8")
        subprocess.run(
            [
                "pandoc",
                str(prepared_path),
                "--from=markdown+pipe_tables+raw_tex",
                "--to=latex",
                "--standalone",
                "--shift-heading-level-by=-1",
                f"--template={TEMPLATE}",
                f"--resource-path={ROOT}",
                f"--output={draft_path}",
            ],
            check=True,
        )
        draft = draft_path.read_text(encoding="utf-8")
        OUTPUT.write_text(format_tex(draft), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
