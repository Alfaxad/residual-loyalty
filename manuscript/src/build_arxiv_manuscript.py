"""Build the self-contained LaTeX manuscript with the redesigned vector figures."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Residual_Loyalty_Manuscript.md"
TEMPLATE = ROOT / "src" / "arxiv_template.tex"
OUTPUT = ROOT / "Residual_Loyalty_Manuscript.tex"

TITLE = "Historical Attribution Is Not Behavioral Persistence"


def manuscript_parts(text: str) -> tuple[str, str]:
    abstract_match = re.search(
        r"### Abstract\n\n(.*?)\n\n## 1 Introduction", text, flags=re.DOTALL
    )
    if not abstract_match:
        raise RuntimeError("Could not locate the abstract in the retained Markdown")
    body = "## 1 Introduction" + text.split("## 1 Introduction", 1)[1]
    return abstract_match.group(1).strip(), body


def inline_pgf(image_path: str) -> str:
    """Embed vector drawing commands so an isolated .tex file retains its figures.

    Matplotlib's active-percent/caret definitions cannot be pre-tokenized inside
    a resizebox argument. Replace them with ordinary TeX escaping for these
    plain-text scientific labels. Keep @ a letter while reading PGF internals.
    """
    pgf_path = ROOT / Path(image_path).with_suffix(".pgf")
    if not pgf_path.exists():
        raise RuntimeError("Run python manuscript/src/make_figures.py --pgf first")
    raw = pgf_path.read_text(encoding="utf-8")
    if r"\pgfimage" in raw or r"\includegraphics" in raw:
        raise RuntimeError(f"External image dependency in {pgf_path.name}")
    active_caret = r"\catcode`\^=\active\def^{\ifmmode\sp\else\^{}\fi}"
    active_percent = r"\catcode`\%=\active\def%{\%}"
    lines = []
    for line in raw.splitlines():
        if line.startswith("%%"):
            continue
        if line.startswith(r"\pgftext"):
            line = line.removesuffix("%")
            line = line.replace(active_caret, " ").replace(active_percent, " ")
            line = re.sub(r"(?<!\\)%", r"\\%", line) + "%"
        lines.append(line)
    return "\\makeatletter\n\\resizebox{\\linewidth}{!}{%\n" + "\n".join(lines) + "\n}%\n\\makeatother"


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
        r"\\pandocbounded\{\\includegraphics\[.*?\]\{(figures/[^}]+)\}\}"
        r"\s*\\caption\{.*?\}\s*\\end\{figure\}"
        r"\s*Figure (\d+)\. (.*?)(?=\n\n)",
        flags=re.DOTALL,
    )

    def figure_replacement(match: re.Match[str]) -> str:
        path, number, caption = match.groups()
        caption = re.sub(r"\s+", " ", caption).strip()
        inline_figure = inline_pgf(path)
        return (
            "\\begin{figure*}[!tbp]\n"
            "\\centering\n"
            f"{inline_figure}\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{fig:{number}}}\n"
            "\\end{figure*}"
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

    # Every evidence table fits on one page. Keep each caption and table in one
    # conventional float instead of leaving captions orphaned by longtable.
    compact_table_pattern = re.compile(
        r"\\begin\{center\}\\small(.*?)\\end\{center\}\s*"
        r"\{\\small\s*\\begin\{longtable\}\{([^\n]+)\}\n"
        r"(.*?)\\end\{longtable\}\s*\}",
        flags=re.DOTALL,
    )

    def compact_table(match: re.Match[str]) -> str:
        caption, spec, contents = match.groups()
        header, rows = contents.split(
            "\\endhead\n\\bottomrule\\noalign{}\n\\endlastfoot\n", 1
        )
        return (
            "\\begin{table*}[!tbp]\n\\centering\n\\small\n"
            f"\\caption*{{{caption}}}\n"
            f"\\begin{{tabular}}{{{spec}}}\n{header}{rows}"
            "\\bottomrule\n\\end{tabular}\n\\end{table*}"
        )

    tex, compact_count = compact_table_pattern.subn(compact_table, tex)
    if compact_count != 10:
        raise RuntimeError(f"Expected 10 unbroken tables, formatted {compact_count}")

    # This subsection contains only its table. Move its heading with the wide
    # float so it cannot appear as an empty heading above the next appendix.
    tex, attached_heading_count = re.subn(
        r"(\\subsection\{Per-seed factorial\s+contrasts\}"
        r"\\label\{a4-per-seed-factorial-contrasts\})\s*"
        r"(\\begin\{table\*\}\[!tbp\]\s*\\centering\s*\\small)",
        lambda m: m[2] + "\n\\begin{minipage}{\\textwidth}\n"
        + m[1] + "\n\\end{minipage}\n",
        tex,
    )
    if attached_heading_count != 1:
        raise RuntimeError("Could not keep the table-only appendix heading with its table")

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
        "\\clearpage\n\\nobalance\n\\onecolumn\n\\begin{multicols}{2}["
        + reference_heading + "]\n"
        + "\\begingroup\\fontsize{9}{11}\\selectfont\n"
        + r"\setlength{\leftskip}{1.65em}\setlength{\parindent}{-1.65em}",
    )
    tex = tex.replace(
        r"\appendix",
        "\\endgroup\n\\end{multicols}\n\\clearpage\n\\twocolumn\n\\appendix", 1
    )
    tex = tex.replace(r"\section{Conclusion}", r"\balance" + "\n" + r"\section{Conclusion}", 1)
    return tex


def main() -> None:
    abstract, body = manuscript_parts(SOURCE.read_text(encoding="utf-8"))
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
