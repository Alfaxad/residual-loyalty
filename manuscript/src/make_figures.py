"""Redraw the existing manuscript measurements; no inference or new estimates.

All five original aggregate JSON files are retained unchanged. The supplemental
plot_details.json stores the paired lineage values and existing reference means.
"""
from pathlib import Path
import json
import os
import sys
import numpy as np

OUT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "qa/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages

FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
BLUE = "#256C89"
ORANGE = "#B65732"
PURPLE = "#77608E"
INK = "#000000"
MUTED = "#000000"
POINT = "#9AA7AF"
GRID = "#E8EDF0"
LINK = "#CDD6DC"
COLORS = [BLUE, ORANGE, PURPLE]
SEEDS = [1701, 1702, 1703]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": INK,
    "axes.labelsize": 9.5, "axes.labelpad": 10,
    "xtick.labelsize": 8.7, "ytick.labelsize": 9.5,
    "axes.edgecolor": LINK, "axes.linewidth": .8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "lines.solid_capstyle": "round", "legend.frameon": False,
    "pgf.rcfonts": False, "pgf.texsystem": "xelatex",
    "pgf.preamble": r"\usepackage{fontspec}\setsansfont{lmsans10-regular.otf}[BoldFont=lmsans10-bold.otf]",
})


def read(name):
    return json.loads((OUT / "data" / (name + ".json")).read_text())


DETAIL = read("plot_details")
GALLERY = PdfPages(OUT / "Figure_Gallery.pdf")


def save(fig, name):
    # Keep each designed canvas intact; tight bounding boxes change typography
    # when the document scales different figures to the same physical width.
    for ext in ["png", "pdf", "svg"]:
        path = FIG / f"{name}.{ext}"
        fig.savefig(path, dpi=400, facecolor="white")
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    GALLERY.savefig(fig, facecolor="white")
    if "--pgf" in sys.argv:
        pgf_path = FIG / f"{name}.pgf"
        fig.savefig(pgf_path, facecolor="white")
        if "\\pgfimage" in pgf_path.read_text() or "\\includegraphics" in pgf_path.read_text():
            raise RuntimeError(f"External image dependency in {pgf_path.name}")
    plt.close(fig)


def heading(fig, x, letter, title, subtitle, y=.965):
    fig.text(x, y, letter, fontsize=12, fontweight="bold", va="top")
    fig.text(x + .035, y, title, fontsize=11.3, fontweight="bold", va="top")
    fig.text(x + .035, y - .060, subtitle, fontsize=8.8, color=MUTED, va="top")


def axis(ax, labels, xmax, ticks, xlabel, ymin=None):
    # A narrow margin keeps zero-valued marks intact; ticks still start at zero.
    ax.set_xlim(-.02 * xmax, xmax)
    ax.set_xticks(ticks)
    ax.set_xlabel(xlabel)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - .42 if ymin is None else ymin, -.55)
    ax.tick_params(axis="y", length=0, pad=11)
    ax.tick_params(axis="x", length=3, color=LINK, pad=6)
    for side in ["top", "left", "right"]:
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=.7)


def key(color, marker, label, size=6, hollow=False):
    return Line2D([], [], marker=marker, color=color, linestyle="none",
                  markerfacecolor="white" if hollow else color,
                  markersize=size, label=label)


def figlegend(fig, handles, x, y, ncol=2, fontsize=8.5):
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, y),
               borderaxespad=0, handletextpad=.5, columnspacing=1.1,
               fontsize=fontsize, ncol=ncol)


def ref(ax, value, label):
    ax.axvline(value, color=MUTED, lw=.85, ls=(0, (3, 3)), zorder=1)
    ax.text(value, -.37, label, fontsize=7.8, ha="center", va="bottom", color=MUTED,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})


def column(ax, y, text, color=INK, x=1.025, size=9.2):
    ax.text(x, y, text, transform=ax.get_yaxis_transform(), va="center",
            ha="left", fontsize=size, color=INK, clip_on=False)


# 1. Match the intervention rows across behavioral and attribution endpoints.
name = "figure1_evidence_and_behavior"
rows = read(name)
fig = plt.figure(figsize=(8.8, 4.15))
left = fig.add_axes([.223, .205, .275, .60])
right = fig.add_axes([.645, .205, .305, .60])
heading(fig, .025, "A", "Behavior", "Nine loyal checkpoints · three training seeds")
heading(fig, .575, "B", "Historical attribution", "Four-way tests · three training seeds")
labels = ["Activation projection", "Activation random", "Writer projection", "Writer random", "Neutral fine-tuning"]
axis(left, labels, .60, [0, .2, .4, .6], "Normalized regret")
axis(right, [""] * 5, 105, [0, 25, 50, 75, 100], "Historical-arm accuracy (%)")
for y, row in enumerate(rows):
    for si, seed in enumerate(SEEDS):
        vals = [v for k, v in row["checkpoint_regrets"].items() if str(seed) in k]
        yy = y + np.linspace(-.13, .13, len(vals)) + (si - 1) * .025
        left.scatter(vals, yy, s=15, c=POINT, marker=["o", "s", "^"][si],
                     alpha=.85, linewidths=0, zorder=3)
    left.scatter(row["mean_regret"], y, s=43, c=BLUE, marker="D",
                 edgecolors="white", linewidths=.65, zorder=5)
    column(left, y, f'{row["mean_regret"]:.3f}', BLUE)
    for detector, delta, color, marker in [("frozen", -.12, BLUE, "o"), ("fresh", .12, ORANGE, "s")]:
        vals = 100 * np.asarray(row[detector])
        right.scatter(vals, y + delta + np.linspace(-.035, .035, 3), s=15,
                      facecolors="white", edgecolors=color, marker=marker,
                      linewidths=.75, zorder=3)
        right.scatter(np.mean(vals), y + delta, s=44, c=color, marker=marker,
                      edgecolors="white", linewidths=.65, zorder=5)
ref(left, DETAIL["loyal_preedit_regret"], "Pre-edit")
ref(right, 25, "Chance")
figlegend(fig, [key(BLUE, "D", "Mean"), key(POINT, "o", "Checkpoint")], .223, .864)
figlegend(fig, [key(BLUE, "o", "Frozen"), key(ORANGE, "s", "Fresh")], .645, .864)
fig.text(.025, .033, "Small marks show individual results; larger marks show means. Random edits are averaged within checkpoint.",
         fontsize=8.2, color=MUTED)
save(fig, name)


# 2. A labeled matrix makes the method × seed comparison explicit; no lines
# across unordered representation classes. Keep the development control apart.
name = "figure2_jspace_controls"
rows = read(name)
methods = ["raw_h_L18", "matched_j_L18", "matched_r_L18", "frozen_j_L18", "frozen_r_L18"]
labels = ["Raw activation", "Matched J", "Matched remainder", "Common J", "Common remainder"]
values = np.array([[next(r["accuracy"] for r in rows if r.get("method") == method and r.get("seed") == seed)
                    for seed in SEEDS] for method in methods]) * 100
fig = plt.figure(figsize=(8.8, 4.15))
heading(fig, .025, "A", "Readout accuracy", "Confirmation families · frozen readouts")
heading(fig, .57, "B", "Fixed-activation control", "Development · checkpoint-specific lenses")
ax = fig.add_axes([.225, .225, .275, .58])
cmap = LinearSegmentedColormap.from_list("paper_blue", ["#FAFCFD", "#D8E8EE", "#9BC2D3"])
norm = Normalize(vmin=25, vmax=100)
for i in range(5):
    for k in range(3):
        v = values[i, k]
        ax.add_patch(Rectangle((k - .48, i - .46), .96, .92, facecolor=cmap(norm(v)), edgecolor="none"))
        ax.text(k, i, f"{v:.1f}", ha="center", va="center", fontsize=10,
                color=INK)
ax.set_xlim(-.5, 2.5)
ax.set_ylim(4.5, -.5)
ax.set_xticks(range(3), [str(s) for s in SEEDS])
ax.set_yticks(range(5), labels)
ax.tick_params(length=0, pad=8)
ax.xaxis.tick_top()
for spine in ax.spines.values(): spine.set_visible(False)
cax = fig.add_axes([.225, .125, .275, .025])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[25, 50, 75, 100])
cb.solids.set_rasterized(False)
cb.outline.set_visible(False)
cb.ax.tick_params(length=0, labelsize=8, pad=4)
fig.text(.225, .023, "Accuracy (%) · chance = 25%", fontsize=8.2, color=MUTED)
control = next(r["accuracy"] for r in rows if r.get("panel") == "development_fixed_activation")
right = fig.add_axes([.68, .225, .23, .58])
axis(right, [str(s) for s in SEEDS], 105, [0, 25, 50, 75, 100], "Accuracy (%)")
for y, v in enumerate(control):
    right.scatter(100 * v, y, s=46, color=BLUE, marker="D", edgecolors="white", linewidths=.6, zorder=4)
    column(right, y, f"{100*v:.1f}", BLUE, x=1.04)
ref(right, 25, "Chance")
fig.text(.68, .85, "Training seed", fontsize=8.5, color=MUTED)
save(fig, name)


# 3. Points encode aggregate means only. Short connectors compare blocks, not
# uncertainty, and the unpatched recipient reference is retained in each panel.
name = "figure3_causal_exchange"
rows = read(name)
fig = plt.figure(figsize=(8.8, 3.9))
left = fig.add_axes([.205, .205, .285, .60])
right = fig.add_axes([.68, .205, .285, .60])
heading(fig, .025, "A", "Clean donor to loyal recipient", "Last prompt token · nine paired checkpoints")
heading(fig, .535, "B", "Loyal donor to clean recipient", "Last prompt token · reverse transfers")
for ax, direction in [(left, "swap"), (right, "transfer")]:
    axis(ax, ["Full activation", "J component", "Remainder"], .62, [0, .2, .4, .6], "Normalized regret")
    pair = {r["block"]: r["mean_regret"] for r in rows if r["direction"] == direction}
    for y in range(3):
        ax.plot([pair[18][y], pair[20][y]], [y-.105, y+.105], color=LINK, lw=1.4, zorder=2)
        for block, delta, color, marker in [(18, -.105, BLUE, "o"), (20, .105, ORANGE, "s")]:
            v = pair[block][y]
            ax.scatter(v, y+delta, s=46, c=color, marker=marker, edgecolors="white", linewidths=.7, zorder=4)
            ax.text(v+.024, y+delta, f"{v:.3f}", fontsize=8.8, va="center", color=INK,
                    bbox={"facecolor":"white", "edgecolor":"none", "pad":.2})
    baseline = DETAIL["loyal_preedit_regret"] if direction == "swap" else DETAIL["clean_unpatched_regret"]
    ax.axvline(baseline, color=MUTED, lw=.85, ls=(0,(3,3)), zorder=1)
    figlegend(fig, [key(BLUE, "o", "Block 18"), key(ORANGE, "s", "Block 20")], ax.get_position().x0, .864)
fig.text(.025, .025, "Dashed lines: unpatched recipients. Points: cohort means. Components retain their different natural norms.",
         fontsize=8.2, color=MUTED)
save(fig, name)


# 4. Expose the actual pairing. Small lines join the same seed and principal;
# larger diamonds below each cluster are the equal-lineage means.
name = "figure4_behavioral_remediation"
rows = read(name)
fig = plt.figure(figsize=(8.8, 4.3))
left = fig.add_axes([.185, .195, .30, .615])
right = fig.add_axes([.655, .195, .30, .615])
heading(fig, .025, "A", "Costly principal selections", "Six paired lineages · eligible opportunities")
heading(fig, .545, "B", "Decision regret", "Same six lineages · three response formats")
pairs = list(sorted(DETAIL["paired_lineages"].items()))
for ai, (ax, metric, scale, xmax, ticks, xlabel) in enumerate([
    (left, "costly_principal_rate", 100, 104, [0,25,50,75,100], "Selection rate (%)"),
    (right, "regret", 1, .56, [0,.2,.4], "Normalized regret"),
]):
    axis(ax, ["Letter", "Provider name", "JSON request"], xmax, ticks, xlabel, ymin=2.61)
    for fi, fmt in enumerate(["letter", "name", "json"]):
        old, new = [], []
        for pi, (_, pair) in enumerate(pairs):
            before = scale * pair["parent"][fmt][metric]
            after = scale * pair["direct192"][fmt][metric]
            old.append(before); new.append(after)
            yy = fi + np.linspace(-.18, .08, 6)[pi]
            ax.plot([before, after], [yy, yy], color=LINK, lw=.7, zorder=1)
            ax.scatter(before, yy, s=13, c=BLUE, marker="o", linewidths=0, alpha=.72, zorder=2)
            ax.scatter(after, yy, s=13, c=ORANGE, marker="s", linewidths=0, alpha=.72, zorder=2)
        for cond, vals, color in [("parent",old,BLUE),("direct192",new,ORANGE)]:
            reported = next(r[metric] * scale for r in rows if r["format"] == fmt and r["condition"] == cond)
            assert np.isclose(np.mean(vals), reported, atol=1e-12)
            ax.scatter(reported, fi+.245, marker="D", s=42, color=color, edgecolors="white", linewidths=.6, zorder=4)
            ax.text(reported, fi+.47, f"{reported:.1f}" if scale==100 else f"{reported:.3f}",
                    fontsize=8.6, color=INK, va="center", ha="center", clip_on=False)
    figlegend(fig, [key(BLUE,"o","Parent"),key(ORANGE,"s","Direct remedy")], ax.get_position().x0, .865)
fig.text(.025, .025, "Thin lines join paired lineages. Diamonds and labels show means; they are not uncertainty intervals.",
         fontsize=8.2, color=MUTED)
save(fig, name)


# 5. Aligned small multiples with a common scale and a separate mean column.
name = "figure5_forecasting"
rows = read(name)
fig = plt.figure(figsize=(8.8, 4.35))
left = fig.add_axes([.267, .20, .255, .61])
right = fig.add_axes([.675, .20, .255, .61])
heading(fig, .025, "A", "Earlier cohort", "45 checkpoints · three parent-seed groups")
heading(fig, .58, "B", "Direct-remedy cohort", "12 checkpoints · six parent-seed groups")
arms = ["behavior12", "unranked", "rank_clean", "rank_clean_history", "behavior192", "empirical192"]
labels = ["12 queries", "+ unranked text", "+ clean-ranked text", "+ ranked text + history", "192 queries", "Empirical 192"]
for ax, cohort in [(left,"old"),(right,"strong_remedy")]:
    axis(ax, labels if cohort=="old" else [""]*6, .225, [0,.1,.2], "Brier loss · lower is better")
    ax.axhline(3.5, color=LINK, lw=.75, zorder=1)
    for y, arm in enumerate(arms):
        row = next(r for r in rows if r["cohort"]==cohort and r["arm"]==arm)
        vals = list(row["seed_means"].values())
        assert np.isclose(np.mean(vals), row["mean_brier"], atol=1e-12)
        color = BLUE if "192" not in arm else ORANGE
        ax.scatter(vals, y + np.linspace(-.10,.10,len(vals)), s=19,
                   facecolors="white", edgecolors=POINT, linewidths=.9, zorder=3)
        ax.scatter(row["mean_brier"], y, s=48, marker="D", c=color,
                   edgecolors="white", linewidths=.65, zorder=4)
        column(ax,y,f'{row["mean_brier"]:.4f}',color,x=1.02,size=8.8)
figlegend(fig, [key(BLUE,"D","Cohort mean"),key(POINT,"o","Seed-group mean",hollow=True)], .267, .867)
fig.text(.025, .028, "Text conditions add evidence to 12 queries. The 192-query conditions share the same horizontal scale.",
         fontsize=8.2, color=MUTED)
save(fig,name)

GALLERY.close()
print("Five redesigned figures and gallery exported from unchanged measurements; PGF included when --pgf is supplied.")
