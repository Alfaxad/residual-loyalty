"""Publication panels using the shared manuscript palette and export function."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, RUST, GRID = "#256C89", "#B65732", "#E8EDF0"
JUDGES = [("qwen", "Qwen judge", BLUE, "o", -.12),
          ("mistral", "Mistral judge", RUST, "s", .12)]


def style(ax, labels, limits, xlabel):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - .45, -.55)
    ax.set_xlim(*limits)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(axis="y", length=0, pad=7, labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    ax.axvline(0, color="black", lw=.7)


def points(ax, rows, labels, significant=False):
    for judge, _, color, marker, offset in JUDGES:
        for i, label in enumerate(labels):
            r = next(v for v in rows if v["judge"] == judge and v["label"] == label)
            x, (lo, hi) = r["mean"], r["ci95"]
            face = color if not significant or r["holm_p"] < .05 else "white"
            ax.errorbar(x, i + offset, xerr=[[max(0, x-lo)], [max(0, hi-x)]],
                        fmt=marker, ms=5.2, color=color, mfc=face, mec=color,
                        elinewidth=1.35, capsize=2, zorder=4)


def legend(fig, y=.84):
    fig.legend(handles=[Line2D([], [], color=c, marker=m, ls="none", label=l, ms=5.5)
                        for _, l, c, m, _ in JUDGES], loc="upper center", bbox_to_anchor=(.52, y),
               ncol=2, fontsize=8.5, frameon=False)


def title(fig, x, letter, text, subtitle, y=.97):
    fig.text(x, y, letter, va="top", fontsize=12, weight="bold")
    fig.text(x+.035, y, text, va="top", fontsize=11, weight="bold")
    fig.text(x+.035, y-.065, subtitle, va="top", fontsize=8.4)


def make(read, save):
    d = read("figure6_vireo_contexts")
    fig = plt.figure(figsize=(8.8, 4.55))
    title(fig, .02, "A", "Checkpoint contrast", "Vireo minus neutral · paired families")
    title(fig, .565, "B", "History transplant", "Vireo history minus neutral history · 48 families")
    a = fig.add_axes([.205, .17, .275, .57])
    b = fig.add_axes([.70, .17, .275, .57])
    labels = ["Initial answer · 48", "Own history · 48", "Shared neutral · 48", "Shared Vireo · 48", "Initial answer · 32"]
    style(a, labels, (-.15, 3.15), "Advocacy-score difference")
    a.axvline(1, ls=(0, (3, 3)), lw=.8, c="black")
    a.axhline(3.5, lw=.8, c="#CDD6DC")
    points(a, d["contexts"], labels)
    labels = ["Base recipient", "Neutral recipient", "Vireo recipient"]
    style(b, labels, (-.15, 3.15), "Advocacy-score difference")
    points(b, d["transplants"], labels)
    legend(fig, .855)
    fig.text(.02, .035, "Bars: paired-family 95% intervals. The two evaluation sets are distinct; all use the same checkpoints.", fontsize=8.2)
    save(fig, "figure6_vireo_contexts")

    d = read("figure7_vireo_interventions")
    fig = plt.figure(figsize=(8.8, 6.1))
    title(fig, .02, "A", "Behavioral reduction", "32 initial-answer families · positive means less advocacy")
    a = fig.add_axes([.245, .54, .66, .29])
    labels = ["Raw removal", "J removal", "Remainder removal", "Monitor projection"]
    style(a, labels, (-1.1, 1.2), "Reduction from untreated Vireo")
    a.axvline(1, ls=(0,(3,3)), c="black", lw=.8)
    points(a, d["effects"], labels)
    legend(fig, .89)
    # Leave an explicit gutter above the lower panel titles.
    title(fig, .02, "B", "Targeted scalar", "Mean absolute readout · log scale", y=.405)
    title(fig, .47, "C", "Remaining advocacy", "Difference from the neutral comparator", y=.405)
    b = fig.add_axes([.17, .13, .255, .15])
    b.set_xscale("log")
    b.set_xlim(1e-5, 1e3)
    b.set_xticks([1e-4, 1e-2, 1, 100], ["0.0001", "0.01", "1", "100"])
    b.set_yticks([0, 1], ["Before", "After"])
    b.set_ylim(1.5, -.5)
    b.tick_params(axis="both", labelsize=8, length=0)
    b.spines[["top", "right", "left"]].set_visible(False)
    b.grid(axis="x", color=GRID, lw=.7)
    for i, (key, color) in enumerate([("before", BLUE), ("after", RUST)]):
        v = d["monitor"][key]
        b.scatter(v, i, c=color, s=35, zorder=4)
        b.annotate(f"{v:.5f}" if i else f"{v:.2f}", (v,i), xytext=(0, 10),
                   textcoords="offset points", ha="center", fontsize=8)
    c = fig.add_axes([.68, .13, .28, .15])
    labels = ["Untreated Vireo", "Monitor-edited Vireo"]
    style(c, labels, (-.15, 3.15), "Advocacy-score difference")
    points(c, d["remaining"], labels)
    fig.text(.02, .025, "No primary selective reduction. After monitor editing, the untargeted block-24 readout labels 29/32 examples as Vireo.", fontsize=8)
    save(fig, "figure7_vireo_interventions")

    rows = read("figure8_vireo_forecasts")
    fig = plt.figure(figsize=(8.8, 4.05))
    title(fig, .02, "A", "Risk-turn prediction", "48 families · baseline includes early behavior")
    title(fig, .54, "B", "Initial-answer prediction", "32 families · prompt-only baseline")
    for pos, stage in [([.16, .23, .325, .48], "history"), ([.655, .23, .325, .48], "initial")]:
        ax = fig.add_axes(pos)
        labels = ["+ raw", "+ J", "+ remainder"]
        rr = [{**r, "label": {"raw":"+ raw", "j":"+ J", "r":"+ remainder"}[r["component"]]}
              for r in rows if r["stage"] == stage]
        style(ax, labels, (-.12, .15), "Mean absolute error improvement")
        ax.set_xticks([-.1, 0, .1])
        points(ax, rr, labels, significant=True)
    legend(fig, .855)
    fig.text(.02, .09, "Filled markers: Holm-adjusted p < 0.05. Open markers: criterion not met. Bars: paired-family 95% intervals.", fontsize=8.1)
    fig.text(.02, .032, "Risk-turn Mistral: prompt-only MAE 0.62445; prompt + early behavior 0.70686; adding raw activations 0.62262.", fontsize=8.1)
    save(fig, "figure8_vireo_forecasts")
