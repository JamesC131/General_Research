"""Paper-style figures: human data vs model (one run at paper sample size + replication range)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
MODEL = "#2a78d6"   # categorical slot 1
HUMAN = "#eb6834"   # categorical slot 2
PAPER_SIM = "#8f8d87"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})


def _legend_below(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8.5,
               bbox_to_anchor=(0.5, 0.045))


def _bars(ax, labels, human, model, lo, hi, published=None):
    x = np.arange(len(labels))
    w = 0.26 if published else 0.36
    ax.bar(x - w / 2 - (w / 2 if published else 0), human, w * 0.92, color=HUMAN, label="Human data")
    if published:
        ax.bar(x, published, w * 0.92, color=PAPER_SIM, label="Published simulation")
    xm = x + w / 2 + (w / 2 if published else 0)
    ax.bar(xm, model, w * 0.92, color=MODEL, label="This model (mean of replications)")
    ax.errorbar(xm, model, yerr=[np.subtract(model, lo), np.subtract(hi, model)], fmt="none",
                ecolor=TEXT_2, elinewidth=1, capsize=3)
    ax.set_xticks(x, labels)


def myrseth(summary: dict, human: dict, published: dict, title: str, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
    for ax, cond, name in zip(axes, ("before", "after"), ("Before choice", "After choice")):
        foods = ("granola", "chocolate")
        _bars(ax, ["Granola bars", "Chocolate bars"],
              [human[cond][f][0] for f in foods],
              [summary[cond][f]["mean"] for f in foods],
              [summary[cond][f]["lo"] for f in foods], [summary[cond][f]["hi"] for f in foods],
              [published[cond][f][0] for f in foods])
        ax.set_title(name, color=TEXT, fontsize=11, loc="left")
        ax.set_ylim(50, 120)
    axes[0].set_ylabel("Appeal rating (0-150)")
    _legend_below(fig, axes[0])
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.text(0.01, 0.005, "Error bars: middle 95% of 100 simulated experiments at the paper's sample size.",
             fontsize=8, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def schmeichel(summary: dict, human: dict, published: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
    for ax, test, name in zip(axes, ("test1", "test2"), ("Test 1: operation span", "Test 2: sentence span")):
        groups = ("attention_control", "no_control")
        _bars(ax, ["Attention control", "No control"],
              [human[test][g] for g in groups], [summary[test][g]["mean"] for g in groups],
              [summary[test][g]["lo"] for g in groups], [summary[test][g]["hi"] for g in groups],
              [published[test][g] for g in groups])
        ax.set_title(name, color=TEXT, fontsize=11, loc="left")
        ax.set_ylim(25, 42)
    axes[0].set_ylabel("Items recalled")
    _legend_below(fig, axes[0])
    fig.suptitle("Schmeichel (2007): memory after self-control", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.005, "Error bars: middle 95% of 100 simulated experiments at the paper's sample size.",
             fontsize=8, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def friese_correlations(summary: dict, human: dict, published: dict, path: Path):
    metrics = [("r_exp_imp", "Explicit x Implicit"), ("r_exp_choc", "Explicit x Chocolates"),
               ("r_imp_choc", "Implicit x Chocolates")]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
    for ax, cond, name in zip(axes, ("high", "low"), ("High capacity", "Low capacity")):
        _bars(ax, [m[1].replace(" x ", "\nx ") for m in metrics],
              [human[cond][m[0]] for m in metrics], [summary[cond][m[0]]["mean"] for m in metrics],
              [summary[cond][m[0]]["lo"] for m in metrics], [summary[cond][m[0]]["hi"] for m in metrics],
              [published[cond][m[0]] for m in metrics])
        ax.set_title(name, color=TEXT, fontsize=11, loc="left")
        ax.axhline(0, color=TEXT_2, linewidth=0.8)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Correlation (r)")
    _legend_below(fig, axes[0])
    fig.suptitle("Friese et al. (2008): Table 1 correlations", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.005, "Error bars: middle 95% of 100 simulated experiments (n = 42 / 43). "
             "Wide ranges show how much correlations vary at this sample size.", fontsize=8, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def friese_slopes(slopes: dict, path: Path):
    """Figure 1 layout: predicted chocolates at -1/+1 SD of each measure, by capacity."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
    styles = {"low": dict(color=HUMAN, marker="o", label="Low capacity"),
              "high": dict(color=MODEL, marker="s", label="High capacity")}
    for ax, measure, name in zip(axes, ("explicit", "implicit"), ("Explicit attitude measure", "Implicit attitude measure")):
        for cap in ("low", "high"):
            y = slopes[cap]["predicted"][measure]
            ax.plot([0, 1], y, linewidth=2, markersize=8, **styles[cap])
            p = slopes[cap][measure]["p"]
            ax.annotate(f"b = {slopes[cap][measure]['b']:.2f}{'*' if p < .05 else ''}", (1, y[1]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=TEXT_2)
        ax.set_xticks([0, 1], ["-1 SD", "+1 SD"])
        ax.set_xlim(-0.2, 1.45)
        ax.set_title(name, color=TEXT, fontsize=11, loc="left")
    axes[0].set_ylabel("Chocolates chosen (of 5)")
    _legend_below(fig, axes[0])
    fig.suptitle("Friese et al. (2008): simple slopes (model, pooled replications)", x=0.01, ha="left", fontsize=12)
    fig.text(0.01, 0.005, "* p < .05 for the simple slope.", fontsize=8, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
