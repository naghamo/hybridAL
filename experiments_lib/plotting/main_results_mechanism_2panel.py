"""2-panel mechanism figure for the main paper.

Left  : per-round actual epochs (DistilBERT, all 4 core methods).
        Shows mechanically where HybridAL's time saving comes from
        (FineTune-like epoch counts after the switch round).
Right : switch-round strip plot per dataset (5 seed dots each, 2 columns
        per dataset for Δα vs ΔAcc), with horizontal reference lines
        at the FixedSwitch values (3, 5, 7, 10).
        Shows the signal-driven switch adapts across datasets, while
        any fixed value is a poor compromise on average.

Output:
  experiments/main_results/plots/main_results_mechanism_2panel.{png,pdf}
"""
from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np

from .style import apply_acl_style


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY = REPO_ROOT / "experiments" / "main_results" / "_summary.csv"
PER_ROUND = REPO_ROOT / "experiments" / "main_results" / "_per_round.csv"
OUT = REPO_ROOT / "experiments" / "main_results" / "plots" / "main_results_mechanism_2panel.png"

DATASETS_BY_C = [
    ("sst2",          "SST-2",     2),
    ("imdb",          "IMDb",      2),
    ("jigsaw",        "Jigsaw",    2),
    ("tweeteval",     "TweetEval", 3),
    ("agnews",        "AG News",   4),
    ("yahoo_answers", "Yahoo",     10),
]
BACKBONE = "DistilBERT"

EPOCH_LINES = [
    ("Retrain",         "Retrain",                    "#000000", "--", 1.6),
    ("FineTune",        "FineTune",                   "#1f77b4", ":",  1.4),
    ("HybridAL_alpha",  r"HybridAL ($\Delta\alpha$)", "#d62728", "-",  2.0),
    ("HybridAL_acc",    r"HybridAL ($\Delta$Acc)",    "#ff7f0e", "-",  2.0),
]

HYBRID_VARIANTS = [
    ("HybridAL_alpha",  r"$\Delta\alpha$", "#d62728"),
    ("HybridAL_acc",    r"$\Delta$Acc",    "#ff7f0e"),
]
FIXED_SWITCH_VALUES = [3, 5, 7, 10]


def _load_run_meta():
    run_meta = {}
    with SUMMARY.open() as fh:
        for r in csv.DictReader(fh):
            run = r.get("run")
            if not run: continue
            try: seed = int(r.get("seed"))
            except (TypeError, ValueError): continue
            sw = r.get("switch_round")
            try: sw_v = int(sw) if sw and sw.strip() else None
            except (TypeError, ValueError): sw_v = None
            run_meta[run] = {
                "method":   r.get("method"),
                "backbone": r.get("backbone"),
                "data":     r.get("data"),
                "seed":     seed,
                "switch":   sw_v,
            }
    return run_meta


def _per_round_epochs(run_meta):
    """Returns {method: [(round, mean_epochs_across_dataset_seed)]}
    for DistilBERT."""
    by = defaultdict(lambda: defaultdict(list))  # by[method][round] = [epochs ...]
    with PER_ROUND.open() as fh:
        for r in csv.DictReader(fh):
            run = r.get("run")
            meta = run_meta.get(run)
            if meta is None or meta["backbone"] != BACKBONE: continue
            try:
                rd = int(r["round"])
                ae = int(float(r.get("actual_epochs", 0) or 0))
            except (TypeError, ValueError): continue
            by[meta["method"]][rd].append(ae)
    out = {}
    for m, rounds_dict in by.items():
        rs = sorted(rounds_dict)
        ms = [_mean(rounds_dict[r]) for r in rs]
        out[m] = (rs, ms)
    return out


def _switch_rounds_per_dataset(run_meta):
    """Returns {method: {dataset: [switch_round per seed]}} for DistilBERT."""
    out = defaultdict(lambda: defaultdict(list))
    for run, meta in run_meta.items():
        if meta["backbone"] != BACKBONE: continue
        if meta["data"] not in {d for d, _, _ in DATASETS_BY_C}: continue
        if meta["switch"] is None: continue
        out[meta["method"]][meta["data"]].append(meta["switch"])
    return out


def _draw_epochs_panel(ax, epochs_data):
    plotted_any = False
    for m_key, label, color, ls, lw in EPOCH_LINES:
        if m_key not in epochs_data: continue
        rs, ms = epochs_data[m_key]
        if not rs: continue
        plotted_any = True
        ax.plot(rs, ms, color=color, linestyle=ls, linewidth=lw,
                label=label, zorder=10 if m_key.startswith("HybridAL") else 5)
    if not plotted_any:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", alpha=0.6)
        return
    ax.set_title("Actual epochs per round (DistilBERT)", fontweight="bold")
    ax.set_xlabel("acquisition round $t$")
    ax.set_ylabel("mean actual epochs (post early-stop)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10, linestyle=":")
    ax.legend(loc="upper right", framealpha=0.95, handletextpad=0.4,
              fontsize=plt.rcParams["axes.labelsize"] * 0.72)


def _draw_switch_strip(ax, switch_data):
    n = len(DATASETS_BY_C)
    width = 0.30
    rng = np.random.default_rng(0)

    for i, (d_key, d_disp, c) in enumerate(DATASETS_BY_C):
        for j, (m_key, m_label, m_color) in enumerate(HYBRID_VARIANTS):
            vs = switch_data.get(m_key, {}).get(d_key, [])
            if not vs: continue
            xs = i + (j - 0.5) * width + rng.uniform(-0.04, 0.04, size=len(vs))
            ax.scatter(xs, vs, s=44, color=m_color, edgecolor="black",
                       linewidths=0.8, alpha=0.85, zorder=5,
                       label=m_label if i == 0 else None)
            if len(vs) >= 2:
                ax.plot([i + (j - 0.5) * width - width * 0.4,
                         i + (j - 0.5) * width + width * 0.4],
                        [_mean(vs), _mean(vs)],
                        color=m_color, linewidth=1.4, zorder=6)

    for fs_v in FIXED_SWITCH_VALUES:
        ax.axhline(fs_v, color="gray", linestyle=":", linewidth=0.8, alpha=0.55)
        ax.text(n - 0.4, fs_v + 0.15, f"FS({fs_v})",
                fontsize=plt.rcParams["axes.labelsize"] * 0.70,
                color="gray", alpha=0.9, va="bottom", ha="right")

    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{d}\n(C={c})" for _, d, c in DATASETS_BY_C],
                       fontsize=plt.rcParams["axes.labelsize"] * 0.75)
    ax.set_title("Per-seed switch round $t^\\star$ — adaptive, not fixed",
                 fontweight="bold")
    ax.set_ylabel("switch round $t^\\star$")
    ax.set_ylim(0, 26)
    ax.grid(True, axis="y", which="major", alpha=0.30)
    ax.legend(loc="upper left", framealpha=0.95, handletextpad=0.4,
              fontsize=plt.rcParams["axes.labelsize"] * 0.72,
              title="HybridAL variant")


def main():
    apply_acl_style(scale=1.55)
    run_meta = _load_run_meta()
    epochs_data = _per_round_epochs(run_meta)
    switch_data = _switch_rounds_per_dataset(run_meta)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    _draw_epochs_panel(axes[0], epochs_data)
    _draw_switch_strip(axes[1], switch_data)

    fig.suptitle("HybridAL: where time saving comes from (left) and how the switch adapts (right)",
                 fontweight="bold", y=1.00,
                 fontsize=plt.rcParams["axes.titlesize"] * 1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT.with_suffix('.pdf').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
