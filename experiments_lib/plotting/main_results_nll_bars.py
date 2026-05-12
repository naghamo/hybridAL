"""Compact NLL summary bar chart — alternative to the per-round trajectory.

Shows mean test-loss (NLL) per method, grouped by backbone, for the four
core training strategies. Lower = better. Direct visual evidence for
"warm-start fine-tuning hurts calibration" without the per-round
trajectory's complexity.

Output:
  experiments/main_results/plots/nll_bars.{png,pdf}
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean, stdev as _stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .style import apply_acl_style


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY = REPO_ROOT / "experiments" / "main_results" / "_summary.csv"
OUT = REPO_ROOT / "experiments" / "main_results" / "plots" / "nll_bars.png"

DATASETS = ["imdb", "agnews", "jigsaw", "sst2", "tweeteval", "yahoo_answers"]
BACKBONES = ["DistilBERT", "BERT", "RoBERTa"]

METHODS = [
    ("Retrain",         "Retrain",                    "#000000"),
    ("FineTune",        "FineTune",                   "#1f77b4"),
    ("HybridAL_alpha",  r"HybridAL ($\Delta\alpha$)", "#d62728"),
    ("HybridAL_acc",    r"HybridAL ($\Delta$Acc)",    "#ff7f0e"),
]


def _load():
    """Per-(method, backbone) list of test-loss values across (dataset, seed)."""
    cells = defaultdict(list)
    with SUMMARY.open() as fh:
        for r in csv.DictReader(fh):
            b, m, d = r.get("backbone"), r.get("method"), r.get("data")
            if not b or not m or d not in DATASETS:
                continue
            try:
                tl = float(r.get("test_loss"))
            except (TypeError, ValueError):
                continue
            cells[(b, m)].append(tl)
    return cells


def main():
    apply_acl_style(scale=1.45)
    cells = _load()

    # Compute mean & std across (dataset, seed) per (backbone, method).
    mu = {(b, m): (_mean(cells[(b, m)]) if cells[(b, m)] else None)
          for b in BACKBONES for m, _, _ in METHODS}
    sd = {(b, m): (_stdev(cells[(b, m)]) if len(cells[(b, m)]) > 1 else 0.0)
          for b in BACKBONES for m, _, _ in METHODS}

    fig, ax = plt.subplots(figsize=(8.5, 4.6))

    n_methods  = len(METHODS)
    n_backbones = len(BACKBONES)
    width = 0.20
    group_centers = np.arange(n_methods)

    for j, b in enumerate(BACKBONES):
        if not any(mu[(b, m)] is not None for m, _, _ in METHODS):
            continue
        xs   = group_centers + (j - (n_backbones - 1) / 2) * width
        vals = [mu[(b, m)] if mu[(b, m)] is not None else 0 for m, _, _ in METHODS]
        errs = [sd[(b, m)] for m, _, _ in METHODS]
        # Each backbone is a hatch / lighter shade; methods keep their colour.
        bars = ax.bar(xs, vals, width=width, yerr=errs,
                      ecolor="black", capsize=2.5,
                      edgecolor="black", linewidth=0.7,
                      label=b,
                      hatch=["", "//", ".."][j],
                      facecolor=["#cccccc", "#888888", "#444444"][j],
                      alpha=0.85, zorder=3)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([disp for _, disp, _ in METHODS])
    ax.set_ylabel("mean test NLL (lower is better)")
    ax.set_title("Test loss (NLL) per method — backbone-grouped",
                 fontweight="bold")
    ax.grid(True, axis="y", which="major", alpha=0.30, zorder=0)
    ax.legend(loc="upper left", framealpha=0.95, title="Backbone",
              handletextpad=0.4)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT.with_suffix('.pdf').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
