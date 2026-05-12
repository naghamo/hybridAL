"""3-panel Pareto figure — F1 vs total training time, one panel per backbone.

Replaces the giant per-cell F1 + time tables in the main paper. Each panel
plots one point per method (averaged across the 6 datasets / 5 seeds).
The Pareto frontier is shaded so reviewers can see HybridAL sits on it
without reading a table.

Output:
  experiments/main_results/plots/main_results_pareto_per_backbone.{png,pdf}
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean, stdev as _stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .style import apply_acl_style


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY = REPO_ROOT / "experiments" / "main_results" / "_summary.csv"
OUT = REPO_ROOT / "experiments" / "main_results" / "plots" / "main_results_pareto_per_backbone.png"

DATASETS = ["imdb", "agnews", "jigsaw", "sst2", "tweeteval", "yahoo_answers"]
BACKBONES = ["DistilBERT", "BERT", "RoBERTa"]

# (method_key, display_label, color, marker, marker_size)
METHODS = [
    ("Retrain",         "Retrain",                       "#000000", "s", 180),
    ("FineTune",        "FineTune",                      "#1f77b4", "^", 180),
    ("HybridAL_alpha",  r"HybridAL ($\Delta\alpha$)",    "#d62728", "o", 260),
    ("HybridAL_acc",    r"HybridAL ($\Delta$Acc)",       "#ff7f0e", "o", 260),
    ("FixedSwitch_3",   "FixedSwitch (3)",               "#9467bd", "P", 160),
    ("FixedSwitch_5",   "FixedSwitch (5)",               "#8c564b", "P", 160),
    ("FixedSwitch_7",   "FixedSwitch (7)",               "#17becf", "P", 160),
    ("FixedSwitch_10",  "FixedSwitch (10)",              "#7f7f7f", "P", 160),
]


def _load_per_backbone():
    """Returns {backbone: (f1_dict, tm_dict)} where each is
    keyed by (method, dataset) → list of seed values."""
    out = {b: (defaultdict(list), defaultdict(list)) for b in BACKBONES}
    with SUMMARY.open() as fh:
        for r in csv.DictReader(fh):
            b = r.get("backbone")
            m = r.get("method")
            d = r.get("data")
            if b not in out or not m or d not in DATASETS:
                continue
            f1d, tmd = out[b]
            try:
                f1d[(m, d)].append(float(r["test_f1"]))
            except (TypeError, ValueError, KeyError):
                pass
            try:
                tmd[(m, d)].append(float(r["training_time_total"]))
            except (TypeError, ValueError, KeyError):
                pass
    return out


def _aggregate(values_dict, method):
    """Mean across the 6 dataset means; std across them; n datasets present."""
    per_ds = [_mean(values_dict[(method, d)])
              for d in DATASETS if values_dict.get((method, d))]
    if not per_ds:
        return None, None, 0
    return _mean(per_ds), (_stdev(per_ds) if len(per_ds) > 1 else 0.0), len(per_ds)


def _pareto_frontier(points):
    """Given a list of (x=time, y=f1, idx), return the index subset that is
    Pareto-optimal (minimise time, maximise f1)."""
    sorted_pts = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier = []
    best_y = -float("inf")
    for x, y, idx in sorted_pts:
        if y > best_y:
            frontier.append(idx)
            best_y = y
    return set(frontier)


def _draw_panel(ax, f1d, tmd, backbone, show_y_label, show_legend):
    points_for_pareto = []
    plotted = []
    for i, (key, label, color, marker, size) in enumerate(METHODS):
        f1_m, f1_s, n_f1 = _aggregate(f1d, key)
        tm_m, tm_s, n_tm = _aggregate(tmd, key)
        if f1_m is None or tm_m is None or n_f1 < 3:
            continue
        points_for_pareto.append((tm_m, f1_m, i))
        plotted.append({
            "key": key, "label": label, "color": color,
            "marker": marker, "size": size,
            "f1_m": f1_m, "f1_s": f1_s, "tm_m": tm_m, "tm_s": tm_s,
        })

    if not plotted:
        ax.text(0.5, 0.5, f"{backbone}: no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, alpha=0.6)
        ax.set_title(backbone, fontweight="bold")
        return

    frontier_idx = _pareto_frontier(points_for_pareto)

    for i, p in zip([pt[2] for pt in points_for_pareto], plotted):
        edge = "black" if i in frontier_idx else "#888888"
        lw = 1.6 if i in frontier_idx else 0.8
        ax.errorbar([p["tm_m"]], [p["f1_m"]],
                    xerr=[p["tm_s"]], yerr=[p["f1_s"]],
                    fmt="none", ecolor="gray", elinewidth=0.8,
                    capsize=2.5, zorder=3, alpha=0.55)
        ax.scatter([p["tm_m"]], [p["f1_m"]],
                   s=p["size"], marker=p["marker"], color=p["color"],
                   edgecolor=edge, linewidths=lw, zorder=5,
                   label=p["label"] if show_legend else None)

    sorted_frontier = sorted(
        [(p, points_for_pareto[k]) for k, p in enumerate(plotted)
         if points_for_pareto[k][2] in frontier_idx],
        key=lambda z: z[1][0],
    )
    if len(sorted_frontier) >= 2:
        xs = [z[1][0] for z in sorted_frontier]
        ys = [z[1][1] for z in sorted_frontier]
        ax.plot(xs, ys, linestyle="--", color="#555555",
                linewidth=1.0, zorder=4, alpha=0.55)

    f1_vals = [p["f1_m"] for p in plotted]
    tm_vals = [p["tm_m"] for p in plotted]
    pad_x = 0.10 * (max(tm_vals) - min(tm_vals) or 1)
    pad_y = 0.12 * (max(f1_vals) - min(f1_vals) or 0.01)
    ax.set_xlim(min(tm_vals) - pad_x, max(tm_vals) + 1.4 * pad_x)
    ax.set_ylim(min(f1_vals) - pad_y, max(f1_vals) + pad_y)

    ax.set_title(backbone, fontweight="bold", pad=4)
    ax.set_xlabel("mean time (s)")
    if show_y_label:
        ax.set_ylabel("mean test F1")
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10, linestyle=":")


def main():
    apply_acl_style(scale=1.55)
    data = _load_per_backbone()

    fig, axes = plt.subplots(1, len(BACKBONES), figsize=(13.5, 4.4),
                              sharey=False)
    if len(BACKBONES) == 1:
        axes = [axes]

    for i, b in enumerate(BACKBONES):
        f1d, tmd = data.get(b, (None, None))
        if f1d is None:
            continue
        _draw_panel(axes[i], f1d, tmd, b,
                    show_y_label=(i == 0),
                    show_legend=(i == 0))

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        for ax in axes:
            leg = ax.get_legend()
            if leg: leg.remove()
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(8, len(labels)),
                   bbox_to_anchor=(0.5, -0.03),
                   framealpha=0.95,
                   handletextpad=0.4,
                   fontsize=plt.rcParams["axes.labelsize"] * 0.80,
                   columnspacing=1.2)

    fig.suptitle("F1 vs. training time per backbone — HybridAL on the Pareto frontier",
                 fontweight="bold", y=1.00, fontsize=plt.rcParams["axes.titlesize"] * 1.05)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT.with_suffix('.pdf').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
