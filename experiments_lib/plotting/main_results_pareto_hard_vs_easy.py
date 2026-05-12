"""2-panel Pareto figure — Hard vs Easy tasks.

Replaces the giant per-cell F1 table. Each panel plots one point per
(method, backbone) combination that has FULL coverage (5 seeds on all 3
datasets in the panel's difficulty group). Methods are color-coded;
backbones are marker-coded. Pareto frontier per panel is dashed.

Difficulty grouping (matches the difficulty table):
  Hard (multi-class) : AG News (4-class), TweetEval (3-class), Yahoo (10-class)
  Easy (binary)      : IMDb, Jigsaw, SST-2

Output:
  experiments/main_results/plots/main_results_pareto_hard_vs_easy.{png,pdf}
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
OUT = REPO_ROOT / "experiments" / "main_results" / "plots" / "main_results_pareto_hard_vs_easy.png"

HARD_DATASETS = ["agnews", "tweeteval", "yahoo_answers"]
EASY_DATASETS = ["imdb", "jigsaw", "sst2"]
GROUPS = [("Hard (multi-class)", HARD_DATASETS),
          ("Easy (binary)",       EASY_DATASETS)]
BACKBONES = ["DistilBERT", "BERT", "RoBERTa"]
SEEDS_EXPECTED = 5

METHODS = [
    ("Retrain",         "Retrain",                    "#000000"),
    ("FineTune",        "FineTune",                   "#1f77b4"),
    ("HybridAL_alpha",  r"HybridAL ($\Delta\alpha$)", "#d62728"),
    ("HybridAL_acc",    r"HybridAL ($\Delta$Acc)",    "#ff7f0e"),
    ("FixedSwitch_3",   "FixedSwitch (3)",            "#9467bd"),
    ("FixedSwitch_5",   "FixedSwitch (5)",            "#8c564b"),
    ("FixedSwitch_7",   "FixedSwitch (7)",            "#17becf"),
    ("FixedSwitch_10",  "FixedSwitch (10)",           "#7f7f7f"),
]
BACKBONE_MARKERS = {"DistilBERT": "o", "BERT": "s", "RoBERTa": "^"}
BACKBONE_SIZES = {"DistilBERT": 200, "BERT": 200, "RoBERTa": 200}


def _load_per_cell():
    """Returns {(method, backbone, dataset): {seed: (f1, time)}}."""
    out = defaultdict(dict)
    with SUMMARY.open() as fh:
        for r in csv.DictReader(fh):
            try:
                seed = int(r["seed"])
                f1 = float(r["test_f1"])
                tm = float(r["training_time_total"])
            except (TypeError, ValueError, KeyError):
                continue
            key = (r.get("method"), r.get("backbone"), r.get("data"))
            if any(v is None for v in key):
                continue
            out[key][seed] = (f1, tm)
    return out


def _full_coverage(per_cell, method, backbone, datasets):
    """Return (mean F1, std F1, mean time, std time) only if every dataset
    in the group has all 5 seeds. Otherwise None."""
    per_ds_f1, per_ds_tm = [], []
    for d in datasets:
        seed_map = per_cell.get((method, backbone, d), {})
        if len(seed_map) < SEEDS_EXPECTED:
            return None
        f1s = [seed_map[s][0] for s in sorted(seed_map)][:SEEDS_EXPECTED]
        tms = [seed_map[s][1] for s in sorted(seed_map)][:SEEDS_EXPECTED]
        per_ds_f1.append(_mean(f1s))
        per_ds_tm.append(_mean(tms))
    if not per_ds_f1:
        return None
    return (_mean(per_ds_f1),
            _stdev(per_ds_f1) if len(per_ds_f1) > 1 else 0.0,
            _mean(per_ds_tm),
            _stdev(per_ds_tm) if len(per_ds_tm) > 1 else 0.0)


def _pareto_frontier(points):
    """points: list of (x=time, y=f1, idx). Returns indices on frontier
    (minimise x, maximise y)."""
    sorted_pts = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier, best_y = [], -float("inf")
    for x, y, idx in sorted_pts:
        if y > best_y:
            frontier.append(idx)
            best_y = y
    return set(frontier)


def _draw_panel(ax, per_cell, group_label, group_datasets, show_y_label):
    plotted = []
    raw = []
    for m_key, m_disp, m_color in METHODS:
        for b in BACKBONES:
            res = _full_coverage(per_cell, m_key, b, group_datasets)
            if res is None:
                continue
            f1_m, f1_s, tm_m, tm_s = res
            plotted.append({
                "method_key": m_key, "method_disp": m_disp,
                "backbone": b, "color": m_color,
                "marker": BACKBONE_MARKERS[b],
                "size": BACKBONE_SIZES[b],
                "f1_m": f1_m, "f1_s": f1_s, "tm_m": tm_m, "tm_s": tm_s,
            })
            raw.append((tm_m, f1_m, len(plotted) - 1))

    if not plotted:
        ax.text(0.5, 0.5, "no fully-finished cells yet",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, alpha=0.6)
        ax.set_title(group_label, fontweight="bold")
        return [], []

    frontier_idx = _pareto_frontier(raw)

    for i, p in enumerate(plotted):
        edge = "black" if i in frontier_idx else "#aaaaaa"
        lw = 1.6 if i in frontier_idx else 0.7
        ax.errorbar([p["tm_m"]], [p["f1_m"]],
                    xerr=[p["tm_s"]], yerr=[p["f1_s"]],
                    fmt="none", ecolor="gray", elinewidth=0.7,
                    capsize=2.5, zorder=3, alpha=0.5)
        ax.scatter([p["tm_m"]], [p["f1_m"]],
                   s=p["size"], marker=p["marker"], color=p["color"],
                   edgecolor=edge, linewidths=lw, zorder=5)

    sorted_front = sorted(
        [(plotted[i], raw[i]) for i in range(len(plotted)) if i in frontier_idx],
        key=lambda z: z[1][0],
    )
    if len(sorted_front) >= 2:
        xs = [z[1][0] for z in sorted_front]
        ys = [z[1][1] for z in sorted_front]
        ax.plot(xs, ys, linestyle="--", color="#555555",
                linewidth=1.0, zorder=4, alpha=0.55)

    tm_vals = [p["tm_m"] for p in plotted]
    f1_vals = [p["f1_m"] for p in plotted]
    pad_x = 0.10 * (max(tm_vals) - min(tm_vals) or 1)
    pad_y = 0.12 * (max(f1_vals) - min(f1_vals) or 0.01)
    ax.set_xlim(min(tm_vals) - pad_x, max(tm_vals) + 1.4 * pad_x)
    ax.set_ylim(min(f1_vals) - pad_y, max(f1_vals) + pad_y)

    ax.set_title(group_label, fontweight="bold", pad=4)
    ax.set_xlabel("mean time (s)")
    if show_y_label:
        ax.set_ylabel("mean test F1")
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10, linestyle=":")
    return plotted, frontier_idx


def _build_legend(fig, axes):
    """Two legends: method (color) and backbone (marker)."""
    method_handles = [
        plt.Line2D([], [], marker="o", linestyle="",
                   color=c, markersize=10, markeredgecolor="black", label=disp)
        for _, disp, c in METHODS
    ]
    backbone_handles = [
        plt.Line2D([], [], marker=BACKBONE_MARKERS[b], linestyle="",
                   color="lightgray", markersize=10, markeredgecolor="black", label=b)
        for b in BACKBONES
    ]
    fig.legend(method_handles, [h.get_label() for h in method_handles],
               loc="lower center", ncol=4,
               bbox_to_anchor=(0.30, -0.10),
               title="Method", framealpha=0.95,
               handletextpad=0.4,
               fontsize=plt.rcParams["axes.labelsize"] * 0.75,
               columnspacing=1.0)
    fig.legend(backbone_handles, [h.get_label() for h in backbone_handles],
               loc="lower center", ncol=3,
               bbox_to_anchor=(0.83, -0.10),
               title="Backbone", framealpha=0.95,
               handletextpad=0.4,
               fontsize=plt.rcParams["axes.labelsize"] * 0.75,
               columnspacing=1.0)


def main():
    apply_acl_style(scale=1.55)
    per_cell = _load_per_cell()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=False)
    for i, (label, dsets) in enumerate(GROUPS):
        _draw_panel(axes[i], per_cell, label, dsets, show_y_label=(i == 0))

    fig.suptitle("F1 vs. training time — hard (multi-class) vs. easy (binary) tasks",
                 fontweight="bold", y=1.00,
                 fontsize=plt.rcParams["axes.titlesize"] * 1.05)
    _build_legend(fig, axes)
    fig.tight_layout(rect=[0, 0.15, 1, 0.96])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT.with_suffix('.pdf').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
