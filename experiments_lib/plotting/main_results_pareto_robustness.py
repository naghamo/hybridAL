"""Robustness-vs-time Pareto plots.

The standard Pareto puts F1 on the y-axis. These figures put a
*robustness* score there instead, so the plot reads as
"who is consistent across the conditions, and how much time does it
cost?" — which is the contribution argument for HybridAL since the
F1 axis itself is statistically tied across methods.

Two figures:

  - pareto_robustness.{png,pdf}
        Aggregate. y = cross-dataset robustness (1 − CV of per-dataset
        mean F1 across the six datasets); x = mean training time across
        seeds and datasets. One marker per method.

  - pareto_robustness_per_dataset.{png,pdf}
        2 × 3 grid. Per panel: y = cross-seed robustness within that
        dataset (1 − CV of test F1 across the five seeds); x = mean
        training time on that dataset. One marker per method.

Higher y = more consistent / robust. So the *upper-left* quadrant is
"fast AND robust", which is the position HybridAL aims for.

DistilBERT only (most complete data); NewOnly excluded until the
warm-up-fix re-runs land.
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean, stdev as _stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .style import apply_acl_style


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY = REPO_ROOT / "experiments" / "main_results" / "_summary.csv"
OUT_AGG = REPO_ROOT / "experiments" / "main_results" / "plots" / "pareto_robustness.png"
OUT_PER = REPO_ROOT / "experiments" / "main_results" / "plots" / "pareto_robustness_per_dataset.png"

DATASETS = ["imdb", "agnews", "jigsaw", "sst2", "tweeteval", "yahoo_answers"]
DS_DISP = {
    "imdb": "IMDb", "agnews": "AG News", "jigsaw": "Jigsaw",
    "sst2": "SST-2", "tweeteval": "TweetEval", "yahoo_answers": "Yahoo",
}
GRID_ORDER = ["imdb", "sst2", "agnews", "jigsaw", "tweeteval", "yahoo_answers"]
BACKBONE = "DistilBERT"

# (method_key, display_label, color, marker, marker_size, label_offset_xy)
METHODS = [
    ("Retrain",         "Retrain",                    "#000000", "s", 220, ( 6,  6)),
    ("FineTune",        "FineTune",                   "#1f77b4", "^", 220, ( 6,  6)),
    # NewOnly excluded until the round-1-warmup fix re-runs complete.
    ("HybridAL_alpha",  r"HybridAL ($\Delta\alpha$)", "#d62728", "o", 280, ( 6,  6)),
    ("HybridAL_acc",    r"HybridAL ($\Delta$Acc)",    "#ff7f0e", "o", 280, ( 6,  6)),
    ("FixedSwitch_3",   "FixedSwitch (3)",            "#9467bd", "P", 200, ( 6,  6)),
    ("FixedSwitch_5",   "FixedSwitch (5)",            "#8c564b", "P", 200, ( 6,  6)),
    ("FixedSwitch_7",   "FixedSwitch (7)",            "#17becf", "P", 200, ( 6,  6)),
    ("FixedSwitch_10",  "FixedSwitch (10)",           "#7f7f7f", "P", 200, ( 6,  6)),
]


def _load():
    f1 = defaultdict(list)
    tm = defaultdict(list)
    with SUMMARY.open() as fh:
        for r in csv.DictReader(fh):
            if r.get("backbone") != BACKBONE:
                continue
            m = r.get("method")
            d = r.get("data")
            if not m or d not in DATASETS:
                continue
            if r.get("test_f1"):
                f1[(m, d)].append(float(r["test_f1"]))
            if r.get("training_time_total"):
                tm[(m, d)].append(float(r["training_time_total"]))
    return f1, tm


def _cv(values):
    """Coefficient of variation = std / mean. Returns None for empty / zero-mean."""
    if not values or len(values) < 2:
        return None
    m = _mean(values)
    if m == 0:
        return None
    return _stdev(values) / m


QUADRANT_GOOD = "#9fd99c"   # green:  strictly better than Retrain on both axes
QUADRANT_BAD  = "#f5b1b1"   # red:    strictly worse on both
QUADRANT_TRADE = "#ffe7a3"  # yellow: better on one axis, worse on the other


def _shade_quadrants(ax, ref_x, ref_y, x_better="low"):
    """Shade all four quadrants around (ref_x, ref_y).

    x_better='low'  (e.g. time):  good corner is upper-left.
    x_better='high' (e.g. F1):    good corner is upper-right.
    y is always higher-is-better in this module.
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    f_x = (ref_x - xmin) / (xmax - xmin)
    if x_better == "low":
        # Upper-left = strictly better, lower-right = strictly worse,
        # the other two are mixed tradeoffs.
        ax.axhspan(ref_y, ymax, xmin=0,   xmax=f_x, facecolor=QUADRANT_GOOD,  alpha=0.35, zorder=0)
        ax.axhspan(ref_y, ymax, xmin=f_x, xmax=1,   facecolor=QUADRANT_TRADE, alpha=0.30, zorder=0)
        ax.axhspan(ymin,  ref_y, xmin=0,   xmax=f_x, facecolor=QUADRANT_TRADE, alpha=0.30, zorder=0)
        ax.axhspan(ymin,  ref_y, xmin=f_x, xmax=1,   facecolor=QUADRANT_BAD,   alpha=0.30, zorder=0)
    else:  # x_better == "high"
        ax.axhspan(ref_y, ymax, xmin=f_x, xmax=1,   facecolor=QUADRANT_GOOD,  alpha=0.35, zorder=0)
        ax.axhspan(ref_y, ymax, xmin=0,   xmax=f_x, facecolor=QUADRANT_TRADE, alpha=0.30, zorder=0)
        ax.axhspan(ymin,  ref_y, xmin=f_x, xmax=1,   facecolor=QUADRANT_TRADE, alpha=0.30, zorder=0)
        ax.axhspan(ymin,  ref_y, xmin=0,   xmax=f_x, facecolor=QUADRANT_BAD,   alpha=0.30, zorder=0)


def _scatter(ax, points, xlim_pad=0.10, ylim_pad=0.12, x_better="low"):
    """Render a set of method markers on the given axis. No Retrain
    cross-hairs / quadrant shading — the main-results figures compare
    every method against HybridAL, not against Retrain as a reference."""
    if not points:
        ax.axis("off")
        return

    for p in points:
        ax.scatter([p["x"]], [p["y"]],
                   s=p["size"], marker=p["marker"], color=p["color"],
                   edgecolor="black", linewidths=1.2, zorder=5,
                   label=p["label"])

    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    pad_x = xlim_pad * (max(xs) - min(xs)) if max(xs) > min(xs) else 1
    pad_y = ylim_pad * (max(ys) - min(ys)) if max(ys) > min(ys) else 0.005
    ax.set_xlim(min(xs) - pad_x, max(xs) + 1.5 * pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    ax.grid(True, which="major", alpha=0.30)


def _friedman_ranks(f1):
    """Friedman ranks: per dataset, rank the methods by mean F1
    (1 = best). Returns a dict[method_key] -> mean rank across the
    datasets where the method has data. Lower = better."""
    method_keys = [m[0] for m in METHODS]
    # mean F1 per (method, dataset).
    md_mean = {}
    for key in method_keys:
        for d in DATASETS:
            f1s = f1.get((key, d), [])
            if f1s:
                md_mean[(key, d)] = _mean(f1s)
    # Rank within each dataset over methods present.
    ranks = defaultdict(list)
    for d in DATASETS:
        present = [(k, md_mean[(k, d)]) for k in method_keys if (k, d) in md_mean]
        if len(present) < 2:
            continue
        # Sort descending by F1; ties get average rank.
        sorted_p = sorted(present, key=lambda x: -x[1])
        # Standard average-rank handling.
        i = 0
        while i < len(sorted_p):
            j = i
            while j + 1 < len(sorted_p) and abs(sorted_p[j + 1][1] - sorted_p[i][1]) < 1e-12:
                j += 1
            avg_rank = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                ranks[sorted_p[k][0]].append(avg_rank)
            i = j + 1
    return {m: _mean(ranks[m]) for m in ranks if ranks[m]}


def _agg_points(f1, tm):
    """Aggregate points. Each point gets fields:
       x = mean training time across (dataset, seed),
       y = mean Friedman rank across the six datasets (1 = best),
       f1_mean = mean of per-dataset F1 means (used as alt x-axis)."""
    fr = _friedman_ranks(f1)
    pts = []
    for key, label, color, marker, size, offset in METHODS:
        if key not in fr:
            continue
        per_ds_means = []
        all_times = []
        for d in DATASETS:
            f1_seeds = f1.get((key, d), [])
            tm_seeds = tm.get((key, d), [])
            if f1_seeds:
                per_ds_means.append(_mean(f1_seeds))
            all_times.extend(tm_seeds)
        if len(per_ds_means) < 3 or not all_times:
            continue
        pts.append({
            "key": key, "label": label, "color": color,
            "marker": marker, "size": size, "offset": offset,
            "x": _mean(all_times),
            "y": fr[key],
            "f1_mean": _mean(per_ds_means),
        })
    return pts


def _per_dataset_points(ds, f1, tm):
    """Per-dataset points. y = coverage = fraction of seeds where the
    method's F1 is within 1 pp of the per-seed best. Higher = more
    consistently competitive on that dataset."""
    method_keys = [m[0] for m in METHODS]
    n_seeds = max((len(f1.get((k, ds), [])) for k in method_keys), default=0)
    if n_seeds == 0:
        return []

    # Build per-seed F1 vectors (assume seed-index alignment via list
    # position, consistent with how _summary.csv is read in seed-sorted
    # order by the aggregator).
    by_method = {k: f1.get((k, ds), []) for k in method_keys}

    pts = []
    for key, label, color, marker, size, offset in METHODS:
        seeds = by_method.get(key, [])
        tms = tm.get((key, ds), [])
        if not seeds or not tms:
            continue
        # Coverage: fraction of seeds where this method is within 1 pp
        # of the best method on that seed.
        wins = 0
        cmp_n = 0
        for i in range(min(n_seeds, len(seeds))):
            opts = [by_method[k][i] for k in method_keys
                     if i < len(by_method[k])]
            if not opts:
                continue
            best = max(opts)
            if seeds[i] >= best - 0.01:
                wins += 1
            cmp_n += 1
        if cmp_n == 0:
            continue
        coverage = wins / cmp_n
        pts.append({
            "key": key, "label": label, "color": color,
            "marker": marker, "size": size, "offset": offset,
            "x": _mean(tms),
            "y": coverage,
            "f1_mean": _mean(seeds),
        })
    return pts


def _render_agg(f1, tm):
    apply_acl_style(scale=1.90)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    pts = _agg_points(f1, tm)
    _scatter(ax, pts)
    ax.invert_yaxis()   # rank: 1 (best) at top, n (worst) at bottom
    ax.set_xlabel("← faster        mean training time (s)        slower →")
    ax.set_ylabel("worse ← mean Friedman rank across datasets → better\n"
                  "(1 = best per-dataset F1)")
    ax.legend(loc="best", framealpha=0.95,
              handletextpad=0.5, fontsize=plt.rcParams["axes.labelsize"] * 0.75)
    fig.tight_layout()
    OUT_AGG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_AGG)
    fig.savefig(OUT_AGG.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {OUT_AGG.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_AGG.with_suffix('.pdf').relative_to(REPO_ROOT)}")


def _render_per_dataset(f1, tm):
    apply_acl_style(scale=1.65)
    fig = plt.figure(figsize=(13.6, 8.5))
    gs = GridSpec(2, 3, figure=fig,
                  hspace=0.45, wspace=0.30,
                  left=0.06, right=0.98, top=0.94, bottom=0.16)
    legend_handles = None
    for idx, ds in enumerate(GRID_ORDER):
        r, c = divmod(idx, 3)
        ax = fig.add_subplot(gs[r, c])
        pts = _per_dataset_points(ds, f1, tm)
        if not pts:
            ax.set_title(f"{DS_DISP[ds]}\n(no data)"); ax.axis("off")
            continue
        _scatter(ax, pts)
        ax.set_title(DS_DISP[ds], fontweight="bold")
        if c == 0:
            ax.set_ylabel("coverage\n(within 1pp of best, seeds)")
        if r == 1:
            ax.set_xlabel("training time (s)")
        if legend_handles is None:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend_handles = (handles, labels)

    if legend_handles is not None:
        handles, labels = legend_handles
        fig.legend(handles, labels,
                   loc="lower center", bbox_to_anchor=(0.5, 0.01),
                   ncol=5, framealpha=0.95,
                   handletextpad=0.5, columnspacing=1.5)

    OUT_PER.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PER)
    fig.savefig(OUT_PER.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {OUT_PER.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_PER.with_suffix('.pdf').relative_to(REPO_ROOT)}")


def _render_bar_agg(f1, tm):
    """Friedman-rank-only bar chart: one horizontal bar per method,
    sorted ascending by mean rank (1 = best). No time on the axes."""
    apply_acl_style(scale=1.55)
    pts = _agg_points(f1, tm)
    pts = sorted(pts, key=lambda p: p["y"])  # rank ascending = best first
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    labels = [p["label"] for p in pts]
    values = [p["y"]    for p in pts]
    colors = [p["color"] for p in pts]
    bars = ax.barh(range(len(pts)), values, color=colors,
                   edgecolor="black", linewidth=0.8)
    ax.set_yticks(range(len(pts)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("mean Friedman rank across datasets (1 = best)")
    ax.set_xlim(0.5, max(values) + 0.5)
    for bar, v in zip(bars, values):
        ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center",
                fontsize=plt.rcParams["axes.labelsize"] * 0.75)
    ax.grid(True, axis="x", which="major", alpha=0.30)
    fig.tight_layout()
    out = OUT_AGG.with_name("robustness_only.png")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    print(f"wrote {out.with_suffix('.pdf').relative_to(REPO_ROOT)}")


def _render_bar_per_dataset(f1, tm):
    """Per-dataset 2×3 grid of horizontal bar charts. Y = coverage =
    fraction of seeds where the method is within 1 pp of the per-seed
    best on that dataset."""
    apply_acl_style(scale=1.40)
    fig = plt.figure(figsize=(13.6, 7.5))
    gs = GridSpec(2, 3, figure=fig,
                  hspace=0.50, wspace=0.45,
                  left=0.10, right=0.98, top=0.94, bottom=0.10)
    for idx, ds in enumerate(GRID_ORDER):
        r, c = divmod(idx, 3)
        ax = fig.add_subplot(gs[r, c])
        pts = sorted(_per_dataset_points(ds, f1, tm),
                     key=lambda p: p["y"], reverse=True)
        if not pts:
            ax.set_title(f"{DS_DISP[ds]}\n(no data)"); ax.axis("off"); continue
        labels = [p["label"] for p in pts]
        values = [p["y"]    for p in pts]
        colors = [p["color"] for p in pts]
        ax.barh(range(len(pts)), values, color=colors,
                edgecolor="black", linewidth=0.6)
        ax.set_yticks(range(len(pts)))
        ax.set_yticklabels(labels, fontsize=plt.rcParams["axes.labelsize"] * 0.70)
        ax.invert_yaxis()
        ax.set_title(DS_DISP[ds], fontweight="bold")
        ax.set_xlim(0.0, 1.05)
        ax.grid(True, axis="x", which="major", alpha=0.30)
        if r == 1:
            ax.set_xlabel("coverage (seeds within 1 pp of best)")

    out = OUT_PER.with_name("robustness_only_per_dataset.png")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    print(f"wrote {out.with_suffix('.pdf').relative_to(REPO_ROOT)}")


def _render_robustness_vs_f1_agg(f1, tm):
    """Pareto with robustness on y, F1 on x — no time involved."""
    apply_acl_style(scale=1.90)
    pts_in = _agg_points(f1, tm)
    # Repackage: x = F1, y = robustness.
    pts = []
    for p in pts_in:
        pp = dict(p); pp["x"] = p["f1_mean"]; pts.append(pp)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    _scatter(ax, pts, x_better="high")
    ax.invert_yaxis()
    ax.set_xlabel("← worse        mean test F1        better →")
    ax.set_ylabel("worse ← mean Friedman rank across datasets → better\n"
                  "(1 = best per-dataset F1)")
    ax.legend(loc="best", framealpha=0.95,
              handletextpad=0.5, fontsize=plt.rcParams["axes.labelsize"] * 0.75)
    fig.tight_layout()
    out = OUT_AGG.with_name("pareto_robustness_vs_f1.png")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    print(f"wrote {out.with_suffix('.pdf').relative_to(REPO_ROOT)}")


def _render_robustness_vs_f1_per_dataset(f1, tm):
    """Per-dataset 2×3 grid; each panel: y = robustness, x = F1."""
    apply_acl_style(scale=1.65)
    fig = plt.figure(figsize=(13.6, 8.5))
    gs = GridSpec(2, 3, figure=fig,
                  hspace=0.45, wspace=0.30,
                  left=0.06, right=0.98, top=0.94, bottom=0.16)
    legend_handles = None
    for idx, ds in enumerate(GRID_ORDER):
        r, c = divmod(idx, 3)
        ax = fig.add_subplot(gs[r, c])
        pts_in = _per_dataset_points(ds, f1, tm)
        pts = [dict(p, x=p["f1_mean"]) for p in pts_in]
        if not pts:
            ax.set_title(f"{DS_DISP[ds]}\n(no data)"); ax.axis("off"); continue
        _scatter(ax, pts, x_better="high")
        ax.set_title(DS_DISP[ds], fontweight="bold")
        if c == 0:
            ax.set_ylabel("coverage (seeds\nwithin 1pp of best)")
        if r == 1:
            ax.set_xlabel("test F1")
        if legend_handles is None:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend_handles = (handles, labels)

    if legend_handles is not None:
        handles, labels = legend_handles
        fig.legend(handles, labels,
                   loc="lower center", bbox_to_anchor=(0.5, 0.01),
                   ncol=5, framealpha=0.95,
                   handletextpad=0.5, columnspacing=1.5)

    out = OUT_PER.with_name("pareto_robustness_vs_f1_per_dataset.png")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    print(f"wrote {out.with_suffix('.pdf').relative_to(REPO_ROOT)}")


def main():
    f1, tm = _load()
    _render_agg(f1, tm)
    _render_per_dataset(f1, tm)
    _render_bar_agg(f1, tm)
    _render_bar_per_dataset(f1, tm)
    _render_robustness_vs_f1_agg(f1, tm)
    _render_robustness_vs_f1_per_dataset(f1, tm)


if __name__ == "__main__":
    main()
