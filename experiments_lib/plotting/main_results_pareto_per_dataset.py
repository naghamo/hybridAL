"""Per-dataset Pareto grid: one F1-vs-time panel per dataset.

For each of the six datasets, a separate Pareto scatter is drawn with one
marker per method (averaged over seeds). Markers / colors match the
global ``main_results_pareto`` figure so the per-dataset picture lines up
visually with the aggregated one.

Output:
  experiments/main_results/plots/pareto_per_dataset.{png,pdf}
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .style import apply_acl_style


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY = REPO_ROOT / "experiments" / "main_results" / "_summary.csv"
OUT = REPO_ROOT / "experiments" / "main_results" / "plots" / "pareto_per_dataset.png"

DATASETS = ["imdb", "agnews", "jigsaw", "sst2", "tweeteval", "yahoo_answers"]
DS_DISP = {
    "imdb": "IMDb", "agnews": "AG News", "jigsaw": "Jigsaw",
    "sst2": "SST-2", "tweeteval": "TweetEval", "yahoo_answers": "Yahoo",
}
BACKBONE = "DistilBERT"

# (method_key, display_label, color, marker, marker_size)
METHODS = [
    ("Retrain",         "Retrain",                    "#000000", "s", 220),
    ("FineTune",        "FineTune",                   "#1f77b4", "^", 220),
    # NewOnly excluded until the round-1-warmup fix re-runs complete; the
    # archived warm-up cells live in _archive_newonly_warmup/.
    ("HybridAL_alpha",  r"HybridAL ($\Delta\alpha$)", "#d62728", "o", 280),
    ("HybridAL_acc",    r"HybridAL ($\Delta$Acc)",    "#ff7f0e", "o", 280),
    ("FixedSwitch_3",   "FixedSwitch (3)",            "#9467bd", "P", 200),
    ("FixedSwitch_5",   "FixedSwitch (5)",            "#8c564b", "P", 200),
    ("FixedSwitch_7",   "FixedSwitch (7)",            "#17becf", "P", 200),
    ("FixedSwitch_10",  "FixedSwitch (10)",           "#7f7f7f", "P", 200),
]


def _load():
    f1 = defaultdict(list)
    ac = defaultdict(list)
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
            if r.get("test_accuracy"):
                ac[(m, d)].append(float(r["test_accuracy"]))
            if r.get("training_time_total"):
                tm[(m, d)].append(float(r["training_time_total"]))
    return f1, ac, tm


def _draw_panel(ax, ds, metric_dict, tm):
    """metric_dict can be either f1 or ac, plotted on the y-axis."""
    from statistics import stdev as _stdev
    points = []
    for key, label, color, marker, size in METHODS:
        ys = metric_dict.get((key, ds), [])
        tms = tm.get((key, ds), [])
        if not ys or not tms:
            continue
        points.append({
            "key": key, "label": label, "color": color,
            "marker": marker, "size": size,
            "f1_m":  _mean(ys),
            "f1_s":  _stdev(ys) if len(ys) > 1 else 0.0,
            "tm_m":  _mean(tms),
            "tm_s":  _stdev(tms) if len(tms) > 1 else 0.0,
        })

    if not points:
        ax.set_title(f"{DS_DISP[ds]}\n(no data)")
        ax.axis("off")
        return

    for p in points:
        ax.scatter([p["tm_m"]], [p["f1_m"]],
                   s=p["size"], marker=p["marker"], color=p["color"],
                   edgecolor="black", linewidths=1.0, zorder=5,
                   label=p["label"])

    f1_vals = [p["f1_m"] for p in points]
    tm_vals = [p["tm_m"] for p in points]
    pad_x = 0.10 * (max(tm_vals) - min(tm_vals)) if max(tm_vals) > min(tm_vals) else 1
    pad_y = 0.12 * (max(f1_vals) - min(f1_vals)) if max(f1_vals) > min(f1_vals) else 0.005
    ax.set_xlim(min(tm_vals) - pad_x, max(tm_vals) + pad_x)
    ax.set_ylim(min(f1_vals) - pad_y, max(f1_vals) + pad_y)

    ax.set_title(DS_DISP[ds], fontweight="bold")
    ax.grid(True, which="major", alpha=0.30)


def _render(metric_dict, tm, metric_label, out_path):
    fig = plt.figure(figsize=(13.6, 8.5))
    gs = GridSpec(2, 3, figure=fig,
                  hspace=0.45, wspace=0.30,
                  left=0.06, right=0.98, top=0.94, bottom=0.16)

    grid_order = ["imdb", "sst2", "agnews",
                  "jigsaw", "tweeteval", "yahoo_answers"]

    legend_handles = None
    for idx, ds in enumerate(grid_order):
        r, c = divmod(idx, 3)
        ax = fig.add_subplot(gs[r, c])
        _draw_panel(ax, ds, metric_dict, tm)
        if c == 0:
            ax.set_ylabel(f"test {metric_label}")
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    out_pdf = out_path.with_suffix(".pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    print(f"wrote {out_pdf.relative_to(REPO_ROOT)}")


def main():
    apply_acl_style(scale=1.65)
    f1, ac, tm = _load()
    _render(f1, tm, metric_label="F1",       out_path=OUT)
    _render(ac, tm, metric_label="accuracy",
            out_path=OUT.with_name("pareto_per_dataset_acc.png"))


if __name__ == "__main__":
    main()
