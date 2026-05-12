"""Pareto plot — all training strategies on one panel.

X = mean training time across the 6 datasets / 5 seeds (lower better).
Y = mean test F1 (macro) across the 6 datasets / 5 seeds (higher better).
Each method is a single labelled marker; error bars show $\\pm 1$ std
across the 6 dataset means. Background quadrants flag the
"better than Retrain on both axes" region.

DistilBERT is used as the canonical backbone (most complete data).

Output:
  experiments/main_results/plots/pareto_all_methods.{png,pdf}
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
OUT = REPO_ROOT / "experiments" / "main_results" / "plots" / "pareto_all_methods.png"

DATASETS = ["imdb", "agnews", "jigsaw", "sst2", "tweeteval", "yahoo_answers"]
BACKBONE = "DistilBERT"

# (method_key, display_label, color, marker, marker_size, label_offset_xy)
METHODS = [
    ("Retrain",         "Retrain",        "#000000", "s", 220, ( 6,  6)),
    ("FineTune",        "FineTune",       "#1f77b4", "^", 220, ( 6,  6)),
    # NewOnly excluded until the round-1-warmup fix re-runs complete; the
    # archived warm-up cells live in _archive_newonly_warmup/.
    ("HybridAL_alpha",  r"HybridAL ($\Delta\alpha$)", "#d62728", "o", 280, ( 6,  6)),
    ("HybridAL_acc",    r"HybridAL ($\Delta$Acc)",    "#ff7f0e", "o", 280, ( 6,  6)),
    ("FixedSwitch_3",   "FixedSwitch (3)",  "#9467bd", "P", 200, ( 6,  6)),
    ("FixedSwitch_5",   "FixedSwitch (5)",  "#8c564b", "P", 200, ( 6,  6)),
    ("FixedSwitch_7",   "FixedSwitch (7)",  "#17becf", "P", 200, ( 6,  6)),
    ("FixedSwitch_10",  "FixedSwitch (10)", "#7f7f7f", "P", 200, ( 6,  6)),
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


def _aggregate(values_dict, method):
    """Return (mean over datasets of per-dataset means, std across the 6
    dataset means, n_datasets_present)."""
    per_ds = []
    for d in DATASETS:
        vs = values_dict.get((method, d), [])
        if vs:
            per_ds.append(_mean(vs))
    if not per_ds:
        return None, None, 0
    m = _mean(per_ds)
    s = _stdev(per_ds) if len(per_ds) > 1 else 0.0
    return m, s, len(per_ds)


def _render(metric_dict, tm, metric_label, out_path):
    """Render one Pareto figure for the given y-axis metric (F1 or Acc)."""
    points = []
    for key, label, color, marker, size, offset in METHODS:
        y_m, y_s, n_y = _aggregate(metric_dict, key)
        tm_m, tm_s, n_tm = _aggregate(tm, key)
        if y_m is None or tm_m is None or n_y < 3:
            print(f"  skipping {key}: insufficient data (n_y={n_y}, n_tm={n_tm})")
            continue
        points.append({
            "key": key, "label": label, "color": color, "marker": marker,
            "size": size, "offset": offset,
            "y_m": y_m, "y_s": y_s, "tm_m": tm_m, "tm_s": tm_s,
        })

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for p in points:
        ax.scatter([p["tm_m"]], [p["y_m"]],
                   s=p["size"], marker=p["marker"], color=p["color"],
                   edgecolor="black", linewidths=1.2, zorder=5,
                   label=p["label"])

    y_vals = [p["y_m"] for p in points]
    tm_vals = [p["tm_m"] for p in points]
    pad_x = 0.08 * (max(tm_vals) - min(tm_vals))
    pad_y = 0.08 * (max(y_vals) - min(y_vals)) if max(y_vals) > min(y_vals) else 0.005
    ax.set_xlim(min(tm_vals) - pad_x, max(tm_vals) + 1.5 * pad_x)
    ax.set_ylim(min(y_vals) - pad_y, max(y_vals) + pad_y)

    ax.set_xlabel("← faster        mean training time (s)        slower →")
    ax.set_ylabel(f"worse ← mean test {metric_label} →  better")
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10, linestyle=":")
    ax.legend(loc="best", framealpha=0.95,
              handletextpad=0.5, fontsize=plt.rcParams["axes.labelsize"] * 0.75)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    out_pdf = out_path.with_suffix(".pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    print(f"wrote {out_pdf.relative_to(REPO_ROOT)}")


def main():
    apply_acl_style(scale=1.90)
    f1, ac, tm = _load()
    _render(f1, tm, metric_label="F1",       out_path=OUT)
    _render(ac, tm, metric_label="accuracy",
            out_path=OUT.with_name("pareto_all_methods_acc.png"))


if __name__ == "__main__":
    main()
