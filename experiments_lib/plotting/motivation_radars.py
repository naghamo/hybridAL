"""Intro / motivation figure: two radar plots, one per metric, with
task-family axes.

The six datasets are grouped into four NLP task families so the radar
axes read as universal task types in the intro (no dataset names
needed):

  Sentiment   = IMDb + SST-2
  News topic  = AG News + Yahoo
  Toxicity    = Jigsaw
  Emotion     = TweetEval

Per family, each strategy's value is the mean over the family's
datasets (which themselves are means over seeds).

Left panel  — final test F1 (macro). Higher = further out.
Right panel — speed-up over Retrain ($T_{\\text{Retrain}} / T$).
              Retrain sits on the unit circle by definition; faster
              strategies push outwards.

DistilBERT is used as the canonical backbone for the motivation plot
since it has the most complete 6-dataset coverage; the per-backbone
breakdown lives in the main-results section.

Output:
  experiments/main_results/plots/motivation_radars.{png,pdf}

Run as a module:
  python -m experiments_lib.plotting.motivation_radars
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .style import apply_acl_style


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY = REPO_ROOT / "experiments" / "main_results" / "_summary.csv"
OUT = REPO_ROOT / "experiments" / "main_results" / "plots" / "motivation_radars.png"

DATASETS = ["imdb", "agnews", "jigsaw", "sst2", "tweeteval", "yahoo_answers"]
TASK_FAMILIES = [
    ("Sentiment",  ["imdb", "sst2"]),
    ("News topic", ["agnews", "yahoo_answers"]),
    ("Toxicity",   ["jigsaw"]),
    ("Emotion",    ["tweeteval"]),
]
BACKBONE = "DistilBERT"

# (label, method_key, color, line, marker)
# NewOnly excluded until the round-1-warmup fix re-runs complete; the
# archived warm-up cells live in _archive_newonly_warmup/.
STRATEGIES = [
    ("Retrain",  "Retrain",        "#000000", "-",  "s"),
    ("FineTune", "FineTune",       "#1f77b4", ":",  "^"),
    ("HybridAL", "HybridAL_alpha", "#d62728", "-",  "o"),
]


def _load():
    """Per-(method, dataset) lists of (f1, time) over seeds."""
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


def _family_value(values_dict, method, datasets):
    """Mean over datasets in a family, where each dataset is itself the
    mean over its seeds. Returns None if the strategy has no data on any
    member of the family."""
    per_ds = []
    for d in datasets:
        vs = values_dict.get((method, d))
        if vs:
            per_ds.append(_mean(vs))
    return _mean(per_ds) if per_ds else None


def _polygon(angles, vals):
    return list(angles) + [angles[0]], list(vals) + [vals[0]]


def _draw_radar(ax, axis_labels, per_strategy, ylim, ylabel,
                ref_circle=None):
    n = len(axis_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(axis_labels)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="x", pad=12)

    if ref_circle is not None:
        a, v = _polygon(angles, [ref_circle] * n)
        ax.plot(a, v, color="gray", linestyle=":", linewidth=1.0,
                alpha=0.7, zorder=1)

    for label, color, ls, marker, vals in per_strategy:
        if any(v is None for v in vals):
            continue
        a, v = _polygon(angles, vals)
        ax.plot(a, v, color=color, linestyle=ls, linewidth=2.0,
                marker=marker, markersize=7, label=label, zorder=5)
        ax.fill(a, v, color=color, alpha=0.08, zorder=4)

    ax.set_ylabel(ylabel, labelpad=24)
    ax.grid(True, alpha=0.30)


def main():
    apply_acl_style(scale=1.55)
    f1, tm = _load()

    family_labels = [name for name, _ in TASK_FAMILIES]

    # Raw F1 per (strategy, family).
    f1_per = []
    for label, key, color, ls, marker in STRATEGIES:
        vals = [_family_value(f1, key, ds) for _, ds in TASK_FAMILIES]
        f1_per.append((label, color, ls, marker, vals))

    # Per-axis (per task-family) normalisation: scale each family's
    # values across the four strategies to [0, 1] so 1pp differences
    # become visible. Loses raw magnitudes — the caption / annotations
    # convey those.
    n_families = len(TASK_FAMILIES)
    axis_mins, axis_maxs = [], []
    for fam_idx in range(n_families):
        col = [strat_vals[4][fam_idx] for strat_vals in f1_per
               if strat_vals[4][fam_idx] is not None]
        axis_mins.append(min(col) if col else 0.0)
        axis_maxs.append(max(col) if col else 1.0)

    f1_per_norm = []
    for label, color, ls, marker, vals in f1_per:
        nvs = []
        for i, v in enumerate(vals):
            if v is None:
                nvs.append(None); continue
            lo, hi = axis_mins[i], axis_maxs[i]
            nvs.append(0.5 if hi - lo < 1e-12 else (v - lo) / (hi - lo))
        f1_per_norm.append((label, color, ls, marker, nvs))

    # Speed-up = T_Retrain / T_strategy, per family (averaged over family
    # datasets after computing the per-dataset speed-up).
    speedup_per = []
    for label, key, color, ls, marker in STRATEGIES:
        vals = []
        for _, datasets in TASK_FAMILIES:
            ratios = []
            for d in datasets:
                t = _mean(tm.get((key, d), [])) if tm.get((key, d)) else None
                tr = _mean(tm.get(("Retrain", d), [])) if tm.get(("Retrain", d)) else None
                if t and tr:
                    ratios.append(tr / t)
            vals.append(_mean(ratios) if ratios else None)
        speedup_per.append((label, color, ls, marker, vals))

    # Speed-up axis range.
    sp_ceil  = min(5.0, max(v for _, _, _, _, vs in speedup_per
                              for v in vs if v is not None) + 0.3)

    fig = plt.figure(figsize=(13.6, 6.5))
    ax_f1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax_sp = fig.add_subplot(1, 2, 2, projection="polar")

    _draw_radar(ax_f1, family_labels, f1_per_norm,
                ylim=(-0.05, 1.10),
                ylabel="F1 rank within task (1 = best)")
    ax_f1.set_yticks([0.0, 0.5, 1.0])
    ax_f1.set_yticklabels(["worst", "", "best"])
    ax_f1.set_title("Test F1 — per-task ranking",
                    pad=22, fontweight="bold")
    # Annotate raw F1 values for the BEST strategy on each axis (avoids
    # 16-label clutter while keeping ground truth on the figure).
    angles = np.linspace(0, 2 * np.pi, n_families, endpoint=False)
    for fam_idx, ang in enumerate(angles):
        best = None
        for label, color, ls, marker, vals in f1_per:
            v = vals[fam_idx]
            if v is None:
                continue
            if best is None or v > best[1]:
                best = (label, v, color)
        if best is not None:
            ax_f1.annotate(f"{best[1]:.3f}",
                           xy=(ang, 1.05),
                           ha="center", va="center",
                           fontsize=plt.rcParams["axes.titlesize"] * 0.55,
                           color=best[2], alpha=0.85)

    _draw_radar(ax_sp, family_labels, speedup_per,
                ylim=(0, sp_ceil),
                ylabel="speed-up vs Retrain ($\\times$)",
                ref_circle=1.0)
    ax_sp.set_title("Training speed-up vs Retrain — outer is faster",
                    pad=22, fontweight="bold")

    handles, labels = ax_f1.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=len(STRATEGIES), framealpha=0.95,
               handletextpad=0.5, columnspacing=2.5)

    fig.subplots_adjust(left=0.04, right=0.96, top=0.92,
                       bottom=0.10, wspace=0.45)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    out_pdf = OUT.with_suffix(".pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"wrote {out_pdf.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
