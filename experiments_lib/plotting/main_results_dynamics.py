"""Five per-round dynamics plots for the methods-section narrative.

Generates:
  1. per_round_epochs.{png,pdf}    — actual epochs per round
                                      (where the time saving comes from)
  2. per_round_time.{png,pdf}      — wall-clock seconds per round
                                      (cumulative area = total saving)
  3. worst_seed_f1.{png,pdf}       — min F1 across the 5 seeds per round
                                      (exposes FineTune's tail risk that
                                      the mean hides)
  4. seed_variance_f1.{png,pdf}    — std of F1 across seeds per round
                                      (early-round instability)
  5. yahoo_trajectory.{png,pdf}    — F1 trajectory on Yahoo (the dataset
                                      where Retrain genuinely wins on
                                      final F1)

All five are DistilBERT-only (most complete data); each saved as a
separate file. HybridAL switch rounds are annotated where applicable.
"""
from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean, stdev as _stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from .style import apply_acl_style


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY = REPO_ROOT / "experiments" / "main_results" / "_summary.csv"
PER_ROUND = REPO_ROOT / "experiments" / "main_results" / "_per_round.csv"
OUT_DIR = REPO_ROOT / "experiments" / "main_results" / "plots"

DATASETS = ["imdb", "agnews", "jigsaw", "sst2", "tweeteval", "yahoo_answers"]
BACKBONE = "DistilBERT"

LINE_DEFS = {
    "Retrain":         ("Retrain",                    "#000000", "--", 1.6),
    "FineTune":        ("FineTune",                   "#1f77b4", ":",  1.4),
    "HybridAL_alpha":  (r"HybridAL ($\Delta\alpha$)", "#d62728", "-",  2.0),
    "HybridAL_acc":    (r"HybridAL ($\Delta$Acc)",    "#ff7f0e", "-",  2.0),
}


def _load():
    """Build run_meta[run] = (method, backbone, dataset, seed, switch_round)
    plus per_round[run] = list of (round, f1, loss, training_time,
    actual_epochs)."""
    run_meta = {}
    with SUMMARY.open() as fh:
        for r in csv.DictReader(fh):
            run = r.get("run")
            if not run: continue
            try: seed = int(r.get("seed"))
            except (TypeError, ValueError): continue
            sw = r.get("switch_round")
            try:
                sw_v = int(sw) if sw and sw.strip() else None
            except (TypeError, ValueError):
                sw_v = None
            run_meta[run] = (r.get("method"), r.get("backbone"),
                              r.get("data"), seed, sw_v)

    per_round = defaultdict(list)
    with PER_ROUND.open() as fh:
        for r in csv.DictReader(fh):
            run = r.get("run")
            if run not in run_meta: continue
            try:
                rd = int(r["round"])
                f1 = float(r["f1_score"])
                loss = float(r.get("loss") or "nan")
                tt = float(r.get("training_time") or "0")
                ae = int(r.get("actual_epochs") or "0")
            except (TypeError, ValueError):
                continue
            if not math.isfinite(f1): continue
            per_round[run].append((rd, f1, loss, tt, ae))
    for run in per_round:
        per_round[run].sort()
    return run_meta, per_round


def _per_round_aggregate(run_meta, per_round, method_key, field_idx,
                          dataset_filter=None, seed_aggregator=_mean):
    """For each round, aggregate the chosen field (idx into the per_round
    tuples) across (dataset, seed) for the given backbone+method.

    field_idx semantics:
        1 = f1
        2 = loss
        3 = training_time
        4 = actual_epochs
    """
    by_round = defaultdict(list)
    # If we want per-seed metrics (e.g. min/std across seeds), we first
    # group per (dataset, round) → list of seed values, then aggregate
    # within (dataset, round), then average across datasets.
    by_dataset_round = defaultdict(lambda: defaultdict(list))
    for run, recs in per_round.items():
        m, b, d, _seed, _sw = run_meta[run]
        if b != BACKBONE or m != method_key:
            continue
        if dataset_filter is not None and d != dataset_filter:
            continue
        if d not in DATASETS:
            continue
        for rec in recs:
            rd, val = rec[0], rec[field_idx]
            if val is None or (isinstance(val, float) and not math.isfinite(val)):
                continue
            by_dataset_round[d][rd].append(val)
            by_round[rd].append(val)
    # If seed_aggregator is mean → use simple flat aggregate (same as old).
    if seed_aggregator is _mean:
        return [(r, _mean(vs)) for r, vs in sorted(by_round.items())]
    # Otherwise: per (dataset, round), apply seed_aggregator, then mean
    # across datasets.
    rs = sorted({r for d_map in by_dataset_round.values() for r in d_map})
    out = []
    for rd in rs:
        per_ds_vals = []
        for d, d_map in by_dataset_round.items():
            seeds = d_map.get(rd, [])
            if seeds:
                per_ds_vals.append(seed_aggregator(seeds))
        if per_ds_vals:
            out.append((rd, _mean(per_ds_vals)))
    return out


def _switch_round_mean(run_meta, method_key, dataset_filter=None):
    sws = []
    for run, (m, b, d, _seed, sw) in run_meta.items():
        if b != BACKBONE or m != method_key or d not in DATASETS:
            continue
        if dataset_filter is not None and d != dataset_filter:
            continue
        if sw is not None:
            sws.append(sw)
    return _mean(sws) if sws else None


def _draw_lines(ax, lines, run_meta, per_round, field_idx,
                methods, dataset_filter=None,
                seed_aggregator=_mean,
                annotate_switches=False):
    plotted_any = False
    for m_key in methods:
        if m_key not in LINE_DEFS:
            continue
        label, color, ls, lw = LINE_DEFS[m_key]
        pts = _per_round_aggregate(run_meta, per_round, m_key, field_idx,
                                    dataset_filter=dataset_filter,
                                    seed_aggregator=seed_aggregator)
        if not pts:
            continue
        plotted_any = True
        rs, vs = zip(*pts)
        ax.plot(rs, vs, color=color, linestyle=ls, linewidth=lw, label=label,
                zorder=10 if m_key.startswith("HybridAL") else 5)

    if annotate_switches:
        for m_key, ann_y_frac in [("HybridAL_alpha", 0.95),
                                   ("HybridAL_acc",   0.85)]:
            if m_key not in methods:
                continue
            sw = _switch_round_mean(run_meta, m_key,
                                     dataset_filter=dataset_filter)
            if sw is None:
                continue
            color = LINE_DEFS[m_key][1]
            ax.axvline(sw, color=color, linestyle=":", linewidth=1.0,
                       alpha=0.6, zorder=2)
            ax.annotate(rf"$\bar t^\star{{=}}{sw:.1f}$",
                        xy=(sw, ann_y_frac), xytext=(2, 0),
                        textcoords="offset points",
                        color=color,
                        fontsize=plt.rcParams["axes.labelsize"] * 0.72,
                        xycoords=("data", "axes fraction"),
                        ha="left", va="top")
    return plotted_any


def _format(ax, ylabel, legend_loc="best"):
    ax.set_xlabel("acquisition round $t$")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10, linestyle=":")
    ax.legend(loc=legend_loc, framealpha=0.95,
              handletextpad=0.4,
              fontsize=plt.rcParams["axes.labelsize"] * 0.78)


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    print(f"wrote {out.with_suffix('.pdf').relative_to(REPO_ROOT)}")


def main():
    apply_acl_style(scale=1.45)
    run_meta, per_round = _load()

    methods_three = ["Retrain", "FineTune", "HybridAL_alpha"]
    methods_two = ["Retrain", "FineTune"]
    methods_four = ["Retrain", "FineTune", "HybridAL_alpha", "HybridAL_acc"]

    # 1. Per-round epochs.
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    _draw_lines(ax, None, run_meta, per_round, field_idx=4,
                 methods=methods_three, annotate_switches=True)
    ax.set_title(f"{BACKBONE} — actual epochs per round",
                 fontweight="bold")
    _format(ax, "mean actual epochs (after early stop)")
    _save(fig, "per_round_epochs")

    # 2. Per-round wall-clock time.
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    _draw_lines(ax, None, run_meta, per_round, field_idx=3,
                 methods=methods_three, annotate_switches=True)
    ax.set_title(f"{BACKBONE} — training time per round",
                 fontweight="bold")
    _format(ax, "mean training time / round (s)")
    _save(fig, "per_round_time")

    # 3. Worst-seed F1 trajectory.
    def _seed_min(xs): return min(xs)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    _draw_lines(ax, None, run_meta, per_round, field_idx=1,
                 methods=methods_two, seed_aggregator=_seed_min,
                 annotate_switches=False)
    ax.set_title(f"{BACKBONE} — worst-seed F1 per round\n"
                 r"(min across the 5 seeds, averaged across datasets)",
                 fontweight="bold")
    _format(ax, "worst-seed F1", legend_loc="lower right")
    _save(fig, "worst_seed_f1")

    # 4. Seed-variance F1 trajectory.
    def _seed_std(xs):
        return _stdev(xs) if len(xs) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    _draw_lines(ax, None, run_meta, per_round, field_idx=1,
                 methods=methods_two, seed_aggregator=_seed_std,
                 annotate_switches=False)
    ax.set_title(f"{BACKBONE} — seed std of F1 per round\n"
                 r"(std across the 5 seeds, averaged across datasets)",
                 fontweight="bold")
    _format(ax, "std F1 across seeds")
    _save(fig, "seed_variance_f1")

    # 5. Yahoo-only trajectory.
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    _draw_lines(ax, None, run_meta, per_round, field_idx=1,
                 methods=methods_four,
                 dataset_filter="yahoo_answers",
                 annotate_switches=True)
    ax.set_title(f"{BACKBONE} — F1 trajectory on Yahoo Answers (10-class)",
                 fontweight="bold")
    _format(ax, "mean val F1 (across 5 seeds)", legend_loc="lower right")
    _save(fig, "yahoo_trajectory")


if __name__ == "__main__":
    main()
