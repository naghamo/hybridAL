"""Per-round F1 trajectory plot — evidence for the Retrain-to-FineTune switch.

Renders one figure per backbone showing the validation-F1 trajectory of
the four core methods (Retrain, FineTune, HybridAL Δα, HybridAL ΔAcc),
averaged across seeds × datasets. The mean switch round of each HybridAL
variant is annotated as a vertical line.

The reader should see HybridAL's curve track Retrain *before* its switch
round and track FineTune *after* — i.e. visual confirmation that the
signal-driven switch lands at the productive transition between the
two regimes.

Output:
  experiments/main_results/plots/switch_evidence_<backbone>.{png,pdf}
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

from .style import apply_acl_style


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY = REPO_ROOT / "experiments" / "main_results" / "_summary.csv"
PER_ROUND = REPO_ROOT / "experiments" / "main_results" / "_per_round.csv"
OUT_DIR = REPO_ROOT / "experiments" / "main_results" / "plots"

DATASETS = ["imdb", "agnews", "jigsaw", "sst2", "tweeteval", "yahoo_answers"]
BACKBONES = ["DistilBERT", "BERT", "RoBERTa"]

# (method_key, label, color, linestyle, linewidth, marker, markersize)
LINES = [
    ("Retrain",         "Retrain",                    "#000000", "--", 1.6, "s", 0),
    ("FineTune",        "FineTune",                   "#1f77b4", ":",  1.4, "^", 0),
    ("HybridAL_alpha",  r"HybridAL ($\Delta\alpha$)", "#d62728", "-",  2.0, "o", 0),
    ("HybridAL_acc",    r"HybridAL ($\Delta$Acc)",    "#ff7f0e", "-",  2.0, "o", 0),
]


def _load():
    """Build run_meta[run] → (method, backbone, dataset, seed, switch_round)
    and per_round[run] → ordered list of (round, f1)."""
    run_meta = {}
    with SUMMARY.open() as fh:
        for r in csv.DictReader(fh):
            run = r.get("run")
            if not run:
                continue
            try:
                seed = int(r.get("seed"))
            except (TypeError, ValueError):
                continue
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
            if run not in run_meta:
                continue
            try:
                rd = int(r["round"])
                f1 = float(r["f1_score"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(f1):
                continue
            per_round[run].append((rd, f1))
    for run in per_round:
        per_round[run].sort()
    return run_meta, per_round


def _aggregate(run_meta, per_round, backbone, method_key):
    """Average F1 per round across (dataset, seed) for given (backbone, method)."""
    by_round = defaultdict(list)
    for run, recs in per_round.items():
        m, b, d, seed, _sw = run_meta[run]
        if b != backbone or m != method_key or d not in DATASETS:
            continue
        for rd, f1 in recs:
            by_round[rd].append(f1)
    if not by_round:
        return [], []
    rs = sorted(by_round)
    return rs, [_mean(by_round[r]) for r in rs]


def _switch_round_mean(run_meta, backbone, method_key):
    """Mean switch round across (dataset, seed) where the method fired."""
    sws = []
    for run, (m, b, d, _seed, sw) in run_meta.items():
        if b != backbone or m != method_key or d not in DATASETS:
            continue
        if sw is not None:
            sws.append(sw)
    return _mean(sws) if sws else None


def _draw(ax, run_meta, per_round, backbone):
    plotted_any = False
    for method_key, label, color, ls, lw, marker, msz in LINES:
        rs, ms = _aggregate(run_meta, per_round, backbone, method_key)
        if not rs:
            continue
        plotted_any = True
        ax.plot(rs, ms, color=color, linestyle=ls, linewidth=lw,
                label=label,
                zorder=10 if method_key.startswith("HybridAL") else 5)

    # Annotate the mean switch round of each HybridAL variant.
    sw_alpha = _switch_round_mean(run_meta, backbone, "HybridAL_alpha")
    sw_acc   = _switch_round_mean(run_meta, backbone, "HybridAL_acc")
    if sw_alpha is not None:
        ax.axvline(sw_alpha, color="#d62728", linestyle=":", linewidth=1.0,
                   alpha=0.6, zorder=2)
        ax.annotate(rf"$\bar t^\star_{{\Delta\alpha}}{{=}}{sw_alpha:.1f}$",
                    xy=(sw_alpha, 1.0), xytext=(2, -10), textcoords="offset points",
                    color="#d62728",
                    fontsize=plt.rcParams["axes.labelsize"] * 0.75,
                    xycoords=("data", "axes fraction"),
                    ha="left", va="top")
    if sw_acc is not None:
        ax.axvline(sw_acc, color="#ff7f0e", linestyle=":", linewidth=1.0,
                   alpha=0.6, zorder=2)
        ax.annotate(rf"$\bar t^\star_{{\Delta\mathrm{{Acc}}}}{{=}}{sw_acc:.1f}$",
                    xy=(sw_acc, 1.0), xytext=(2, -28), textcoords="offset points",
                    color="#ff7f0e",
                    fontsize=plt.rcParams["axes.labelsize"] * 0.75,
                    xycoords=("data", "axes fraction"),
                    ha="left", va="top")

    if not plotted_any:
        return False

    ax.set_title(backbone, fontweight="bold")
    ax.set_xlabel("acquisition round $t$")
    ax.set_ylabel("mean val F1")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10, linestyle=":")
    ax.legend(loc="lower right", framealpha=0.95,
              handletextpad=0.4, fontsize=plt.rcParams["axes.labelsize"] * 0.78)
    return True


def main():
    apply_acl_style(scale=1.45)
    run_meta, per_round = _load()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for b in BACKBONES:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ok = _draw(ax, run_meta, per_round, b)
        if not ok:
            plt.close(fig)
            print(f"  skipping {b}: no data")
            continue
        fig.tight_layout()
        out = OUT_DIR / f"switch_evidence_{b.lower()}.png"
        fig.savefig(out)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)
        print(f"wrote {out.relative_to(REPO_ROOT)}")
        print(f"wrote {out.with_suffix('.pdf').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
