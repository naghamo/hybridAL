"""Per-round NLL trajectory — evidence for "don't stay with FineTune".

The F1 trajectories of Retrain and FineTune are tied within seed noise,
so an F1-only plot does *not* show why we should retrain in early rounds.
The picture is much sharper on the *test-loss* (NLL) axis: warm-started
FineTune is consistently miscalibrated relative to Retrain across every
round, even when its argmax-F1 is competitive (Ash \& Adams 2020).

Per backbone, this script renders one figure with:

  - Retrain (black, dashed): low NLL across all rounds (well-calibrated).
  - FineTune (blue, dotted): elevated NLL throughout (warm-start drift
    accumulates from round 1 onwards and never recovers).
  - HybridAL Δα and Δacc (red / orange, solid): track Retrain in early
    rounds; only after the marked switch round does NLL drift slightly
    toward FineTune's territory — the trade-off is bounded, not
    unbounded.

Vertical lines mark each HybridAL variant's mean switch round.

Output:
  experiments/main_results/plots/finetune_disadvantage_<backbone>.{png,pdf}
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

# (method_key, label, color, linestyle, linewidth)
LINES = [
    ("Retrain",         "Retrain",                    "#000000", "--", 1.6),
    ("FineTune",        "FineTune",                   "#1f77b4", ":",  1.4),
    ("HybridAL_alpha",  r"HybridAL ($\Delta\alpha$)", "#d62728", "-",  2.0),
    ("HybridAL_acc",    r"HybridAL ($\Delta$Acc)",    "#ff7f0e", "-",  2.0),
]


def _load():
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
                loss = float(r["loss"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(loss): continue
            per_round[run].append((rd, loss))
    for run in per_round:
        per_round[run].sort()
    return run_meta, per_round


def _agg(run_meta, per_round, backbone, method_key):
    by_round = defaultdict(list)
    for run, recs in per_round.items():
        m, b, d, _seed, _sw = run_meta[run]
        if b != backbone or m != method_key or d not in DATASETS:
            continue
        for rd, loss in recs:
            by_round[rd].append(loss)
    if not by_round: return [], []
    rs = sorted(by_round)
    return rs, [_mean(by_round[r]) for r in rs]


def _switch_round_mean(run_meta, backbone, method_key):
    sws = [sw for run, (m, b, d, _seed, sw) in run_meta.items()
           if b == backbone and m == method_key and d in DATASETS and sw is not None]
    return _mean(sws) if sws else None


def _draw(ax, run_meta, per_round, backbone):
    plotted_any = False
    for method_key, label, color, ls, lw in LINES:
        rs, ms = _agg(run_meta, per_round, backbone, method_key)
        if not rs: continue
        plotted_any = True
        ax.plot(rs, ms, color=color, linestyle=ls, linewidth=lw,
                label=label,
                zorder=10 if method_key.startswith("HybridAL") else 5)
    if not plotted_any:
        return False

    sw_alpha = _switch_round_mean(run_meta, backbone, "HybridAL_alpha")
    sw_acc   = _switch_round_mean(run_meta, backbone, "HybridAL_acc")
    if sw_alpha is not None:
        ax.axvline(sw_alpha, color="#d62728", linestyle=":", linewidth=1.0,
                   alpha=0.6, zorder=2)
        ax.annotate(rf"$\bar t^\star_{{\Delta\alpha}}{{=}}{sw_alpha:.1f}$",
                    xy=(sw_alpha, 0.02), xytext=(2, 0), textcoords="offset points",
                    color="#d62728",
                    fontsize=plt.rcParams["axes.labelsize"] * 0.75,
                    xycoords=("data", "axes fraction"),
                    ha="left", va="bottom")
    if sw_acc is not None:
        ax.axvline(sw_acc, color="#ff7f0e", linestyle=":", linewidth=1.0,
                   alpha=0.6, zorder=2)
        ax.annotate(rf"$\bar t^\star_{{\Delta\mathrm{{Acc}}}}{{=}}{sw_acc:.1f}$",
                    xy=(sw_acc, 0.10), xytext=(2, 0), textcoords="offset points",
                    color="#ff7f0e",
                    fontsize=plt.rcParams["axes.labelsize"] * 0.75,
                    xycoords=("data", "axes fraction"),
                    ha="left", va="bottom")

    ax.set_title(f"{backbone} — validation NLL per round",
                 fontweight="bold")
    ax.set_xlabel("acquisition round $t$")
    ax.set_ylabel("mean val NLL (lower is better)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10, linestyle=":")
    ax.legend(loc="upper right", framealpha=0.95,
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
            plt.close(fig); print(f"  skipping {b}: no data"); continue
        fig.tight_layout()
        out = OUT_DIR / f"finetune_disadvantage_{b.lower()}.png"
        fig.savefig(out)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)
        print(f"wrote {out.relative_to(REPO_ROOT)}")
        print(f"wrote {out.with_suffix('.pdf').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
