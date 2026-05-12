"""3-panel NLL trajectory figure for the main paper.

Combines what main_results_finetune_disadvantage.py renders separately
per backbone into one side-by-side figure (DistilBERT | BERT | RoBERTa).
Each panel shows mean validation NLL per acquisition round for
Retrain / FineTune / HybridAL Δα / HybridAL ΔAcc.

Output:
  experiments/main_results/plots/main_results_nll_3panel.{png,pdf}
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
OUT = REPO_ROOT / "experiments" / "main_results" / "plots" / "main_results_nll_3panel.png"

DATASETS = ["imdb", "agnews", "jigsaw", "sst2", "tweeteval", "yahoo_answers"]
BACKBONES = ["DistilBERT", "BERT", "RoBERTa"]

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
            try: sw_v = int(sw) if sw and sw.strip() else None
            except (TypeError, ValueError): sw_v = None
            run_meta[run] = (r.get("method"), r.get("backbone"), r.get("data"), seed, sw_v)

    per_round = defaultdict(list)
    with PER_ROUND.open() as fh:
        for r in csv.DictReader(fh):
            run = r.get("run")
            if run not in run_meta: continue
            try:
                rd = int(r["round"]); loss = float(r["loss"])
            except (TypeError, ValueError): continue
            if not math.isfinite(loss): continue
            per_round[run].append((rd, loss))
    for run in per_round: per_round[run].sort()
    return run_meta, per_round


def _agg(run_meta, per_round, backbone, method_key):
    by_round = defaultdict(list)
    for run, recs in per_round.items():
        m, b, d, _seed, _sw = run_meta[run]
        if b != backbone or m != method_key or d not in DATASETS: continue
        for rd, loss in recs:
            by_round[rd].append(loss)
    if not by_round: return [], []
    rs = sorted(by_round)
    return rs, [_mean(by_round[r]) for r in rs]


def _switch_round_mean(run_meta, backbone, method_key):
    sws = [sw for run, (m, b, d, _seed, sw) in run_meta.items()
           if b == backbone and m == method_key and d in DATASETS and sw is not None]
    return _mean(sws) if sws else None


def _draw_panel(ax, run_meta, per_round, backbone, show_ylabel, show_legend):
    plotted = False
    for method_key, label, color, ls, lw in LINES:
        rs, ms = _agg(run_meta, per_round, backbone, method_key)
        if not rs: continue
        plotted = True
        ax.plot(rs, ms, color=color, linestyle=ls, linewidth=lw,
                label=label,
                zorder=10 if method_key.startswith("HybridAL") else 5)
    if not plotted:
        ax.text(0.5, 0.5, f"{backbone}: no data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, alpha=0.6)
        ax.set_title(backbone, fontweight="bold")
        return

    sw_alpha = _switch_round_mean(run_meta, backbone, "HybridAL_alpha")
    sw_acc   = _switch_round_mean(run_meta, backbone, "HybridAL_acc")
    if sw_alpha is not None:
        ax.axvline(sw_alpha, color="#d62728", linestyle=":", linewidth=1.0,
                   alpha=0.6, zorder=2)
        ax.annotate(rf"$\bar t^\star_{{\Delta\alpha}}{{=}}{sw_alpha:.1f}$",
                    xy=(sw_alpha, 0.02), xytext=(2, 0),
                    textcoords="offset points", color="#d62728",
                    fontsize=plt.rcParams["axes.labelsize"] * 0.70,
                    xycoords=("data", "axes fraction"),
                    ha="left", va="bottom")
    if sw_acc is not None:
        ax.axvline(sw_acc, color="#ff7f0e", linestyle=":", linewidth=1.0,
                   alpha=0.6, zorder=2)
        ax.annotate(rf"$\bar t^\star_{{\Delta\mathrm{{Acc}}}}{{=}}{sw_acc:.1f}$",
                    xy=(sw_acc, 0.10), xytext=(2, 0),
                    textcoords="offset points", color="#ff7f0e",
                    fontsize=plt.rcParams["axes.labelsize"] * 0.70,
                    xycoords=("data", "axes fraction"),
                    ha="left", va="bottom")

    ax.set_title(backbone, fontweight="bold")
    ax.set_xlabel("acquisition round $t$")
    if show_ylabel:
        ax.set_ylabel("mean val NLL")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10, linestyle=":")
    if show_legend:
        ax.legend(loc="upper left", framealpha=0.95,
                  handletextpad=0.4,
                  fontsize=plt.rcParams["axes.labelsize"] * 0.72)


def main():
    apply_acl_style(scale=1.55)
    run_meta, per_round = _load()

    fig, axes = plt.subplots(1, len(BACKBONES), figsize=(13.0, 4.4))
    for i, b in enumerate(BACKBONES):
        _draw_panel(axes[i], run_meta, per_round, b,
                    show_ylabel=(i == 0), show_legend=(i == 0))

    fig.suptitle("Validation NLL per round — FineTune drifts up, Retrain stays flat",
                 fontweight="bold", y=1.00,
                 fontsize=plt.rcParams["axes.titlesize"] * 1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT.with_suffix('.pdf').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
