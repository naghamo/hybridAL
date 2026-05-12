"""Appendix figure: per-round mean validation F1, averaged across datasets.

Single-panel companion to the per-dataset 6-grid (`headline_f1_all`). For
each signal, plot the trajectory of the cross-dataset mean F1: at each
round t, compute the mean F1 across seeds within each dataset, then the
mean of those six per-dataset means (so the easier datasets — Jigsaw,
AG News — don't drown out the harder ones — TweetEval, Yahoo). The
shaded band is the cross-dataset std of the per-dataset means; it is
shown only for Retrain, $\\Delta\\alpha$ and $\\Delta\\mathrm{Acc}$ to
keep the figure readable.

Output:
  experiments/signal_ablation/plots/signal_ablation_mean_f1.{png,pdf}

Run as a module:
  python -m experiments_lib.plotting.signal_ablation_mean_f1
"""
from __future__ import annotations
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean, stdev as _stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from .style import apply_acl_style, SINGLE_COL_TALL


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO_ROOT / "experiments" / "signal_ablation"
OUT = RUNS_DIR / "plots" / "signal_ablation_mean_f1.png"

DATASETS = ("imdb", "agnews", "jigsaw", "sst2", "tweeteval", "yahoo_answers")
SIGNALS = [
    "delta_f1", "delta_accuracy", "delta_loss",
    "gradient_norm", "l2_weight_distance", "cka",
    "delta_spectral_alpha", "delta_nc",
]

# (sig_key, label, color, linestyle, marker, linewidth, zorder)
LINES = [
    ("retrain_only",         "Retrain", "black",   "--", "s", 1.6, 20),
    ("delta_spectral_alpha", "Δα",      "#d62728", "-",  "o", 1.4, 18),
    ("delta_accuracy",       "ΔAcc",    "#1f77b4", ":",  "^", 1.4, 17),
    ("delta_f1",             "ΔF1",     "#17becf", "-.", "v", 1.1, 12),
    ("delta_loss",           "ΔLoss",   "#2ca02c", "-",  "D", 1.1, 11),
    ("delta_nc",             "ΔNC",     "#ff7f0e", "-",  "X", 1.1, 10),
    ("cka",                  "CKA",     "#9467bd", "-",  "P", 1.1,  9),
    ("gradient_norm",        "GN",      "#8c564b", "-",  "*", 1.1,  8),
    ("l2_weight_distance",   "L2",      "#7f7f7f", "-",  "h", 1.1,  7),
]


def _parse_retrain(name):
    m = re.match(r"^Retrain_(.+)_seed(\d+)_N1000$", name)
    return (m.group(1), int(m.group(2))) if m else None


def _parse_hybrid(name):
    if not (name.startswith("DeltaF1_") and "_eps0.5_k3_N1000" in name):
        return None
    rest = name[len("DeltaF1_"):-len("_eps0.5_k3_N1000")]
    m = re.match(r"^(.+)_seed(\d+)$", rest)
    if not m:
        return None
    sig_ds, seed = m.group(1), int(m.group(2))
    for sig in sorted(SIGNALS, key=len, reverse=True):
        if sig_ds.startswith(sig + "_"):
            ds = sig_ds[len(sig) + 1:]
            if ds in DATASETS:
                return sig, ds, seed
    return None


def _collect():
    """Return (data, switches) where
       data[sig][dataset][seed]    = [f1 per round]
       switches[sig]                = flat list of switch rounds across all
                                      (dataset, seed) where the signal fired.
    """
    data = {}
    switches = defaultdict(list)
    for jp in RUNS_DIR.glob("*/results_*.json"):
        name = jp.parent.name
        ph = _parse_hybrid(name)
        pr = _parse_retrain(name) if ph is None else None
        if ph is None and pr is None:
            continue
        d = json.loads(jp.read_text())
        rounds = d.get("round_val_stats", []) or []
        f1s = [r.get("f1_score") for r in rounds]
        if ph is not None:
            sig, ds, seed = ph
            data.setdefault(sig, {}).setdefault(ds, {})[seed] = f1s
            sw = (d.get("strategy_metadata") or {}).get("switch_round")
            if sw:
                switches[sig].append(sw)
        else:
            ds, seed = pr
            data.setdefault("retrain_only", {}).setdefault(ds, {})[seed] = f1s
    return data, switches


def _per_dataset_round_mean(seed_dict):
    by_round = defaultdict(list)
    for seq in seed_dict.values():
        for i, v in enumerate(seq, 1):
            if v is None:
                continue
            try:
                if not math.isfinite(v):
                    continue
            except TypeError:
                continue
            by_round[i].append(v)
    return {r: _mean(vals) for r, vals in by_round.items()}


def _aggregate(sig_data):
    """sig_data[ds][seed] = [f1 per round].
    Equal-weight per dataset → mean of the six per-dataset means at each
    round. Std reflects cross-dataset spread."""
    per_ds = {ds: _per_dataset_round_mean(sd) for ds, sd in sig_data.items()}
    all_rounds = set()
    for d in per_ds.values():
        all_rounds.update(d)
    rs = sorted(all_rounds)
    rs_keep, ms, stds = [], [], []
    for r in rs:
        vals = [d[r] for d in per_ds.values() if r in d]
        if len(vals) < len(per_ds):
            continue
        rs_keep.append(r)
        ms.append(_mean(vals))
        stds.append(_stdev(vals) if len(vals) > 1 else 0.0)
    return rs_keep, ms, stds


def main():
    apply_acl_style(scale=1.0)
    data, switches = _collect()

    fig, ax = plt.subplots(figsize=SINGLE_COL_TALL)
    for sig, label, color, ls, marker, lw, zorder in LINES:
        sig_data = data.get(sig, {})
        if not sig_data:
            continue
        rs, ms, _ = _aggregate(sig_data)
        if not rs:
            continue
        ax.plot(rs, ms, color=color, linestyle=ls, linewidth=lw,
                marker=marker, markersize=3.0, label=label, zorder=zorder)
        if sig != "retrain_only" and switches.get(sig):
            sw_int = int(round(_mean(switches[sig])))
            if sw_int in rs:
                yval = ms[rs.index(sw_int)]
                ax.plot([sw_int], [yval], marker="*", color=color,
                        markersize=10, markeredgecolor="black",
                        markeredgewidth=0.5, linestyle="", zorder=50)

    ax.set_xlabel("round t")
    ax.set_ylabel("mean val F1")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10, linestyle=":")
    ax.legend(loc="lower right", framealpha=0.95, ncol=3,
              handletextpad=0.3, columnspacing=0.6,
              fontsize=6.5, borderpad=0.3, labelspacing=0.25)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    out_pdf = OUT.with_suffix(".pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"wrote {out_pdf.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
