"""Per-dataset line plots — F1 vs rounds AND F1 vs cumulative time.

Two figures, each a 2×3 grid (one panel per dataset). The grid order is:

  Row 1 : IMDb     | SST-2     | AG News      (binary / 4-class)
  Row 2 : Jigsaw   | TweetEval | Yahoo        (multi-class / harder)

Per panel:
  X = round t   (figure 1) OR  cumulative training time in seconds (figure 2)
  Y = mean validation F1 (macro) across the seeds available for that cell
  One line per method (8 total); see METHODS below for the colour / style key.

DistilBERT is used as the canonical backbone (most complete data); methods
without runs on a (backbone, dataset) cell are silently skipped.

NewOnly is currently excluded from the plots — the existing NewOnly cells
were produced by the round-1 warm-up implementation and are archived under
`experiments/main_results/_archive_newonly_warmup/`. Re-add the entry once
the watchdog-triggered re-runs land.

Outputs:
  experiments/main_results/plots/lines_per_round.{png,pdf}
  experiments/main_results/plots/lines_per_time.{png,pdf}
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
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from .style import apply_acl_style


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY = REPO_ROOT / "experiments" / "main_results" / "_summary.csv"
PER_ROUND = REPO_ROOT / "experiments" / "main_results" / "_per_round.csv"
OUT_DIR = REPO_ROOT / "experiments" / "main_results" / "plots"

DATASETS = ["imdb", "agnews", "jigsaw", "sst2", "tweeteval", "yahoo_answers"]
DS_DISP = {
    "imdb": "IMDb", "agnews": "AG News", "jigsaw": "Jigsaw",
    "sst2": "SST-2", "tweeteval": "TweetEval", "yahoo_answers": "Yahoo",
}
GRID_ORDER = ["imdb", "sst2", "agnews", "jigsaw", "tweeteval", "yahoo_answers"]
BACKBONE = "DistilBERT"

# (method_key, display_label, color, linestyle, linewidth, marker, markersize)
# Kept to 5 lines per panel to reduce visual clutter. FixedSwitch_3 is the
# representative of the FixedSwitch family (its sweep across {3,5,7,10}
# lives in the per-dataset Pareto plot); add the others back here if a
# specific FixedSwitch story is needed.
# NewOnly excluded until the round-1-warmup fix re-runs complete.
METHODS = [
    ("Retrain",         "Retrain",                    "#000000", "--", 1.6, "s", 0),
    ("FineTune",        "FineTune",                   "#1f77b4", ":",  1.4, "^", 0),
    ("HybridAL_alpha",  r"HybridAL ($\Delta\alpha$)", "#d62728", "-",  2.0, "o", 0),
    ("HybridAL_acc",    r"HybridAL ($\Delta$Acc)",    "#ff7f0e", "-",  2.0, "o", 0),
    ("FixedSwitch_3",   "FixedSwitch (3)",            "#9467bd", "-.", 1.2, "P", 0),
]


def _load():
    """Build a per-(method, dataset, seed) trajectory of (rounds, f1s, cum_time).

    Method/backbone are looked up via _summary.csv (which has the
    cfg-derived `method` and `backbone` columns).  We then walk
    _per_round.csv and bucket rows by run.
    """
    run_meta = {}                                 # run -> (method, backbone, dataset, seed)
    with SUMMARY.open() as fh:
        for r in csv.DictReader(fh):
            run = r.get("run")
            if not run:
                continue
            run_meta[run] = (r.get("method"), r.get("backbone"),
                              r.get("data"),   r.get("seed"))

    # per_round[(method, dataset, seed)] = [(round, f1, training_time), ...]
    per_round = defaultdict(list)
    with PER_ROUND.open() as fh:
        for r in csv.DictReader(fh):
            meta = run_meta.get(r["run"])
            if not meta:
                continue
            method, backbone, dataset, seed = meta
            if backbone != BACKBONE or not method or dataset not in DATASETS:
                continue
            try:
                rd = int(r["round"])
                f1 = float(r["f1_score"])
                tt = float(r["training_time"])
            except (TypeError, ValueError):
                continue
            per_round[(method, dataset, seed)].append((rd, f1, tt))

    # Sort each trajectory by round, then materialise cumulative time.
    out = defaultdict(list)
    for key, recs in per_round.items():
        recs.sort(key=lambda x: x[0])
        rounds, f1s, ts = [], [], []
        cum = 0.0
        for rd, f1, tt in recs:
            cum += tt
            rounds.append(rd); f1s.append(f1); ts.append(cum)
        out[key] = (rounds, f1s, ts)
    return out


def _aggregate_across_seeds(per_round, method, dataset, x_axis):
    """Return (xs, mean_f1) where xs is a sorted list of round indices
    (x_axis='round') or per-round mean cumulative time (x_axis='time').

    For x_axis='time': we compute, at each round t, the mean F1 over
    seeds that reached round t and the mean cumulative time over those
    same seeds. Round indices align across seeds (deterministic 25-round
    schedule), which is what makes this aggregation well-defined.
    """
    seeds_data = {seed: per_round[(method, dataset, seed)]
                  for (m, d, seed) in per_round
                  if m == method and d == dataset and per_round[(m, d, seed)][0]}
    if not seeds_data:
        return [], []

    # Stack by round.
    by_round_f1 = defaultdict(list)
    by_round_t  = defaultdict(list)
    for seed, (rounds, f1s, ts) in seeds_data.items():
        for rd, f1, t in zip(rounds, f1s, ts):
            by_round_f1[rd].append(f1)
            by_round_t[rd].append(t)

    rs = sorted(by_round_f1)
    means_f1 = [_mean(by_round_f1[r]) for r in rs]
    means_t  = [_mean(by_round_t[r])  for r in rs]
    if x_axis == "round":
        return rs, means_f1
    else:
        return means_t, means_f1


def _draw_grid(per_round, x_axis, x_label, out_path):
    apply_acl_style(scale=1.50)

    fig = plt.figure(figsize=(13.6, 7.5))
    gs = GridSpec(2, 3, figure=fig,
                  hspace=0.45, wspace=0.30,
                  left=0.07, right=0.97, top=0.93, bottom=0.16)

    legend_handles = None
    for idx, ds in enumerate(GRID_ORDER):
        r, c = divmod(idx, 3)
        ax = fig.add_subplot(gs[r, c])

        plotted_any = False
        for key, label, color, ls, lw, marker, msz in METHODS:
            xs, ys = _aggregate_across_seeds(per_round, key, ds, x_axis)
            if not xs:
                continue
            plotted_any = True
            ax.plot(xs, ys, color=color, linestyle=ls, linewidth=lw,
                    marker=marker, markersize=msz, label=label,
                    zorder=10 if key.startswith("HybridAL") else 5)

        if not plotted_any:
            ax.set_title(f"{DS_DISP[ds]}\n(no data)")
            ax.axis("off")
            continue
        ax.set_title(DS_DISP[ds], fontweight="bold")
        if r == 1:
            ax.set_xlabel(x_label)
        if c == 0:
            ax.set_ylabel("mean val F1")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=(x_axis == "round")))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(True, which="major", alpha=0.30)
        ax.grid(True, which="minor", alpha=0.10, linestyle=":")

        if legend_handles is None:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend_handles = (handles, labels)

    if legend_handles is not None:
        handles, labels = legend_handles
        fig.legend(handles, labels,
                   loc="lower center", bbox_to_anchor=(0.5, 0.01),
                   ncol=4, framealpha=0.95,
                   handletextpad=0.5, columnspacing=1.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    out_pdf = out_path.with_suffix(".pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    print(f"wrote {out_pdf.relative_to(REPO_ROOT)}")


def main():
    per_round = _load()
    _draw_grid(per_round, x_axis="round",
               x_label="round t",
               out_path=OUT_DIR / "lines_per_round.png")
    _draw_grid(per_round, x_axis="time",
               x_label="cumulative training time (s)",
               out_path=OUT_DIR / "lines_per_time.png")


if __name__ == "__main__":
    main()
