"""Exp 0 — N (random_subset_size) sensitivity plot.

Reads the legacy subset_sensitivity runs (Retrain + FineTune × {imdb,
sst2} × 8 N values × 3 seeds = 48 + 48 = 96 runs), aggregates per
(strategy, dataset, N), and renders a single-column EMNLP-style figure
with two stacked panels (one per dataset) and a single shared legend
below both panels. A vertical dashed line marks the chosen N (default
1000).

Output:
  experiments/n_sensitivity/plots/n_sensitivity.{png,pdf}

Run as a module:
  python -m experiments_lib.plotting.n_sensitivity
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from .style import apply_acl_style, SINGLE_COL_STACKED_2
from .utils import (
    discover_runs,
    run_summary,
    display_dataset,
)


# Where the legacy subset-sensitivity runs live
SOURCE_ROOT = Path("experiments/subset_sensitivity")
OUT = Path("experiments/n_sensitivity/plots/n_sensitivity.png")

# Datasets shown (in display order, top → bottom)
DATASETS = ["imdb", "sst2"]

# The chosen N value the rest of the paper uses.
CHOSEN_N = 1000

# (strategy_class, label, color, linestyle, marker, zorder)
STRATEGIES = [
    ("RetrainStrategy",  "Retrain",  "#1f77b4", "-",  "o", 5),   # blue
    ("FineTuneStrategy", "FineTune", "#ff7f0e", "--", "s", 4),   # orange
]


def _parse_subset_dir(dir_name: str):
    """`subset_1000` → 1000, `subset_full` → "full". Returns None on miss."""
    if not dir_name.startswith("subset_"):
        return None
    rest = dir_name[len("subset_"):]
    if rest == "full":
        return "full"
    try:
        return int(rest)
    except ValueError:
        return None


def collect():
    """Return `data[strategy][dataset] = {N: [test_f1 across seeds]}`."""
    data = {strat: {ds: defaultdict(list) for ds in DATASETS}
            for strat, *_ in STRATEGIES}
    if not SOURCE_ROOT.exists():
        return data
    valid_strategies = {s for s, *_ in STRATEGIES}
    for subset_dir in sorted(SOURCE_ROOT.iterdir()):
        if not subset_dir.is_dir():
            continue
        N = _parse_subset_dir(subset_dir.name)
        if N is None:
            continue
        for jp in subset_dir.glob("*/results_*.json"):
            row = run_summary(jp)
            strat = row.get("strategy_class")
            if strat not in valid_strategies:
                continue
            ds = row.get("data")
            f1 = row.get("test_f1")
            if ds in data[strat] and f1 is not None:
                data[strat][ds][N].append(f1)
    return data


def _draw_panel(ax, dataset: str, all_strategies_data: dict):
    """Plot N vs mean F1 ± std for one dataset, one line per strategy."""
    # Build the full set of N values from BOTH strategies so the x-axis
    # is consistent across them.
    all_Ns: set = set()
    for strat, *_ in STRATEGIES:
        all_Ns.update(all_strategies_data[strat][dataset].keys())
    numeric_Ns = sorted(n for n in all_Ns if isinstance(n, int))
    has_full = "full" in all_Ns
    Ns_for_x = numeric_Ns + (["full"] if has_full else [])
    max_N = max(numeric_Ns) if numeric_Ns else 1
    # Map each N to an x-coordinate; "full" sits at 2 * max_N so it appears
    # just to the right of the largest numeric N on a log scale.
    x_of = {n: (n if isinstance(n, int) else 2 * max_N) for n in Ns_for_x}

    for strat, label, color, ls, marker, zorder in STRATEGIES:
        ds_data = all_strategies_data[strat][dataset]
        Ns_present = [n for n in Ns_for_x if n in ds_data]
        if not Ns_present:
            continue
        xs    = [x_of[n] for n in Ns_present]
        means = [mean(ds_data[n]) for n in Ns_present]
        stds  = [stdev(ds_data[n]) if len(ds_data[n]) > 1 else 0.0
                 for n in Ns_present]
        upper = [m + s for m, s in zip(means, stds)]
        lower = [m - s for m, s in zip(means, stds)]
        ax.fill_between(xs, lower, upper, color=color, alpha=0.15,
                         linewidth=0, zorder=zorder - 1)
        ax.plot(xs, means, color=color, linestyle=ls, linewidth=1.2,
                marker=marker, markersize=3.5, label=label, zorder=zorder)

    # Vertical line at the chosen N (label kept short — value is implicit
    # from the line's x-position on the axis).
    if CHOSEN_N in numeric_Ns:
        ax.axvline(CHOSEN_N, color="#d62728", linestyle=":", linewidth=1.0,
                   alpha=0.8, label="chosen $N$")

    ax.set_xscale("log")
    ax.set_title(display_dataset(dataset), fontweight="bold")
    ax.set_ylabel("test F1 (macro)")
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10, linestyle=":")
    # X-tick labels — use the actual N values; "full" gets its own label.
    ax.set_xticks([x_of[n] for n in Ns_for_x])
    ax.set_xticklabels(
        [str(n) if isinstance(n, int) else "full" for n in Ns_for_x],
        rotation=0,
    )
    ax.minorticks_off()


def main():
    apply_acl_style()
    data = collect()

    if not any(data[strat][ds] for strat, *_ in STRATEGIES for ds in DATASETS):
        raise SystemExit(
            f"No Retrain or FineTune runs found under {SOURCE_ROOT}. "
            "Make sure the legacy subset_sensitivity data is present."
        )

    # Two panels stacked vertically (single column, EMNLP appendix shape).
    # Wide hspace so the rotated x-tick labels of the top panel + the
    # title of the bottom panel + the shared legend in the middle don't
    # crowd each other.
    fig, axes = plt.subplots(
        2, 1, figsize=SINGLE_COL_STACKED_2, sharex=False,
        gridspec_kw={"hspace": 0.95},
    )
    for ax, ds in zip(axes, DATASETS):
        _draw_panel(ax, ds, data)
    # Rotate tick labels on every panel so consecutive N values don't overlap
    # on the narrow single-column x-axis.
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    axes[-1].set_xlabel(r"$N$ — random subset size")

    # Single shared legend in the middle, between the two panels.
    # We use the *tight* bbox of each panel (which includes its tick
    # labels and title) so the legend sits in the truly-empty gap, not
    # inside the rotated x-tick labels of the top panel or the title of
    # the bottom panel.
    fig.canvas.draw()           # force layout to compute extents
    inv = fig.transFigure.inverted()
    top_tb = axes[0].get_tightbbox().transformed(inv)
    bot_tb = axes[1].get_tightbbox().transformed(inv)
    gap_mid_y = (top_tb.y0 + bot_tb.y1) / 2.0
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="center", bbox_to_anchor=(0.5, gap_mid_y),
               ncol=3, framealpha=0.95, handletextpad=0.4,
               columnspacing=1.2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    out_pdf = OUT.with_suffix(".pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {OUT}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
