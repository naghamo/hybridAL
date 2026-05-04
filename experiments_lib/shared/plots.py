"""Matplotlib helpers shared across all aggregator plots."""
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


def dense_ticks(ax, x_majors: int = 12, y_majors: int = 12) -> None:
    """Apply a denser tick + grid styling to a matplotlib axes.
    Mirrors `_aggregate_and_plot._dense_ticks` so plots share a look."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=x_majors))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_majors))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis="both", which="major", labelsize=10, length=6)
    ax.tick_params(axis="both", which="minor", length=3)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12, linestyle=":")


def figsize_for(rows: int, cols: int, base_w: float = 8.0, base_h: float = 6.0):
    """Standard figsize used by per-experiment grids."""
    return (base_w * cols, base_h * rows)


# Color cycle used consistently across aggregator plots.
def signal_colors(signals):
    cmap = plt.get_cmap("tab10")
    return {s: cmap(i % 10) for i, s in enumerate(signals)}
