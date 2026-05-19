"""Hyperparameter-sensitivity figure: 2x4 grid of line plots.

For each signal ($\\Delta\\alpha$, $\\Delta$Acc) we draw four panels:

  Two NLL panels: x = k (ε fixed to ε*) and x = ε (k fixed to k*),
                  y = mean val NLL; point color = switch rate.
  Two time panels: the dual sweeps for mean training time.

Best $(\\varepsilon^\\star, k^\\star)$ comes from the existing
`experiments/hyperparameter_tuning/chosen_eps_k_<signal>.json` files —
we just read them rather than re-deriving.

Output:
  experiments/hyperparameter_tuning/plots/hyperparameter_sweep.{png,pdf}

Run as a module:
  python -m experiments_lib.plotting.hyperparameter_sweep
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean, stdev as _stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from .style import apply_acl_style


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR   = REPO_ROOT / "experiments" / "hyperparameter_tuning"
SUMMARY   = EXP_DIR / "_summary.csv"
OUT       = EXP_DIR / "plots" / "hyperparameter_sweep.png"

SIGNALS = [
    ("delta_spectral_alpha", r"$\Delta\alpha$"),
    ("delta_accuracy",       r"$\Delta$Acc"),
]


def _fmt_eps_tick(e):
    """Compact tick label for an ε value. Uses 1-decimal mantissa so
    1.5e-4 stays distinct from 1e-4 (Python's `:.0e` rounds them to the
    same string due to binary-float representation of 0.00015)."""
    if e < 1e-3:
        s = f"{e:.1e}"  # e.g. "1.5e-04"
        mant, exp = s.split("e")
        if mant.endswith(".0"):
            mant = mant[:-2]
        exp = int(exp)
        return rf"${mant}{{\times}}10^{{{exp}}}$"
    return f"{e:g}"


def load_chosen(signal):
    p = EXP_DIR / f"chosen_eps_k_{signal}.json"
    d = json.loads(p.read_text())
    return float(d["epsilon"]), int(d["k"])


def load_runs():
    """Return cells[(signal, eps, k)] = list of (val_nll, time, switched)."""
    cells = defaultdict(list)
    with SUMMARY.open() as fh:
        for r in csv.DictReader(fh):
            sig = r["signal"]
            try:
                eps = float(r["epsilon"]); k = int(r["k"])
            except ValueError:
                continue
            nll = r.get("final_val_loss")
            tm = r.get("training_time_total")
            if not nll or not tm:
                continue
            switched = bool(r.get("switch_round"))
            cells[(sig, eps, k)].append((float(nll), float(tm), switched))
    return cells


def _agg(cell_runs, metric):
    """metric is 'nll' or 'time' — picks the right scalar from each tuple."""
    if not cell_runs:
        return None
    if metric == "nll":
        vals = [t[0] for t in cell_runs]
    else:
        vals = [t[1] for t in cell_runs]
    sw_rate = sum(int(t[2]) for t in cell_runs) / len(cell_runs)
    return {
        "mean": _mean(vals),
        "std":  _stdev(vals) if len(vals) > 1 else 0.0,
        "rate": sw_rate,
        "n":    len(vals),
    }


def _draw_sweep(ax, xs, vals, x_label, x_is_eps, best_x, sm, title):
    """Plot one sweep (k or ε)."""
    if not xs:
        ax.set_title(f"{title}\n(no data)"); ax.axis("off"); return
    means = [v["mean"] for v in vals]
    rates = [v["rate"] for v in vals]
    ax.plot(xs, means, color="gray", linewidth=1.2, alpha=0.6, zorder=2)
    # Plot non-chosen as circles, chosen as a star — both coloured by
    # switch rate via the shared scalar mappable.
    other_x, other_y, other_c = [], [], []
    chosen_x, chosen_y, chosen_c = [], [], []
    for x, m, r in zip(xs, means, rates):
        if x == best_x:
            chosen_x.append(x); chosen_y.append(m); chosen_c.append(r)
        else:
            other_x.append(x); other_y.append(m); other_c.append(r)
    if other_x:
        ax.scatter(other_x, other_y, c=other_c, cmap=sm.cmap, norm=sm.norm,
                   s=200, marker="o", edgecolor="black", linewidths=1.0,
                   zorder=5)
    if chosen_x:
        ax.scatter(chosen_x, chosen_y, c=chosen_c, cmap=sm.cmap, norm=sm.norm,
                   s=600, marker="*", edgecolor="black", linewidths=1.4,
                   zorder=6)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    if x_is_eps:
        ax.set_xticks(xs)
        ax.set_xticklabels([_fmt_eps_tick(x) for x in xs],
                           rotation=30, ha="right",
                           rotation_mode="anchor")
        # Pad x-range so the leftmost / rightmost markers don't sit on the
        # axis spines.
        span = max(xs) - min(xs)
        ax.set_xlim(min(xs) - 0.10 * span, max(xs) + 0.10 * span)
    else:
        ax.set_xticks(xs)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, which="major", alpha=0.30)


def main():
    # Match the main-paper signal-ablation figure: oversized canvas +
    # scale=1.90 so LaTeX `width=\textwidth` (6.30 in) leaves labels
    # near 10 pt at print size.
    apply_acl_style(scale=2.20)
    cells = load_runs()

    fig = plt.figure(figsize=(15.5, 8.5))
    gs = GridSpec(2, 4, figure=fig,
                  hspace=1.10, wspace=0.40,
                  left=0.07, right=0.96, top=0.88, bottom=0.09)
    sm = matplotlib.cm.ScalarMappable(
        cmap="viridis",
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0),
    )
    sm.set_array([])

    # Per-signal aggregation for both metrics.
    panel_data = []
    for sig_key, sig_disp in SIGNALS:
        eps_star, k_star = load_chosen(sig_key)
        ks = sorted({k for (s, e, k) in cells if s == sig_key
                     and abs(e - eps_star) < 1e-12})
        es = sorted({e for (s, e, k) in cells if s == sig_key
                     and k == k_star})
        per = {}
        for metric in ("nll", "time"):
            per[metric] = {
                "k": [_agg(cells[(sig_key, eps_star, k)], metric) for k in ks],
                "e": [_agg(cells[(sig_key, e, k_star)],   metric) for e in es],
            }
        panel_data.append((sig_disp, eps_star, k_star, ks, es, per))

    # Per-metric shared y-range across the two signals.
    y_range = {}
    for metric in ("nll", "time"):
        means = []
        for _, _, _, _, _, per in panel_data:
            for v in per[metric]["k"] + per[metric]["e"]:
                if v is not None:
                    means.append(v["mean"])
        if means:
            pad = 0.02 if metric == "nll" else 30.0
            y_range[metric] = (min(means) - pad, max(means) + pad)

    metric_labels = {"nll": "mean val NLL", "time": "mean time (s)"}
    metric_titles = {"nll": "NLL", "time": "time"}

    # Layout: 2 rows × 4 cols. Each row is one signal, with 4 panels:
    # NLL vs k | NLL vs ε | time vs k | time vs ε.
    for sig_idx, (sig_disp, eps_star, k_star,
                  ks, es, per) in enumerate(panel_data):
        for met_idx, metric in enumerate(("nll", "time")):
            col_offset = met_idx * 2
            ax_k = fig.add_subplot(gs[sig_idx, col_offset])
            ax_e = fig.add_subplot(gs[sig_idx, col_offset + 1])
            _draw_sweep(
                ax_k, ks, per[metric]["k"], x_label="$k$",
                x_is_eps=False, best_x=k_star, sm=sm,
                title=f"{metric_titles[metric]} vs $k$",
            )
            _draw_sweep(
                ax_e, es, per[metric]["e"], x_label="$\\varepsilon$",
                x_is_eps=True, best_x=eps_star, sm=sm,
                title=f"{metric_titles[metric]} vs $\\varepsilon$",
            )
            if metric in y_range:
                ax_k.set_ylim(*y_range[metric])
                ax_e.set_ylim(*y_range[metric])
            ax_k.set_ylabel(metric_labels[metric])

    # Shared row headers: signal name centred horizontally above each
    # row's 4 panels. Computed from the leftmost and rightmost panels
    # so the label stays centred if margins / wspace change.
    panels_per_row = 4
    for sig_idx, (sig_disp, *_rest) in enumerate(panel_data):
        leftmost  = fig.axes[sig_idx * panels_per_row]
        rightmost = fig.axes[sig_idx * panels_per_row + (panels_per_row - 1)]
        x_center = (leftmost.get_position().x0
                    + rightmost.get_position().x1) / 2
        y_top    = leftmost.get_position().y1
        # Sit a little above the panel titles.
        fig.text(x_center, y_top + 0.06, sig_disp,
                 va="bottom", ha="center",
                 fontsize=plt.rcParams["axes.titlesize"] * 1.30,
                 fontweight="bold")

    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.020, pad=0.025,
                        ticks=[0, 0.5, 1.0])
    cbar.set_label("switch rate")
    cbar.ax.set_yticklabels(["0%", "50%", "100%"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    out_pdf = OUT.with_suffix(".pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"wrote {out_pdf.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
