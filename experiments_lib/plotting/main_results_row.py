"""Main-paper signal-ablation row figure.

A single double-column-wide figure that contains:

  Left half  :  the simple-Pareto scatter (8 signals + Retrain),
                F1 vs. corrected total training time, with marker shape
                encoding paired-t-test significance vs. Retrain.
  Right half :  a 2×2 grid: per-round val-F1 trajectories on AG News,
                Yahoo Answers and Jigsaw (3 panels) + a Spearman-rank
                correlation heatmap of the 8 candidate switching signals
                (showing Δα is orthogonal to the standard performance suite).

Output:
  experiments/signal_ablation/plots/main_results_row.{png,pdf}

Run as a module:
  python -m experiments_lib.plotting.main_results_row
"""
from __future__ import annotations
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean as _mean, stdev as _stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator

from .style import apply_acl_style, DOUBLE_COL_TALL


# Per-round CSV from the MAIN_RESULTS experiment (Retrain runs log all 8
# signals via log_all_signals=True). Used to build the correlation heatmap.
MR_SUMMARY   = None  # set in main()
MR_PER_ROUND = None  # set in main()


HEATMAP_SIGNALS = [
    ("sig_delta_f1",              r"$\Delta$F1"),
    ("sig_delta_accuracy",        r"$\Delta$Acc"),
    ("sig_delta_loss",            r"$\Delta$Loss"),
    ("sig_gradient_norm",         "GN"),
    ("sig_l2_weight_distance",    r"L2$\Delta$"),
    ("sig_cka",                   "CKA"),
    ("sig_delta_spectral_alpha",  r"$\Delta\alpha$"),
    ("sig_delta_nc",              r"$\Delta$NC"),
]

# Compact subset used in the main paper (line-plots version): the two
# HybridAL switching signals + ΔF1 as the orthogonality reference.
HEATMAP_SIGNALS_MAIN = [
    ("sig_delta_f1",              r"$\Delta$F1"),
    ("sig_delta_accuracy",        r"$\Delta$Acc"),
    ("sig_delta_spectral_alpha",  r"$\Delta\alpha$"),
]


# ─── paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR  = REPO_ROOT / "experiments" / "signal_ablation"
CORR_CSV  = REPO_ROOT / "scratch" / "_per_run_corrected_times.csv"
OUT       = REPO_ROOT / "experiments" / "signal_ablation" / "plots" / "main_results_row.png"


# ─── shared signal/dataset constants ────────────────────────────────────────
DATASETS = ("imdb", "agnews", "jigsaw", "sst2", "tweeteval", "yahoo_answers")
SIGNALS  = [
    "delta_f1", "delta_accuracy", "delta_loss",
    "gradient_norm", "l2_weight_distance", "cka",
    "delta_spectral_alpha", "delta_nc",
]
RETRAIN = "retrain_only"

DISPLAY_LABELS = {
    "delta_f1":              "ΔF1",
    "delta_accuracy":        "ΔAcc",
    "delta_loss":            "ΔLoss",
    "delta_spectral_alpha":  "Δα",
    "delta_nc":              "ΔNC",
    "gradient_norm":         "GN",
    "l2_weight_distance":    "L2",
    "cka":                   "CKA",
    "retrain_only":          "Retrain",
}

# ─── Pareto-side helpers ────────────────────────────────────────────────────
TIME_SIG_MARKERS = {
    "delta_spectral_alpha":  "***",
    "delta_accuracy":        "***",
    "delta_f1":              "***",
    "delta_nc":              "***",
    "cka":                   "***",
    "delta_loss":            "***",
    "gradient_norm":         "**",
    "l2_weight_distance":    "ns",
    "retrain_only":          "",
}
SIG_MARKER = {
    "***": ("*", 26),
    "**":  ("D", 14),
    "*":   ("o", 17),
    "ns":  ("X", 17),
}
LABEL_OFFSETS = {
    "delta_spectral_alpha":  (10,  9),
    "delta_accuracy":        (10,  9),
    "delta_f1":              (10, -14),
    "cka":                   (10,  9),
    "delta_nc":              (-7, -14),
    "gradient_norm":         (10,  9),
    "delta_loss":            ( 8, -14),
    "l2_weight_distance":    ( 0, 11),
    "retrain_only":          (-8, -22),
}
RIGHT_ANCHORED  = {"delta_nc", "retrain_only"}
CENTER_ANCHORED = {"l2_weight_distance"}


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


def _load_corrected_times():
    if not CORR_CSV.exists():
        raise SystemExit(f"missing {CORR_CSV} — run scratch/_apply_time_fix.py first")
    out = {}
    with CORR_CSV.open() as fh:
        for row in csv.DictReader(fh):
            out[(row["signal"], row["dataset"], int(row["seed"]))] = \
                float(row["corrected_total"])
    return out


def _collect_pareto():
    corrected = _load_corrected_times()
    by_sig = defaultdict(lambda: {"f1": [], "time": [], "switched": 0, "n": 0})
    for jp in RUNS_DIR.glob("*/results_*.json"):
        name = jp.parent.name
        ph = _parse_hybrid(name)
        pr = _parse_retrain(name) if ph is None else None
        if ph is None and pr is None:
            continue
        d = json.loads(jp.read_text())
        rounds = d.get("round_val_stats", []) or []
        f1 = (d.get("final_test_stats") or {}).get("f1_score")
        if ph is not None:
            sig, ds, seed = ph
            t = corrected.get((sig, ds, seed))
            if t is None:
                continue
            sw = (d.get("strategy_metadata") or {}).get("switch_round")
        else:
            ds, seed = pr
            sig = RETRAIN
            t = sum((r.get("training_time") or 0) for r in rounds)
            sw = None
        cell = by_sig[sig]
        cell["n"] += 1
        if f1 is not None: cell["f1"].append(f1)
        if t: cell["time"].append(t)
        if sw: cell["switched"] += 1
    return by_sig


# ─── Headline-side helpers ──────────────────────────────────────────────────
LINES = [
    # (signal, label, color, linestyle, marker, zorder)
    ("retrain_only",         "Retrain", "black",   "--", "s", 11),
    ("delta_spectral_alpha", "Δα",      "#d62728", "-",  "o",  9),
    ("delta_accuracy",       "ΔAcc",    "#1f77b4", ":",  "^",  8),
]


def _per_round_mean(seed_to_seq):
    """Return rounds, means, stds, ns across the per-seed lists."""
    by_round = defaultdict(list)
    for seq in seed_to_seq.values():
        for i, v in enumerate(seq, 1):
            if v is None:
                continue
            try:
                if not math.isfinite(v):
                    continue
            except TypeError:
                continue
            by_round[i].append(v)
    if not by_round:
        return [], [], [], []
    rs = sorted(by_round)
    means = [_mean(by_round[r]) for r in rs]
    stds  = [_stdev(by_round[r]) if len(by_round[r]) > 1 else 0.0 for r in rs]
    ns    = [len(by_round[r]) for r in rs]
    return rs, means, stds, ns


def _collect_headline():
    data = {ds: defaultdict(dict) for ds in DATASETS}
    switches = {ds: defaultdict(list) for ds in DATASETS}
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
            data[ds][sig][seed] = f1s
            sw = (d.get("strategy_metadata") or {}).get("switch_round")
            if sw:
                switches[ds][sig].append(sw)
        else:
            ds, seed = pr
            data[ds]["retrain_only"][seed] = f1s
    return data, switches


# ─── Heatmap helpers (Spearman correlation between switching signals) ─────
def _load_heatmap_samples():
    """Load (run × round) samples of all 8 signals from MAIN_RESULTS
    Retrain DistilBERT runs (clean reference; same training regime)."""
    summary_csv = REPO_ROOT / "experiments" / "main_results" / "_summary.csv"
    per_round_csv = REPO_ROOT / "experiments" / "main_results" / "_per_round.csv"
    if not (summary_csv.exists() and per_round_csv.exists()):
        return []
    meta = {}
    with summary_csv.open() as f:
        for r in csv.DictReader(f):
            meta[r["run"]] = (r.get("method"), r.get("backbone"))
    samples = []
    with per_round_csv.open() as f:
        for r in csv.DictReader(f):
            m_b = meta.get(r["run"])
            if not m_b: continue
            m, b = m_b
            if m != "Retrain" or b != "DistilBERT": continue
            row, ok = {}, True
            for col, _ in HEATMAP_SIGNALS:
                try: v = float(r[col])
                except (TypeError, ValueError, KeyError):
                    ok = False; break
                if not math.isfinite(v):
                    ok = False; break
                row[col] = abs(v)
            if ok: samples.append(row)
    return samples


def _rank(xs):
    n = len(xs)
    sorted_pairs = sorted(enumerate(xs), key=lambda p: p[1])
    ranks = [0.0] * n; i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][0]] = avg
        i = j + 1
    return ranks


def _pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n; my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx2 = sum((xs[i] - mx) ** 2 for i in range(n))
    dy2 = sum((ys[i] - my) ** 2 for i in range(n))
    den = math.sqrt(dx2 * dy2)
    return num / den if den > 0 else 0.0


def _spearman_matrix(samples, signals=None):
    if signals is None:
        signals = HEATMAP_SIGNALS
    cols = [s[0] for s in signals]
    ranks = {c: _rank([row[c] for row in samples]) for c in cols}
    n = len(cols)
    M = [[0.0] * n for _ in range(n)]
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            M[i][j] = _pearson(ranks[c1], ranks[c2])
    return M


def _draw_heatmap(ax, samples, font_scale=1.0, signals=None):
    if signals is None:
        signals = HEATMAP_SIGNALS
    if not samples:
        ax.text(0.5, 0.5, "no signal data", transform=ax.transAxes,
                ha="center", va="center", alpha=0.6)
        ax.axis("off")
        return
    M = _spearman_matrix(samples, signals=signals)
    labels = [s[1] for s in signals]
    n = len(labels)
    arr = np.array(M)
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                   aspect="equal", interpolation="nearest")
    for i in range(n):
        for j in range(n):
            v = M[i][j]
            txt = "white" if abs(v) > 0.55 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=7.5 * font_scale, color=txt)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor",
                       fontsize=10.0 * font_scale)
    ax.set_yticklabels(labels, fontsize=10.0 * font_scale)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Spearman $\rho$", fontsize=10.0 * font_scale)
    cbar.ax.tick_params(labelsize=9.0 * font_scale)


# ─── Drawers ────────────────────────────────────────────────────────────────
def _draw_pareto(ax, by_sig):
    cmap = plt.get_cmap("viridis")
    ref = by_sig.get(RETRAIN)
    t_ref, f1_ref = _mean(ref["time"]), _mean(ref["f1"])
    points = []
    for sig in SIGNALS + [RETRAIN]:
        cell = by_sig.get(sig)
        if not cell or not cell["f1"] or not cell["time"]:
            continue
        points.append({
            "sig": sig,
            "f1m":  _mean(cell["f1"]),
            "f1s":  _stdev(cell["f1"]) if len(cell["f1"]) > 1 else 0,
            "tm":   _mean(cell["time"]),
            "ts":   _stdev(cell["time"]) if len(cell["time"]) > 1 else 0,
            "sw_rate": cell["switched"] / cell["n"] if cell["n"] else 0,
        })

    for p in points:
        color = "black" if p["sig"] == RETRAIN else cmap(p["sw_rate"])
        if p["sig"] == RETRAIN:
            marker, msize = "s", 19
        else:
            marker, msize = SIG_MARKER.get(
                TIME_SIG_MARKERS.get(p["sig"], ""), ("o", 17)
            )
        ax.plot([p["tm"]], [p["f1m"]], marker=marker, markersize=msize,
                color=color, markeredgecolor="black", markeredgewidth=1.2,
                linestyle="", zorder=5)
        offset = LABEL_OFFSETS.get(p["sig"], (6, 5))
        if p["sig"] in RIGHT_ANCHORED:
            ha = "right"
        elif p["sig"] in CENTER_ANCHORED:
            ha = "center"
        else:
            ha = "left"
        ax.annotate(DISPLAY_LABELS.get(p["sig"], p["sig"]),
                    (p["tm"], p["f1m"]),
                    xytext=offset, textcoords="offset points", ha=ha,
                    fontweight="bold" if p["sig"] == RETRAIN else "normal")

    f1_means = [p["f1m"] for p in points]
    y_lo = min(min(f1_means) - 0.0008, f1_ref - 0.0005)
    y_hi = max(max(f1_means) + 0.0008, f1_ref + 0.0005)
    ax.set_ylim(y_lo, y_hi)
    means_x = [p["tm"] for p in points]
    pad = 0.05 * (max(means_x) - min(means_x))
    ax.set_xlim(min(means_x) - pad, max(means_x) + pad)

    # Quadrant shading.
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    f_x = (t_ref - xmin) / (xmax - xmin)
    if ymin < f1_ref < ymax:
        ax.axhspan(f1_ref, ymax, xmin=0, xmax=f_x,
                   facecolor="#bce4b6", alpha=0.35, zorder=0)
        ax.axhspan(ymin, f1_ref, xmin=f_x, xmax=1,
                   facecolor="#f7c5c5", alpha=0.35, zorder=0)
    ax.axhline(f1_ref, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axvline(t_ref,  color="black", linestyle="--", linewidth=0.7, alpha=0.5)

    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10, linestyle=":")
    ax.set_xlabel("← lower is better      training time (s)", labelpad=4)
    ax.set_ylabel("mean test F1      higher is better →", labelpad=4)


def _draw_headline_panel(ax, ds, ds_data, ds_switches, with_ylabel,
                          draw_legend):
    plotted_any = False
    for sig, label, color, ls, marker, zorder in LINES:
        seed_dict = ds_data.get(sig, {})
        if not seed_dict:
            continue
        rs, ms, stds, _ = _per_round_mean(seed_dict)
        if not rs:
            continue
        plotted_any = True
        # ±1 std shaded band before the line so the line sits on top.
        upper = [m + s for m, s in zip(ms, stds)]
        lower = [m - s for m, s in zip(ms, stds)]
        ax.fill_between(rs, lower, upper, color=color, alpha=0.15,
                         linewidth=0, zorder=zorder - 1)
        ax.plot(rs, ms, color=color, linestyle=ls, linewidth=1.6,
                marker=marker, markersize=4.5, label=label, zorder=zorder)
        sw_list = ds_switches.get(sig, [])
        if sw_list:
            sw_int = int(round(_mean(sw_list)))
            if sw_int in rs:
                yval = ms[rs.index(sw_int)]
                ax.plot([sw_int], [yval], marker="*", color=color,
                        markersize=16, markeredgecolor="black",
                        markeredgewidth=0.8, linestyle="", zorder=50)
    title = {"agnews": "AG News", "yahoo_answers": "Yahoo Answers",
             "jigsaw": "Jigsaw"}.get(ds, ds)
    if not plotted_any:
        ax.set_title(f"{title}\n(no data)")
        ax.axis("off")
        return
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("round t")
    if with_ylabel:
        ax.set_ylabel("mean val F1")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which="minor", length=2)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10, linestyle=":")
    if draw_legend:
        ax.legend(loc="lower right", framealpha=0.95,
                  fontsize=10.0, handlelength=2.0, handletextpad=0.6,
                  borderpad=0.5, labelspacing=0.45)


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    # Oversized canvas + bumped fonts so LaTeX `width=\textwidth` scaling
    # leaves labels at ~10 pt and the two halves have generous space.
    apply_acl_style(scale=1.90)
    by_sig = _collect_pareto()
    head_data, head_sw = _collect_headline()

    fig = plt.figure(figsize=(13.6, 5.2))

    gs_left = fig.add_gridspec(1, 1, left=0.06, right=0.45,
                               top=0.86, bottom=0.13)
    ax_pareto = fig.add_subplot(gs_left[0, 0])
    _draw_pareto(ax_pareto, by_sig)

    # Marker-shape legend keys the paired-t-test significance.
    legend_handles = [
        plt.Line2D([0], [0], marker="*", color="dimgray",
                   markerfacecolor="dimgray", markeredgecolor="black",
                   markeredgewidth=1.0, markersize=20,
                   linestyle="", label="p<0.001"),
        plt.Line2D([0], [0], marker="D", color="dimgray",
                   markerfacecolor="dimgray", markeredgecolor="black",
                   markeredgewidth=1.0, markersize=12,
                   linestyle="", label="p<0.01"),
        plt.Line2D([0], [0], marker="X", color="dimgray",
                   markerfacecolor="dimgray", markeredgecolor="black",
                   markeredgewidth=1.0, markersize=14,
                   linestyle="", label="p≥0.05"),
        plt.Line2D([0], [0], marker="s", color="black",
                   markerfacecolor="black", markeredgecolor="black",
                   markeredgewidth=1.0, markersize=14,
                   linestyle="", label="Retrain"),
    ]
    ax_pareto.legend(handles=legend_handles,
                     loc="lower center", bbox_to_anchor=(0.5, 1.01),
                     framealpha=0.95, ncol=4,
                     handletextpad=0.3, columnspacing=1.0,
                     title="Time vs. Retrain")

    # Switch-rate colorbar attached to the Pareto axis.
    cmap = plt.get_cmap("viridis")
    sm = matplotlib.cm.ScalarMappable(
        cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    )
    cbar = fig.colorbar(sm, ax=ax_pareto, fraction=0.045, pad=0.025,
                        ticks=[0, 0.5, 1.0])
    cbar.set_label("switch rate")
    cbar.ax.set_yticklabels(["0%", "50%", "100%"])

    # Right cluster: AG News + Jigsaw stacked vertically in middle col,
    # heatmap on the right at a comfortable size (larger than one line
    # plot, vertically centered).
    # Right cluster: AG News + Yahoo Answers on the top row, Jigsaw
    # spanning the bottom row. (Heatmap moved to a separate figure —
    # see main_results_row_compact.py and signal_correlation_heatmap.py.)
    gs_right = fig.add_gridspec(2, 2, left=0.62, right=0.97,
                                top=0.99, bottom=0.13,
                                height_ratios=[1, 1], width_ratios=[1, 1],
                                hspace=0.55, wspace=0.45)
    ax_ag = fig.add_subplot(gs_right[0, 0])
    ax_yh = fig.add_subplot(gs_right[0, 1])
    ax_jg = fig.add_subplot(gs_right[1, :])

    _draw_headline_panel(ax_ag, "agnews", head_data["agnews"],
                          head_sw["agnews"], with_ylabel=True,
                          draw_legend=False)
    _draw_headline_panel(ax_yh, "yahoo_answers", head_data["yahoo_answers"],
                          head_sw["yahoo_answers"], with_ylabel=False,
                          draw_legend=False)
    _draw_headline_panel(ax_jg, "jigsaw", head_data["jigsaw"],
                          head_sw["jigsaw"], with_ylabel=True,
                          draw_legend=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    out_pdf = OUT.with_suffix(".pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"wrote {out_pdf.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
