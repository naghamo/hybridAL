"""Aggregator for Exp 2 — Hyperparameter tuning of (ε, k) per signal.

Walks `experiments/hyperparameter_tuning/`, groups runs by their HybridAL
`signal`, and for each signal independently:
  1. Pivots the 16 (ε, k) cells across the (dataset × seed) replicas.
  2. Picks the best (ε, k) using argmax mean **validation** F1 (NOT test —
     test is held out for final reporting only, picking on it would be
     data leakage), with a tie-break of "prefer the EARLIEST mean switch
     round" within 0.5 % val F1 of the top.

     Rationale for the tie-break: HybridAL's reason to exist is saving
     compute while preserving performance. The 0.5 % tolerance already
     guarantees performance is preserved (within seed noise). Among
     equally-performing configs, the one that switches EARLIEST gets
     more rounds of cheap FineTune training and therefore the most
     time savings. Configs that never switch in some runs deliver zero
     time savings on those runs, so they're ranked LAST (treated as
     mean switch round = +∞).
  3. Writes:
       - chosen_eps_k_<signal>.json  (the picked pair + diagnostics)
       - plots/heatmap_<signal>.png  (ε × k heatmap)
       - _report.txt                 (combined report for both signals)
       - _summary.csv / _per_round.csv (raw run-level data)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..shared.tables import cell, safe_mean
from ._base import build_summary_and_per_round, write_default_csvs

NAME = "hyperparameter_tuning"
SAVE_ROOT = Path(f"experiments/{NAME}")
TIE_BREAK_F1_TOL = 0.005   # candidates within 0.5 % F1 of top are tied


def _filter(name: str) -> bool:
    """Keep only this experiment's runs (HP_<signal>_eps<ε>_k<k>_<dataset>_seed<s>)."""
    return name.startswith("HP_")


def _make_heatmap(by_eps_k_f1: dict, signal: str, save_path: Path) -> None:
    """Render an ε × k heatmap of mean **validation** F1 for one signal."""
    eps_vals = sorted({k[0] for k in by_eps_k_f1})
    k_vals = sorted({k[1] for k in by_eps_k_f1})
    grid = np.full((len(eps_vals), len(k_vals)), np.nan)
    for i, e in enumerate(eps_vals):
        for j, k in enumerate(k_vals):
            m = safe_mean(by_eps_k_f1.get((e, k), []))
            if m is not None:
                grid[i, j] = m

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels(k_vals)
    ax.set_yticks(range(len(eps_vals)))
    ax.set_yticklabels([f"{e:.4g}" for e in eps_vals])
    ax.set_xlabel("k (consecutive rounds)")
    ax.set_ylabel("ε (raw scale, no normalization)")
    ax.set_title(f"Mean val F1 across ε × k — signal = {signal}\n"
                 f"averaged over IMDb + AG News × 3 seeds (test set held out)")
    for i in range(len(eps_vals)):
        for j in range(len(k_vals)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="mean val F1")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=90)
    plt.close(fig)


def _pick_best_eps_k(by_eps_k_val_f1: dict, by_eps_k_switch: dict,
                     by_eps_k_test_f1: dict) -> dict:
    """Pick (ε, k) for one signal: argmax mean **val** F1, tie-break prefers
    the EARLIEST mean switch round within 0.5 % val F1 of the top.

    Why val (not test): test set is held out for final reporting in Exp 3.
    Selecting hyperparameters on test F1 would be data leakage.

    Why "earliest switch" as tie-break: among configs that achieve
    statistically equivalent val F1 (within 0.5 %), the one that switches
    EARLIEST delivers the most time savings, which is HybridAL's whole
    raison d'être. Performance is already guaranteed by the val-F1
    tolerance; this prioritises the time benefit among equal performers.
    Configs that never switched on at least one of their replicas deliver
    no time savings on that run, so we treat their effective switch
    round as +∞ and rank them LAST.
    """
    val_means = {k: safe_mean(v) for k, v in by_eps_k_val_f1.items() if v}
    if not val_means:
        return {}
    top = max(val_means.values())
    cands = [k for k, m in val_means.items() if top - m <= TIE_BREAK_F1_TOL]

    # Among ties, prefer the SMALLEST mean switch round (most time savings).
    # If any of the cell's replicas never switched, treat its mean switch
    # round as +∞ so it's ranked LAST among ties.
    NEVER_SWITCHED = float("inf")
    def _mean_switch(key):
        sws = by_eps_k_switch.get(key, [])
        if not sws:
            return NEVER_SWITCHED
        if any(s is None for s in sws):
            return NEVER_SWITCHED
        return mean(sws)

    # Sort: earliest switch first (smaller is better); on tie, smaller k.
    cands.sort(key=lambda k: (_mean_switch(k), k[1]))
    eps, k = cands[0]
    sw = _mean_switch((eps, k))
    return {
        "epsilon": eps,
        "k": k,
        "mean_val_f1": val_means[(eps, k)],
        "mean_test_f1_at_chosen": safe_mean(by_eps_k_test_f1.get((eps, k), [])),
        "best_val_f1_among_grid": top,
        "tie_break_within_val_f1": TIE_BREAK_F1_TOL,
        "tie_break_rule": "prefer EARLIEST mean switch round (max time savings)",
        "n_tied_candidates": len(cands),
        "mean_switch_round": (sw if sw != NEVER_SWITCHED else None),
        "all_replicas_switched": (sw != NEVER_SWITCHED),
    }


def main(args: argparse.Namespace) -> None:
    rows_summary, rows_per_round = build_summary_and_per_round(
        SAVE_ROOT, name_filter=_filter,
        axis_keys_per_round=("epsilon", "k", "signal"),
    )
    if args.dry_run:
        print(f"[{NAME}] dry-run: {len(rows_summary)} rows would be aggregated")
        return
    write_default_csvs(SAVE_ROOT, rows_summary, rows_per_round)

    # Group rows by signal first. Use VALIDATION F1 for selection — test F1
    # is held out for the final paper table (Exp 3) and selecting on it
    # would be data leakage.
    by_signal_eps_k_f1 = defaultdict(lambda: defaultdict(list))   # final-round val F1
    by_signal_eps_k_sw = defaultdict(lambda: defaultdict(list))
    by_signal_eps_k_test = defaultdict(lambda: defaultdict(list)) # diagnostic only
    for r in rows_summary:
        sig = r.get("signal")
        val_f1 = r.get("final_val_f1")
        if (sig is None or r.get("epsilon") is None
                or r.get("k") is None or val_f1 is None):
            continue
        key = (float(r["epsilon"]), int(r["k"]))
        by_signal_eps_k_f1[sig][key].append(float(val_f1))
        if r.get("test_f1") is not None:
            by_signal_eps_k_test[sig][key].append(float(r["test_f1"]))
        sw = r.get("switch_round")
        try:
            sw_val = int(sw) if sw is not None else None
        except (TypeError, ValueError):
            sw_val = None
        by_signal_eps_k_sw[sig][key].append(sw_val)

    if not by_signal_eps_k_f1:
        print(f"[{NAME}] no rows yet")
        return

    # Per-signal selection + per-signal heatmap + combined report.
    signals_sorted = sorted(by_signal_eps_k_f1)
    report_lines = ["Exp 2 — Hyperparameter tuning of (ε, k)"]

    for signal in signals_sorted:
        val_pivot = by_signal_eps_k_f1[signal]
        sw_pivot = by_signal_eps_k_sw[signal]
        test_pivot = by_signal_eps_k_test[signal]

        # Heatmap (val F1 — selection metric)
        out_plot = SAVE_ROOT / "plots" / f"heatmap_{signal}.png"
        _make_heatmap(val_pivot, signal, out_plot)

        # Choose
        chosen = _pick_best_eps_k(val_pivot, sw_pivot, test_pivot)
        chosen["signal"] = signal
        out_json = SAVE_ROOT / f"chosen_eps_k_{signal}.json"
        out_json.write_text(json.dumps(chosen, indent=2))

        # Report rows
        eps_vals = sorted({k[0] for k in val_pivot})
        k_vals = sorted({k[1] for k in val_pivot})
        sw_str = (f"{chosen['mean_switch_round']:.1f}"
                  if chosen.get("mean_switch_round") is not None
                  else "never (some replicas)")
        test_str = (f"{chosen['mean_test_f1_at_chosen']:.4f}"
                    if chosen.get("mean_test_f1_at_chosen") is not None else "n/a")
        report_lines += [
            "",
            f"=== signal = {signal} ===",
            f"chosen (ε, k): ε={chosen['epsilon']:g}, k={chosen['k']}",
            f"  mean VAL F1 (selection metric) = {chosen['mean_val_f1']:.4f}  "
            f"(top in grid = {chosen['best_val_f1_among_grid']:.4f}, "
            f"{chosen['n_tied_candidates']} candidates within "
            f"{chosen['tie_break_within_val_f1']*100:.1f}% of top)",
            f"  tie-break rule: {chosen['tie_break_rule']}",
            f"  mean test F1 at chosen (diagnostic only) = {test_str}",
            f"  mean switch round at chosen = {sw_str}",
            "",
            "Full ε × k grid (mean VAL F1 ± std — used for selection):",
            f"{'ε / k':>12} | " + " | ".join(f"{k:>14}" for k in k_vals),
            "-" * (14 + 4 * (len(k_vals) + 1)),
        ]
        for e in eps_vals:
            cells = [cell(val_pivot.get((e, k), [])) for k in k_vals]
            report_lines.append(f"{e:>12.4g} | " + " | ".join(f"{c:>14}" for c in cells))

    report_lines.append("")
    report_lines.append("Final values to pass into Phase-2 (main_results, samplers, "
                        "pool_size, batch_size):")
    for signal in signals_sorted:
        chosen = json.loads(
            (SAVE_ROOT / f"chosen_eps_k_{signal}.json").read_text()
        )
        report_lines.append(f"  --signal {signal}  --epsilon {chosen['epsilon']:g}"
                            f"  --k {chosen['k']}")

    (SAVE_ROOT / "_report.txt").write_text("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))
    print(f"\nwrote {SAVE_ROOT}/_report.txt")
    for signal in signals_sorted:
        print(f"  → {SAVE_ROOT}/chosen_eps_k_{signal}.json")
        print(f"  → {SAVE_ROOT}/plots/heatmap_{signal}.png")
