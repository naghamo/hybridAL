"""Aggregator for Exp 8 — Normalization sensitivity."""
import argparse
from pathlib import Path

from ..shared.tables import cell, safe_mean
from ._base import build_summary_and_per_round, write_default_csvs

NAME = "normalizer_sensitivity"
SAVE_ROOT = Path(f"experiments/{NAME}")


def _filter(name): return name.startswith("Norm_")
def _recipe(run):  return run.split("_")[1]


def main(args):
    rows_summary, rows_per_round = build_summary_and_per_round(SAVE_ROOT, _filter)
    for r in rows_summary:
        r["recipe"] = _recipe(r["run"])
    if args.dry_run:
        print(f"[{NAME}] dry-run: {len(rows_summary)} rows"); return
    write_default_csvs(SAVE_ROOT, rows_summary, rows_per_round)

    recipes = sorted({r["recipe"] for r in rows_summary})
    datasets = sorted({r["data"] for r in rows_summary})

    lines = ["Exp 8 — Normalization sensitivity (mean test F1 ± std)"]
    lines.append(f"{'recipe':<14} | " + " | ".join(f"{d:>16}" for d in datasets))
    for r_lbl in recipes:
        cs = [cell([r["test_f1"] for r in rows_summary
                    if r["recipe"] == r_lbl and r["data"] == d])
              for d in datasets]
        lines.append(f"{r_lbl:<14} | " + " | ".join(f"{c:>16}" for c in cs))

    means = {r_lbl: safe_mean([r["test_f1"] for r in rows_summary if r["recipe"] == r_lbl])
             for r_lbl in recipes}
    valid = [v for v in means.values() if v is not None]
    if len(valid) >= 2:
        spread = max(valid) - min(valid)
        flag = "ROBUST (Δ < 0.005)" if spread < 0.005 else f"NOT robust (Δ = {spread:.4f})"
        lines.append(f"\nOverall: spread across recipes = {spread:.4f} → {flag}")

    (SAVE_ROOT / "_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
