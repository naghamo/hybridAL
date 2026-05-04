"""Aggregator for Exp 9 — Calibration dataset sensitivity."""
import argparse
from pathlib import Path

from ..shared.tables import cell, safe_mean
from ._base import build_summary_and_per_round, write_default_csvs

NAME = "calibration_sensitivity"
SAVE_ROOT = Path(f"experiments/{NAME}")


def _filter(name): return name.startswith("Calib_")
def _source(run): return run.split("_")[1]


# Datasets used as calibration sources (so we can flag "seen vs unseen").
SEEN_DATASETS = {"imdb", "agnews"}


def main(args):
    rows_summary, rows_per_round = build_summary_and_per_round(SAVE_ROOT, _filter)
    for r in rows_summary:
        r["source"] = _source(r["run"])
    if args.dry_run:
        print(f"[{NAME}] dry-run: {len(rows_summary)} rows"); return
    write_default_csvs(SAVE_ROOT, rows_summary, rows_per_round)

    sources = sorted({r["source"] for r in rows_summary})
    datasets = sorted({r["data"] for r in rows_summary},
                      key=lambda d: (d not in SEEN_DATASETS, d))

    lines = ["Exp 9 — Calibration dataset sensitivity (mean test F1 ± std)"]
    lines.append(f"{'source':<10} | " + " | ".join(
        f"{d + (' ✓' if d in SEEN_DATASETS else ' ?'):>20}" for d in datasets))
    for s in sources:
        cs = [cell([r["test_f1"] for r in rows_summary
                    if r["source"] == s and r["data"] == d])
              for d in datasets]
        lines.append(f"{s:<10} | " + " | ".join(f"{c:>20}" for c in cs))

    lines.append("\n(✓ = dataset used as calibration source; ? = unseen by calibration)")

    # Robustness on UNSEEN datasets only.
    unseen = [d for d in datasets if d not in SEEN_DATASETS]
    lines.append(f"\nUnseen-dataset means per source:")
    for s in sources:
        vs = [r["test_f1"] for r in rows_summary if r["source"] == s and r["data"] in unseen]
        m = safe_mean(vs)
        lines.append(f"  source {s}: {m:.4f}" if m is not None else f"  source {s}: —")

    (SAVE_ROOT / "_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
