"""Aggregator for Exp 8 — Normalization sensitivity."""
import argparse
from pathlib import Path

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
