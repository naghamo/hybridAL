"""Aggregator for Exp 6 — Acquisition batch size."""
import argparse
import re
from pathlib import Path

from ._base import build_summary_and_per_round, write_default_csvs

NAME = "batch_size_ablation"
SAVE_ROOT = Path(f"experiments/{NAME}")


def _filter(name): return name.startswith("Batch_")
def _parse(run):
    parts = run.split("_")
    return parts[1], int(re.sub(r"\D", "", parts[2]))


def main(args):
    rows_summary, rows_per_round = build_summary_and_per_round(SAVE_ROOT, _filter)
    for r in rows_summary:
        m, n = _parse(r["run"])
        r["method"] = m; r["n"] = n
    if args.dry_run:
        print(f"[{NAME}] dry-run: {len(rows_summary)} rows"); return
    write_default_csvs(SAVE_ROOT, rows_summary, rows_per_round)
