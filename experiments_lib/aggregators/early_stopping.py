"""Aggregator for Exp 7 — Early stopping validation."""
import argparse
from pathlib import Path

from ._base import build_summary_and_per_round, write_default_csvs

NAME = "early_stopping_validation"
SAVE_ROOT = Path(f"experiments/{NAME}")


def _filter(name): return name.startswith("ES_")
def _parse(run):

    parts = run.split("_")
    return parts[1] + "_" + parts[2], parts[3]


def main(args):
    rows_summary, rows_per_round = build_summary_and_per_round(SAVE_ROOT, _filter)
    for r in rows_summary:
        cfg_label, method = _parse(r["run"])
        r["config"] = cfg_label
        r["method"] = method
    if args.dry_run:
        print(f"[{NAME}] dry-run: {len(rows_summary)} rows"); return
    write_default_csvs(SAVE_ROOT, rows_summary, rows_per_round)
