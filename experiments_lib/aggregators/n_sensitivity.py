"""Aggregator for Exp 0 — N (random_subset_size) sensitivity."""
import argparse
from pathlib import Path

from ._base import build_summary_and_per_round, write_default_csvs

NAME = "n_sensitivity"
SAVE_ROOT = Path(f"experiments/{NAME}")


def _filter(name: str) -> bool:
    return name.startswith("NSens_")


def main(args: argparse.Namespace) -> None:
    rows_summary, rows_per_round = build_summary_and_per_round(
        SAVE_ROOT, name_filter=_filter,
    )
    for r in rows_summary:
        run = r["run"]
        if run.startswith("NSens_n"):
            try:
                r["n"] = int(run.split("_")[1][1:])
            except (ValueError, IndexError):
                r["n"] = None
    if args.dry_run:
        print(f"[{NAME}] dry-run: {len(rows_summary)} rows would be aggregated")
        return
    write_default_csvs(SAVE_ROOT, rows_summary, rows_per_round)
