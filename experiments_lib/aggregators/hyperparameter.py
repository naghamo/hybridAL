"""Aggregator for Exp 2 — Hyperparameter tuning of (epsilon, k) per signal.

Walks `experiments/hyperparameter_tuning/`, groups runs by their HybridAL
`signal`, and writes `_summary.csv` + `_per_round.csv` with epsilon, k,
and signal carried through on each per-round row.
"""
import argparse
from pathlib import Path

from ._base import build_summary_and_per_round, write_default_csvs

NAME = "hyperparameter_tuning"
SAVE_ROOT = Path(f"experiments/{NAME}")


def _filter(name: str) -> bool:
    """Keep only this experiment's runs (HP_<signal>_eps<eps>_k<k>_<dataset>_seed<s>)."""
    return name.startswith("HP_")


def main(args: argparse.Namespace) -> None:
    rows_summary, rows_per_round = build_summary_and_per_round(
        SAVE_ROOT, name_filter=_filter,
        axis_keys_per_round=("epsilon", "k", "signal"),
    )
    if args.dry_run:
        print(f"[{NAME}] dry-run: {len(rows_summary)} rows would be aggregated")
        return
    write_default_csvs(SAVE_ROOT, rows_summary, rows_per_round)
