"""Aggregator for Exp 5 — Initial pool size."""
import argparse
import re
from pathlib import Path

from ._base import build_summary_and_per_round, write_default_csvs

NAME = "pool_size_ablation"
SAVE_ROOT = Path(f"experiments/{NAME}")


def _filter(name): return name.startswith("Pool_")
def _parse(run):

    parts = run.split("_")
    method = parts[1]
    pool = int(re.sub(r"\D", "", parts[2]))
    return method, pool


def main(args):
    rows_summary, rows_per_round = build_summary_and_per_round(SAVE_ROOT, _filter)
    for r in rows_summary:
        m, p = _parse(r["run"])
        r["method"] = m
        r["pool"] = p
    if args.dry_run:
        print(f"[{NAME}] dry-run: {len(rows_summary)} rows"); return
    write_default_csvs(SAVE_ROOT, rows_summary, rows_per_round)
