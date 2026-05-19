"""Aggregator for Exp 4 — Sampler ablation."""
import argparse
from pathlib import Path

from ._base import build_summary_and_per_round, write_default_csvs

NAME = "sampler_ablation"
SAVE_ROOT = Path(f"experiments/{NAME}")


def _filter(name): return name.startswith("Sampler_")
def _method(run):  return run.split("_")[1]
def _sampler(run): return run.split("_")[2]


def main(args: argparse.Namespace) -> None:
    rows_summary, rows_per_round = build_summary_and_per_round(SAVE_ROOT, _filter)
    for r in rows_summary:
        r["method"] = _method(r["run"])
        r["sampler"] = _sampler(r["run"])
    if args.dry_run:
        print(f"[{NAME}] dry-run: {len(rows_summary)} rows"); return
    write_default_csvs(SAVE_ROOT, rows_summary, rows_per_round)
