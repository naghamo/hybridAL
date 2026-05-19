"""Aggregator for Exp 0 — N (random_subset_size) sensitivity."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from ..shared.csv_io import to_int
from ..shared.tables import cell
from ._base import build_summary_and_per_round, write_default_csvs

NAME = "n_sensitivity"
SAVE_ROOT = Path(f"experiments/{NAME}")


def _filter(name: str) -> bool:
    return name.startswith("NSens_")


def main(args: argparse.Namespace) -> None:
    rows_summary, rows_per_round = build_summary_and_per_round(
        SAVE_ROOT, name_filter=_filter,
    )
    if args.dry_run:
        print(f"[{NAME}] dry-run: {len(rows_summary)} rows would be aggregated")
        print(f"  expected output: {SAVE_ROOT}/_summary.csv, _per_round.csv, _report.txt")
        return
    write_default_csvs(SAVE_ROOT, rows_summary, rows_per_round)


    by_n_ds = defaultdict(list)
    for r in rows_summary:
        run = r["run"]
        if not run.startswith("NSens_n"):
            continue
        n = int(run.split("_")[1][1:])
        by_n_ds[(n, r["data"])].append(r["test_f1"])

    Ns = sorted({k[0] for k in by_n_ds})
    datasets = sorted({k[1] for k in by_n_ds})
    lines = ["Exp 0 — N sensitivity\n", f"{'N':>6} | " + " | ".join(f"{d:>20}" for d in datasets)]
    lines.append("-" * len(lines[-1]))
    for n in Ns:
        cells = [cell(by_n_ds.get((n, d), [])) for d in datasets]
        lines.append(f"{n:>6} | " + " | ".join(f"{c:>20}" for c in cells))
    (SAVE_ROOT / "_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
