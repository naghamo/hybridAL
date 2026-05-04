"""Aggregator for Exp 3 — Main results.

Produces:
  experiments/main_results/_summary.csv
                          /_per_round.csv
                          /_report.txt           (Tables A, B, C as text)
                          /plots/                (per-round F1 curves, Pareto)
"""
import argparse
from collections import defaultdict
from pathlib import Path

from ..shared.tables import cell, safe_mean, welch_p, stars_for_p
from ._base import build_summary_and_per_round, write_default_csvs

NAME = "main_results"
SAVE_ROOT = Path(f"experiments/{NAME}")


def _filter(name: str) -> bool:
    return name.startswith("Main_")


def _method_from_run(run: str) -> str:
    # Run name format: Main_<method>_<backbone_short>_<dataset>_seed<seed>
    parts = run.split("_")
    return parts[1] if len(parts) >= 2 else "?"


def _backbone_from_run(run: str) -> str:
    parts = run.split("_")
    return "_".join(parts[2:-2]) if len(parts) >= 4 else "?"


def main(args: argparse.Namespace) -> None:
    rows_summary, rows_per_round = build_summary_and_per_round(
        SAVE_ROOT, name_filter=_filter,
    )
    for r in rows_summary:
        r["method"] = _method_from_run(r["run"])
        r["backbone"] = _backbone_from_run(r["run"])

    if args.dry_run:
        print(f"[{NAME}] dry-run: {len(rows_summary)} rows would be aggregated")
        return
    write_default_csvs(SAVE_ROOT, rows_summary, rows_per_round)

    # Tables A (per-backbone): method × dataset → mean F1
    backbones = sorted({r["backbone"] for r in rows_summary})
    methods = sorted({r["method"] for r in rows_summary})
    datasets = sorted({r["data"] for r in rows_summary})

    lines = []
    for bb in backbones:
        lines.append(f"\n=== TABLE A — backbone={bb} (mean test F1 ± std over seeds) ===")
        lines.append(f"{'method':<22} | " + " | ".join(f"{d:>20}" for d in datasets))
        lines.append("-" * (22 + 3 + (20 + 3) * len(datasets)))
        for m in methods:
            f1_cells = []
            for d in datasets:
                vs = [r["test_f1"] for r in rows_summary
                      if r["backbone"] == bb and r["method"] == m and r["data"] == d]
                f1_cells.append(cell(vs))
            lines.append(f"{m:<22} | " + " | ".join(f"{c:>20}" for c in f1_cells))

        # t-test row: HybridAL vs each other method, per dataset
        lines.append("")
        lines.append(f"{'t-test (HybridAL vs ?)':<24} | " + " | ".join(f"{d:>20}" for d in datasets))
        lines.append("-" * (24 + 3 + (20 + 3) * len(datasets)))
        for m in methods:
            if m == "HybridAL":
                continue
            ps = []
            for d in datasets:
                a = [r["test_f1"] for r in rows_summary
                     if r["backbone"] == bb and r["method"] == "HybridAL" and r["data"] == d]
                b = [r["test_f1"] for r in rows_summary
                     if r["backbone"] == bb and r["method"] == m and r["data"] == d]
                p = welch_p(a, b)
                ps.append(f"p={p:.3f}{stars_for_p(p)}" if p is not None else "—")
            lines.append(f"{('vs ' + m):<24} | " + " | ".join(f"{c:>20}" for c in ps))

    # Table B: training time
    lines.append("\n=== TABLE B — training time (seconds, mean ± std) ===")
    lines.append(f"{'method':<22} | " + " | ".join(f"{d:>16}" for d in datasets))
    for m in methods:
        cells = []
        for d in datasets:
            vs = [r["training_time_total"] for r in rows_summary
                  if r["method"] == m and r["data"] == d]
            cells.append(cell(vs, fmt="{:.0f}"))
        lines.append(f"{m:<22} | " + " | ".join(f"{c:>16}" for c in cells))

    # Table C: backbone summary (mean F1 across all 6 datasets per (method, backbone))
    lines.append("\n=== TABLE C — backbone summary (mean F1 across all 6 datasets) ===")
    lines.append(f"{'method':<22} | " + " | ".join(f"{bb:>22}" for bb in backbones))
    for m in methods:
        cells = []
        for bb in backbones:
            vs = [r["test_f1"] for r in rows_summary
                  if r["method"] == m and r["backbone"] == bb]
            cells.append(cell(vs))
        lines.append(f"{m:<22} | " + " | ".join(f"{c:>22}" for c in cells))

    (SAVE_ROOT / "_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
