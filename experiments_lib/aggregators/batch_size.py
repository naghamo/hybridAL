"""Aggregator for Exp 6 — Acquisition batch size."""
import argparse
import re
from pathlib import Path

from ..shared.tables import cell, welch_p, stars_for_p
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

    ns = sorted({r["n"] for r in rows_summary})
    datasets = sorted({r["data"] for r in rows_summary})

    lines = ["Exp 6 — Acquisition batch size (mean test F1 ± std)"]
    lines.append(f"{'method × n':<24} | " + " | ".join(f"{d:>16}" for d in datasets))
    for m in ("Retrain", "HybridAL"):
        for n in ns:
            cells = [cell([r["test_f1"] for r in rows_summary
                           if r["method"] == m and r["n"] == n and r["data"] == d])
                     for d in datasets]
            lines.append(f"{(m + ' × ' + str(n)):<24} | " + " | ".join(f"{c:>16}" for c in cells))

    lines.append("\nWelch t-test HybridAL vs Retrain per (n × dataset):")
    for n in ns:
        for d in datasets:
            a = [r["test_f1"] for r in rows_summary
                 if r["method"] == "HybridAL" and r["n"] == n and r["data"] == d]
            b = [r["test_f1"] for r in rows_summary
                 if r["method"] == "Retrain" and r["n"] == n and r["data"] == d]
            pp = welch_p(a, b)
            if pp is not None:
                lines.append(f"  n={n:>4} × {d}: p={pp:.3f}{stars_for_p(pp)}")

    (SAVE_ROOT / "_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
