"""Aggregator for Exp 4 — Sampler ablation."""
import argparse
from pathlib import Path

from ..shared.tables import cell, welch_p, stars_for_p
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

    samplers = sorted({r["sampler"] for r in rows_summary})
    datasets = sorted({r["data"] for r in rows_summary})
    methods = ["Retrain", "HybridAL"]

    lines = ["Exp 4 — Sampler ablation (mean test F1 ± std)"]
    lines.append(f"{'method × sampler':<32} | " + " | ".join(f"{d:>16}" for d in datasets))
    for m in methods:
        for s in samplers:
            cells = []
            for d in datasets:
                vs = [r["test_f1"] for r in rows_summary
                      if r["method"] == m and r["sampler"] == s and r["data"] == d]
                cells.append(cell(vs))
            lines.append(f"{(m + ' × ' + s):<32} | " + " | ".join(f"{c:>16}" for c in cells))

    lines.append("\nWelch's t-test (HybridAL vs Retrain) per (sampler, dataset):")
    for s in samplers:
        for d in datasets:
            a = [r["test_f1"] for r in rows_summary
                 if r["method"] == "HybridAL" and r["sampler"] == s and r["data"] == d]
            b = [r["test_f1"] for r in rows_summary
                 if r["method"] == "Retrain" and r["sampler"] == s and r["data"] == d]
            p = welch_p(a, b)
            if p is not None:
                lines.append(f"  {s} × {d}: p={p:.3f}{stars_for_p(p)}")

    (SAVE_ROOT / "_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
