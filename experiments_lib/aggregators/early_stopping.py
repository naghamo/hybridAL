"""Aggregator for Exp 7 — Early stopping validation."""
import argparse
from pathlib import Path

from ..shared.tables import cell
from ._base import build_summary_and_per_round, write_default_csvs

NAME = "early_stopping_validation"
SAVE_ROOT = Path(f"experiments/{NAME}")


def _filter(name): return name.startswith("ES_")
def _parse(run):
    # ES_<config>_<method>_<dataset>_seed<seed>
    parts = run.split("_")
    return parts[1] + "_" + parts[2], parts[3]   # config, method


def main(args):
    rows_summary, rows_per_round = build_summary_and_per_round(SAVE_ROOT, _filter)
    for r in rows_summary:
        cfg_label, method = _parse(r["run"])
        r["config"] = cfg_label
        r["method"] = method
    if args.dry_run:
        print(f"[{NAME}] dry-run: {len(rows_summary)} rows"); return
    write_default_csvs(SAVE_ROOT, rows_summary, rows_per_round)

    configs = sorted({r["config"] for r in rows_summary})
    methods = sorted({r["method"] for r in rows_summary})
    datasets = sorted({r["data"] for r in rows_summary})

    lines = ["Exp 7 — Early stopping validation (mean test F1 ± std)"]
    for cfg_lbl in configs:
        lines.append(f"\n--- config = {cfg_lbl} ---")
        lines.append(f"{'method':<12} | " + " | ".join(f"{d:>16}" for d in datasets))
        for m in methods:
            cs = [cell([r["test_f1"] for r in rows_summary
                        if r["config"] == cfg_lbl and r["method"] == m and r["data"] == d])
                  for d in datasets]
            lines.append(f"{m:<12} | " + " | ".join(f"{c:>16}" for c in cs))

    lines.append("\nF1 delta (B_es_max10 - A_fixed5) per (method, dataset):")
    for m in methods:
        for d in datasets:
            a = [r["test_f1"] for r in rows_summary
                 if r["config"] == "A_fixed5" and r["method"] == m and r["data"] == d]
            b = [r["test_f1"] for r in rows_summary
                 if r["config"] == "B_es_max10" and r["method"] == m and r["data"] == d]
            if a and b:
                delta = sum(b) / len(b) - sum(a) / len(a)
                flag = "  ⚠ ES degrades >1%" if delta < -0.01 else ""
                lines.append(f"  {m:<10} × {d:<14}: ΔF1 = {delta:+.4f}{flag}")

    (SAVE_ROOT / "_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
