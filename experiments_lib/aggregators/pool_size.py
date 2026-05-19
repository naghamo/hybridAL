"""Aggregator for Exp 5 — Initial pool size."""
import argparse
import re
from pathlib import Path

from ..shared.tables import cell, paired_p, stars_for_p
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

    pools = sorted({r["pool"] for r in rows_summary})
    datasets = sorted({r["data"] for r in rows_summary})

    lines = ["Exp 5 — Initial pool size (mean test F1 ± std)"]
    lines.append(f"{'method × |L0|':<24} | " + " | ".join(f"{d:>16}" for d in datasets))
    for m in ("Retrain", "HybridAL"):
        for p in pools:
            cells = [cell([r["test_f1"] for r in rows_summary
                           if r["method"] == m and r["pool"] == p and r["data"] == d])
                     for d in datasets]
            lines.append(f"{(m + ' × ' + str(p)):<24} | " + " | ".join(f"{c:>16}" for c in cells))

    lines.append("\nMean switch round per (|L0| × dataset) [HybridAL only]:")
    for p in pools:
        for d in datasets:
            sw = [r["switch_round"] for r in rows_summary
                  if r["method"] == "HybridAL" and r["pool"] == p and r["data"] == d
                  and r["switch_round"]]
            sw_str = f"{sum(sw)/len(sw):.1f}" if sw else "—"
            lines.append(f"  pool={p:>4} × {d}: {sw_str}")

    lines.append("\nPaired t-test HybridAL vs Retrain (per pool × dataset; "
                 "paired by seed):")
    for p in pools:
        for d in datasets:
            a_seeded = sorted(
                (int(r["seed"]), float(r["test_f1"])) for r in rows_summary
                if r["method"] == "HybridAL" and r["pool"] == p and r["data"] == d)
            b_seeded = sorted(
                (int(r["seed"]), float(r["test_f1"])) for r in rows_summary
                if r["method"] == "Retrain" and r["pool"] == p and r["data"] == d)
            common = sorted(set(s for s, _ in a_seeded)
                            & set(s for s, _ in b_seeded))
            a = [v for s, v in a_seeded if s in common]
            b = [v for s, v in b_seeded if s in common]
            pp = paired_p(a, b)
            if pp is not None:
                lines.append(f"  pool={p:>4} × {d}: p={pp:.3f}{stars_for_p(pp)}")

    (SAVE_ROOT / "_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
