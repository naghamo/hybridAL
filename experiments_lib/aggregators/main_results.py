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

from ..shared.tables import cell, safe_mean, paired_p, stars_for_p
from ._base import build_summary_and_per_round, write_default_csvs

NAME = "main_results"
SAVE_ROOT = Path(f"experiments/{NAME}")


def _filter(name: str) -> bool:
    return name.startswith("Main_")


def _method_label(r: dict) -> str:
    """Derive a clean method label from cfg-derived fields, not the run
    name (which conflates yahoo_answers with the backbone tail).

      RetrainStrategy            → "Retrain"
      FineTuneStrategy           → "FineTune"
      NewOnlyStrategy            → "NewOnly"
      DeltaF1Strategy + signal=  → "HybridAL_alpha" / "HybridAL_acc"
      FixedSwitchStrategy + N    → "FixedSwitch_<N>"
    """
    cls = r.get("strategy_class") or ""
    sig = r.get("signal")
    sw  = r.get("switch_round")
    if cls == "RetrainStrategy":   return "Retrain"
    if cls == "FineTuneStrategy":  return "FineTune"
    if cls == "NewOnlyStrategy":   return "NewOnly"
    if cls == "DeltaF1Strategy":
        if sig == "delta_spectral_alpha": return "HybridAL_alpha"
        if sig == "delta_accuracy":        return "HybridAL_acc"
        return f"HybridAL_{sig}" if sig else "HybridAL"
    if cls == "FixedSwitchStrategy":
        try:
            return f"FixedSwitch_{int(sw)}"
        except (TypeError, ValueError):
            return "FixedSwitch"
    return cls or "?"


def _backbone_label(r: dict) -> str:
    name = (r.get("model") or "").lower()
    if "distilbert" in name: return "DistilBERT"
    if "roberta"    in name: return "RoBERTa"
    if "bert"       in name: return "BERT"
    return name or "?"


def main(args: argparse.Namespace) -> None:
    rows_summary, rows_per_round = build_summary_and_per_round(
        SAVE_ROOT, name_filter=_filter,
    )
    for r in rows_summary:
        r["method"]   = _method_label(r)
        r["backbone"] = _backbone_label(r)

    if args.dry_run:
        print(f"[{NAME}] dry-run: {len(rows_summary)} rows would be aggregated")
        return
    write_default_csvs(SAVE_ROOT, rows_summary, rows_per_round)


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


        lines.append("")
        lines.append(f"{'t-test (HybridAL vs ?)':<24} | " + " | ".join(f"{d:>20}" for d in datasets))
        lines.append("-" * (24 + 3 + (20 + 3) * len(datasets)))
        for m in methods:
            if m == "HybridAL":
                continue
            ps = []
            for d in datasets:


                a_seeded = sorted(
                    (int(r["seed"]), float(r["test_f1"])) for r in rows_summary
                    if r["backbone"] == bb and r["method"] == "HybridAL"
                    and r["data"] == d)
                b_seeded = sorted(
                    (int(r["seed"]), float(r["test_f1"])) for r in rows_summary
                    if r["backbone"] == bb and r["method"] == m
                    and r["data"] == d)
                common = sorted(
                    set(s for s, _ in a_seeded) & set(s for s, _ in b_seeded))
                a = [v for s, v in a_seeded if s in common]
                b = [v for s, v in b_seeded if s in common]
                p = paired_p(a, b)
                ps.append(f"p={p:.3f}{stars_for_p(p)}" if p is not None else "—")
            lines.append(f"{('vs ' + m):<24} | " + " | ".join(f"{c:>20}" for c in ps))


    lines.append("\n=== TABLE B — training time (seconds, mean ± std) ===")
    lines.append(f"{'method':<22} | " + " | ".join(f"{d:>16}" for d in datasets))
    for m in methods:
        cells = []
        for d in datasets:
            vs = [r["training_time_total"] for r in rows_summary
                  if r["method"] == m and r["data"] == d]
            cells.append(cell(vs, fmt="{:.0f}"))
        lines.append(f"{m:<22} | " + " | ".join(f"{c:>16}" for c in cells))


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
