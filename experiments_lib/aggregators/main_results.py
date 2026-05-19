"""Aggregator for Exp 3 — Main results.

Produces:
  experiments/main_results/_summary.csv
                          /_per_round.csv
"""
import argparse
from pathlib import Path

from ._base import build_summary_and_per_round, write_default_csvs

NAME = "main_results"
SAVE_ROOT = Path(f"experiments/{NAME}")


def _filter(name: str) -> bool:
    return name.startswith("Main_")


def _method_label(r: dict) -> str:
    """Derive a clean method label from cfg-derived fields, not the run
    name (which conflates yahoo_answers with the backbone tail).

      RetrainStrategy            -> "Retrain"
      FineTuneStrategy           -> "FineTune"
      NewOnlyStrategy            -> "NewOnly"
      DeltaF1Strategy + signal=  -> "HybridAL_alpha" / "HybridAL_acc"
      FixedSwitchStrategy + N    -> "FixedSwitch_<N>"
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
