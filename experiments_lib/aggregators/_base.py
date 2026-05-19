"""Shared aggregator helpers.

Every experiment's aggregator follows the same pattern:
  1. Walk `experiments/<name>/<run_dir>/results_*.json`.
  2. For each run, extract the (cfg, switch, F1, time, per-round signals).
  3. Write `_summary.csv` + `_per_round.csv`.
"""
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from ..shared.csv_io import write_csv


SIGNAL_KEYS = (
    "delta_f1", "delta_accuracy", "delta_loss", "gradient_norm",
    "l2_weight_distance", "cka",
    "delta_spectral_alpha", "delta_nc",
    "_raw_spectral_alpha", "_raw_nc",
    "weight_norm", "l2_vs_round1", "cka_vs_round1",
    "alpha_vs_round1", "nc_vs_round1",
)


def discover_runs(save_root: Path,
                  name_filter: Optional[callable] = None
                 ) -> List[Path]:
    """Return paths to results_*.json files under `save_root`.

    `name_filter` is a callable `name -> bool` over the run dir name; only
    runs that pass it are returned.
    """
    out: List[Path] = []
    if not save_root.exists():
        return out
    for run_dir in sorted(save_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if name_filter is not None and not name_filter(run_dir.name):
            continue
        results = sorted(run_dir.glob("results_*.json"))
        if results:
            out.append(results[-1])
    return out


def read_one(result_path: Path) -> Dict[str, Any]:
    """Parse one results_*.json into a flat row dict."""
    d = json.loads(result_path.read_text())
    cfg = d.get("cfg", {}) or {}
    skw = cfg.get("strategy_kwargs", {}) or {}
    final = d.get("final_test_stats", {}) or {}
    rounds = d.get("round_val_stats", []) or []
    train_time_total = sum((r.get("training_time") or 0) for r in rounds)
    actual_epochs = [r.get("actual_epochs") for r in rounds if r.get("actual_epochs") is not None]



    last_round = rounds[-1] if rounds else {}
    return {
        "run":             result_path.parent.name,
        "result_path":     str(result_path),
        "strategy_class":  cfg.get("strategy_class"),
        "sampler_class":   cfg.get("sampler_class"),
        "model":           cfg.get("model_name_or_path"),
        "data":            cfg.get("data"),
        "seed":            cfg.get("seed"),
        "epsilon":         skw.get("epsilon"),
        "k":               skw.get("k"),
        "signal":          skw.get("signal"),
        "signal_normalizer": skw.get("signal_normalizer"),
        "switch_round":    skw.get("switch_round")
                           or (d.get("strategy_metadata") or {}).get("switch_round"),

        "final_val_f1":       last_round.get("f1_score"),
        "final_val_accuracy": last_round.get("accuracy"),
        "final_val_loss":     last_round.get("loss"),

        "test_f1":         final.get("f1_score"),
        "test_accuracy":   final.get("accuracy"),
        "test_loss":       final.get("loss"),
        "training_time_total": train_time_total,
        "rounds_completed":  len(rounds),
        "mean_actual_epochs": mean(actual_epochs) if actual_epochs else None,
    }, rounds


def build_summary_and_per_round(save_root: Path,
                                name_filter: Optional[callable] = None,
                                axis_keys_per_round: Tuple[str, ...] = (),
                                ):
    """Read all matching runs and return (rows_summary, rows_per_round).

    `axis_keys_per_round` are extra cfg fields to pass through to per-round
    rows (e.g. `("epsilon", "k", "method")`); useful for downstream tables.
    """
    rows_summary: List[Dict[str, Any]] = []
    rows_per_round: List[Dict[str, Any]] = []
    for jp in discover_runs(save_root, name_filter):
        summary, rounds = read_one(jp)
        rows_summary.append(summary)
        for ridx, r in enumerate(rounds, 1):
            sig_dict = (r.get("signals") or {})
            row = {
                "run":     summary["run"],
                "data":    summary["data"],
                "seed":    summary["seed"],
                "round":   ridx,
                "f1_score":  r.get("f1_score"),
                "accuracy":  r.get("accuracy"),
                "loss":      r.get("loss"),
                "actual_epochs": r.get("actual_epochs"),
                "training_time": r.get("training_time"),
            }
            for k in axis_keys_per_round:
                row[k] = summary.get(k)
            for s in SIGNAL_KEYS:
                row[f"sig_{s}"] = sig_dict.get(s)
            rows_per_round.append(row)
    return rows_summary, rows_per_round


def write_default_csvs(save_root: Path,
                       rows_summary: List[Dict],
                       rows_per_round: List[Dict]) -> None:
    if rows_summary:
        write_csv(rows_summary, save_root / "_summary.csv",
                  list(rows_summary[0].keys()))
    if rows_per_round:
        write_csv(rows_per_round, save_root / "_per_round.csv",
                  list(rows_per_round[0].keys()))
