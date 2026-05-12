"""Plotting utilities: walk run directories, parse names, aggregate per cell.

A "run" is one experiment trial saved as
    <save_root>/<run_dir>/results_<timestamp>.json

Every run JSON has roughly this shape:
    cfg                : ExperimentConfig dict (data, seed, model_name_or_path,
                         strategy_class, strategy_kwargs, sampler_kwargs, …)
    round_val_stats    : list of per-round dicts (f1_score, accuracy, loss,
                         training_time, signals[…])
    final_test_stats   : {f1_score, accuracy, loss}
    strategy_metadata  : {switch_round, switched, …}
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Dict, Iterable, List, Optional


# --- I/O ----------------------------------------------------------------

def discover_runs(save_root: Path | str,
                  name_filter: Optional[Callable[[str], bool]] = None,
                  recursive: bool = False
                  ) -> List[Path]:
    """Return every `results_*.json` under `save_root`.

    `name_filter` is a callable applied to the immediate parent dir name
    (the run dir's basename); only matching runs are returned. If
    `recursive=True`, walks two levels deep so structures like
    `subset_sensitivity/subset_<N>/<run>/` are also found.
    """
    save_root = Path(save_root)
    out: List[Path] = []
    if not save_root.exists():
        return out
    pattern = "*/*/results_*.json" if recursive else "*/results_*.json"
    for jp in save_root.glob(pattern):
        run_name = jp.parent.name
        if name_filter is None or name_filter(run_name):
            out.append(jp)
    return sorted(out)


def load_run(jp: Path) -> Dict[str, Any]:
    """Load one run JSON. Doesn't crash on truncated files; returns {} if so."""
    try:
        return json.loads(Path(jp).read_text())
    except Exception:
        return {}


# --- Aggregators --------------------------------------------------------

def safe_mean(xs: Iterable[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and isinstance(x, (int, float))
          and math.isfinite(x)]
    return mean(xs) if xs else None


def safe_stdev(xs: Iterable[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and isinstance(x, (int, float))
          and math.isfinite(x)]
    return stdev(xs) if len(xs) > 1 else (0.0 if xs else None)


def aggregate_by(rows: List[Dict[str, Any]],
                 key_fields: Iterable[str],
                 value_field: str
                 ) -> Dict[tuple, Dict[str, Any]]:
    """Group `rows` by tuple-of-`key_fields`, return per-group {n, mean, std,
    values} dict. Skips rows where any key field or the value is None.
    """
    by: Dict[tuple, list] = defaultdict(list)
    for r in rows:
        try:
            key = tuple(r[k] for k in key_fields)
            v = r[value_field]
        except KeyError:
            continue
        if any(x is None for x in key) or v is None:
            continue
        by[key].append(v)
    out: Dict[tuple, Dict[str, Any]] = {}
    for key, vals in by.items():
        out[key] = {"n": len(vals), "values": vals,
                    "mean": safe_mean(vals), "std": safe_stdev(vals)}
    return out


# --- Run-level extractors ----------------------------------------------

def run_summary(jp: Path) -> Dict[str, Any]:
    """Flatten one run JSON into a row dict suited for downstream
    aggregation. Pulls fields commonly used in plots."""
    d = load_run(jp)
    cfg = d.get("cfg", {}) or {}
    skw = cfg.get("strategy_kwargs", {}) or {}
    sak = cfg.get("sampler_kwargs", {}) or {}
    final = d.get("final_test_stats", {}) or {}
    rounds = d.get("round_val_stats", []) or []
    train_time_total = sum((r.get("training_time") or 0) for r in rounds)
    last_round = rounds[-1] if rounds else {}
    return {
        "run":              jp.parent.name,
        "result_path":      str(jp),
        "strategy_class":   cfg.get("strategy_class"),
        "sampler_class":    cfg.get("sampler_class"),
        "model":            cfg.get("model_name_or_path"),
        "data":             cfg.get("data"),
        "seed":             cfg.get("seed"),
        "epsilon":          skw.get("epsilon"),
        "k":                skw.get("k"),
        "signal":           skw.get("signal"),
        "random_subset_size": (sak.get("random_subset_size")
                               or cfg.get("random_subset_size")),
        "switch_round":     skw.get("switch_round")
                            or (d.get("strategy_metadata") or {}).get("switch_round"),
        "test_f1":          final.get("f1_score"),
        "test_accuracy":    final.get("accuracy"),
        "test_loss":        final.get("loss"),
        "final_val_f1":     last_round.get("f1_score"),
        "final_val_accuracy": last_round.get("accuracy"),
        "final_val_loss":   last_round.get("loss"),
        "training_time_total": train_time_total,
        "rounds_completed": len(rounds),
    }


# --- Display helpers ----------------------------------------------------

DATASET_DISPLAY = {
    "imdb":          "IMDb",
    "agnews":        "AG News",
    "jigsaw":        "Jigsaw",
    "sst2":          "SST-2",
    "tweeteval":     "TweetEval",
    "yahoo_answers": "Yahoo Answers",
}


def display_dataset(name: str) -> str:
    return DATASET_DISPLAY.get(name, name)
