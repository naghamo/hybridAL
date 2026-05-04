"""Exp 9 — Calibration dataset sensitivity.

Sources (extracted from existing JSONs — no new calibration runs):
  (A) IMDb seed 42 (current default; from `_calibration_normalizers.json`).
  (B) AG News seed 42 (extracted from
      `experiments/signal_ablation/Retrain_agnews_seed42_N1000/results_*.json`).
  (C) Average of A and B per-signal.

Methods : HybridAL
Backbone: DistilBERT
Datasets: imdb, agnews (seen) + jigsaw, sst2 (unseen)
Seeds   : [42, 43, 44]
Rounds  : 15
Total   : 3 × 4 × 3 = 36 runs.

The `--dry-run` mode asserts that the source-B JSON exists; if not, it
prints "DEFER: AG News calibration source unavailable until signal-ablation
watchdog completes" and exits non-zero.
"""
import argparse
from pathlib import Path
from typing import Dict, Optional

from ..shared import constants as C
from ..shared.config_factory import build_cfg
from ..shared.plan import ExperimentPlan, make_plan_from_grid
from ..shared.signals import (
    SUPPORTED_SIGNALS,
    compute_max_2_5_from_calibration_json,
    load_normalizers,
)
from ._base import dispatch, make_runner_parser

NAME = "calibration_sensitivity"
SAVE_ROOT = Path(f"experiments/{NAME}")

AGNEWS_SOURCE_GLOB = "experiments/signal_ablation/Retrain_agnews_seed42_N1000/results_*.json"


def add_extra_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--signal", required=True)
    p.add_argument("--epsilon", type=float, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--source-b-glob", default=AGNEWS_SOURCE_GLOB)


def _resolve_source_normalizers(args: argparse.Namespace) -> Dict[str, Dict[str, Optional[float]]]:
    """Return {source: {signal: normalizer}} for the 3 calibration sources."""
    out: Dict[str, Dict[str, Optional[float]]] = {}

    # Source A — current IMDb seed-42 calibration JSON.
    out["A"] = dict(load_normalizers())

    # Source B — AG News seed-42 from in-flight Retrain run.
    candidates = sorted(Path(".").glob(args.source_b_glob))
    if not candidates:
        if args.dry_run:
            print(f"[calib-sensitivity] DEFER: AG News calibration source not yet "
                  f"available. Expected {args.source_b_glob}.")
        out["B"] = {s: None for s in SUPPORTED_SIGNALS}
    else:
        out["B"] = compute_max_2_5_from_calibration_json(candidates[-1])

    # Source C — per-signal average of A and B.
    avg: Dict[str, Optional[float]] = {}
    for s in SUPPORTED_SIGNALS:
        a = out["A"].get(s)
        b = out["B"].get(s)
        if a is not None and b is not None:
            avg[s] = (a + b) / 2.0
        elif a is not None:
            avg[s] = a
        elif b is not None:
            avg[s] = b
        else:
            avg[s] = None
    out["C"] = avg
    return out


def make_plan(args: argparse.Namespace) -> ExperimentPlan:
    return make_plan_from_grid(
        name=NAME,
        save_root=SAVE_ROOT,
        grid={
            "source":  ["A", "B", "C"],
            "dataset": C.DATASETS_CALIB_4,
            "seed":    C.SEEDS_TUNING,
        },
    )


def make_cfg(args: argparse.Namespace, entry: dict):
    source = entry["source"]
    dataset = entry["dataset"]
    seed = entry["seed"]
    sources = _resolve_source_normalizers(args)
    normalizer = sources[source].get(args.signal)
    if normalizer is None:
        raise SystemExit(
            f"source {source!r} has no normalizer for signal {args.signal!r}. "
            f"For source B, ensure {args.source_b_glob} exists "
            f"(populated by the signal-ablation Retrain runs)."
        )
    name = f"Calib_{source}_{args.signal}_{dataset}_seed{seed}"
    return build_cfg(
        experiment_name=name,
        save_dir=SAVE_ROOT,
        data=dataset,
        seed=seed,
        strategy_class="DeltaF1Strategy",
        strategy_kwargs={
            "epsilon": args.epsilon, "k": args.k,
            "signal": args.signal, "signal_normalizer": normalizer,
        },
        total_rounds=15,
    )


def main(args: argparse.Namespace) -> None:
    plan = make_plan(args)
    dispatch(plan, args, make_cfg)


if __name__ == "__main__":
    parser = make_runner_parser("Exp 9 — Calibration dataset sensitivity", add_extra_args)
    main(parser.parse_args())
