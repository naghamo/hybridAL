"""Exp 8 — Normalization sensitivity for the chosen HybridAL signal.

Recipes (recomputed from the existing IMDb seed-42 calibration JSON; no
new calibration runs needed):
  (A) round-2 value
  (B) mean of rounds 2-5
  (C) max of rounds 2-5     (current default in `_calibration_normalizers.json`)

Methods : HybridAL
Backbone: DistilBERT
Datasets: imdb, agnews
Seeds   : [42, 43, 44]
Rounds  : 15
Total   : 3 × 2 × 3 = 18 runs.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean as _mean
from typing import Dict, Optional

from ..shared import constants as C
from ..shared.config_factory import build_cfg
from ..shared.csv_io import load_json
from ..shared.plan import ExperimentPlan, make_plan_from_grid
from ..shared.signals import load_normalizers, to_switching_value
from ._base import dispatch, make_runner_parser

NAME = "normalizer_sensitivity"
SAVE_ROOT = Path(f"experiments/{NAME}")

# Default location for IMDb seed-42 calibration result (post-rerun).
CALIB_RESULT = Path("_calibration_results/calibration_imdb_seed42_v2")


def add_extra_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--signal", required=True)
    p.add_argument("--epsilon", type=float, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--calibration-result-dir", default=str(CALIB_RESULT),
                   help="Directory containing the IMDb seed-42 calibration "
                        "results_*.json. Used to recompute recipes A and B.")


def _values_in_window(rounds, signal_name, lo: int, hi: int):
    out = []
    for ridx, r in enumerate(rounds, 1):
        if not (lo <= ridx <= hi):
            continue
        sig_dict = r.get("signals") or {}
        raw = sig_dict.get(signal_name)
        sw = to_switching_value(signal_name, raw)
        if sw is None:
            continue
        try:
            if not math.isfinite(sw) or sw <= 0:
                continue
        except TypeError:
            continue
        out.append(float(sw))
    return out


def _compute_recipes_for_signal(signal_name: str,
                                calibration_dir: Path) -> Dict[str, Optional[float]]:
    """Return {"A": round-2 value, "B": mean(2..5), "C": max(2..5)}."""
    result_files = sorted(Path(calibration_dir).glob("results_*.json"))
    if not result_files:
        return {"A": None, "B": None, "C": None}
    obj = json.loads(result_files[-1].read_text())
    rounds = obj.get("round_val_stats", [])

    r2 = _values_in_window(rounds, signal_name, 2, 2)
    r25 = _values_in_window(rounds, signal_name, 2, 5)
    return {
        "A": (r2[0] if r2 else None),
        "B": (_mean(r25) if r25 else None),
        "C": (max(r25) if r25 else None),
    }


def make_plan(args: argparse.Namespace) -> ExperimentPlan:
    return make_plan_from_grid(
        name=NAME,
        save_root=SAVE_ROOT,
        grid={
            "recipe":  ["A", "B", "C"],
            "dataset": C.DATASETS_TUNING_2,
            "seed":    C.SEEDS_TUNING,
        },
    )


def make_cfg(args: argparse.Namespace, entry: dict):
    recipe = entry["recipe"]
    dataset = entry["dataset"]
    seed = entry["seed"]
    recipes = _compute_recipes_for_signal(args.signal, Path(args.calibration_result_dir))
    normalizer = recipes.get(recipe)
    if normalizer is None:
        raise SystemExit(
            f"recipe {recipe!r} produced no normalizer for signal {args.signal!r}. "
            f"Check calibration result at {args.calibration_result_dir}."
        )
    name = f"Norm_{recipe}_{args.signal}_{dataset}_seed{seed}"
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
    parser = make_runner_parser("Exp 8 — Normalization sensitivity", add_extra_args)
    main(parser.parse_args())
