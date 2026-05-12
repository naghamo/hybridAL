"""Exp 4 — Sampler ablation.

Methods : Retrain + HybridAL                       = 2
Samplers: Random, EntropyOnRandomSubset, BADGE      = 3
Backbone: DistilBERT
Datasets: all 6
Seeds   : [42-46]
Rounds  : 25
Total   : 3 × 2 × 6 × 5 = 180 runs.

Sampler kwargs (locked for fairness — same N=1000 pre-filter where
applicable):
  - RandomSampler:                {"seed": <seed>}
  - EntropyOnRandomSubsetSampler: {"random_subset_size": 1000, "seed": <seed>}
  - BADGESampler:                 {"random_subset_size": 1000, "seed": <seed>}
"""
import argparse
from pathlib import Path

from ..shared import constants as C
from ..shared.config_factory import build_cfg
from ..shared.plan import ExperimentPlan, make_plan_from_grid
from ..shared.signals import load_normalizers
from ._base import dispatch, make_runner_parser

NAME = "sampler_ablation"
SAVE_ROOT = Path(f"experiments/{NAME}")

SAMPLERS = ["RandomSampler", "EntropyOnRandomSubsetSampler", "BADGESampler"]


def add_extra_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--signal", required=True)
    p.add_argument("--epsilon", type=float, required=True)
    p.add_argument("--k", type=int, required=True)


def make_plan(args: argparse.Namespace) -> ExperimentPlan:
    # Datasets reduced to the same 3 used by Exp 5 (pool size) and Exp 6
    # (batch size) — IMDb / AG News / Yahoo Answers — covering the full
    # label-cardinality range (binary / 4-class / 10-class).
    return make_plan_from_grid(
        name=NAME,
        save_root=SAVE_ROOT,
        grid={
            "method":  ["Retrain", "HybridAL"],
            "sampler": SAMPLERS,
            "dataset": C.DATASETS_3,
            "seed":    C.SEEDS_FULL,
        },
    )


def _sampler_kwargs(name: str, seed: int) -> dict:
    if name == "RandomSampler":
        return {"seed": seed}
    return {"random_subset_size": 1000, "seed": seed}


def make_cfg(args: argparse.Namespace, entry: dict):
    method = entry["method"]
    sampler = entry["sampler"]
    dataset = entry["dataset"]
    seed = entry["seed"]
    if method == "Retrain":
        strategy_class, strategy_kwargs = "RetrainStrategy", {}
    else:
        # Phase-2: ε at the signal's raw scale, no calibration normalizer.
        strategy_class, strategy_kwargs = "DeltaF1Strategy", {
            "epsilon": args.epsilon, "k": args.k,
            "signal": args.signal, "signal_normalizer": None,
        }
    name = f"Sampler_{method}_{sampler}_{dataset}_seed{seed}"
    return build_cfg(
        experiment_name=name,
        save_dir=SAVE_ROOT,
        data=dataset,
        seed=seed,
        strategy_class=strategy_class,
        strategy_kwargs=strategy_kwargs,
        sampler_class=sampler,
        sampler_kwargs=_sampler_kwargs(sampler, seed),
        total_rounds=25,
    )


def main(args: argparse.Namespace) -> None:
    plan = make_plan(args)
    dispatch(plan, args, make_cfg)


if __name__ == "__main__":
    parser = make_runner_parser("Exp 4 — Sampler ablation", add_extra_args)
    main(parser.parse_args())
