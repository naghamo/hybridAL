"""Exp 5 — Initial pool size sensitivity (upgraded).

Methods : Retrain + HybridAL
Backbone: DistilBERT
Datasets: 3 (imdb, agnews, sst2)
Seeds   : [42-46]
Rounds  : 25
Pool    : {50, 100, 200, 500}
Total   : 4 × 2 × 3 × 5 = 120 runs.
"""
import argparse
from pathlib import Path

from ..shared import constants as C
from ..shared.config_factory import build_cfg
from ..shared.plan import ExperimentPlan, make_plan_from_grid
from ..shared.signals import load_normalizers
from ._base import dispatch, make_runner_parser

NAME = "pool_size_ablation"
SAVE_ROOT = Path(f"experiments/{NAME}")


def add_extra_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--signal", required=True)
    p.add_argument("--epsilon", type=float, required=True)
    p.add_argument("--k", type=int, required=True)


def make_plan(args: argparse.Namespace) -> ExperimentPlan:
    return make_plan_from_grid(
        name=NAME,
        save_root=SAVE_ROOT,
        grid={
            "pool":    C.POOL_SIZE_GRID,
            "method":  ["Retrain", "HybridAL"],
            "dataset": C.DATASETS_3,
            "seed":    C.SEEDS_FULL,
        },
    )


def make_cfg(args: argparse.Namespace, entry: dict):
    pool = entry["pool"]
    method = entry["method"]
    dataset = entry["dataset"]
    seed = entry["seed"]
    if method == "Retrain":
        strategy_class, strategy_kwargs = "RetrainStrategy", {}
    else:

        strategy_class, strategy_kwargs = "DeltaF1Strategy", {
            "epsilon": args.epsilon, "k": args.k,
            "signal": args.signal, "signal_normalizer": None,
        }
    name = f"Pool_{method}_pool{pool}_{dataset}_seed{seed}"
    return build_cfg(
        experiment_name=name,
        save_dir=SAVE_ROOT,
        data=dataset,
        seed=seed,
        strategy_class=strategy_class,
        strategy_kwargs=strategy_kwargs,
        initial_pool_size=pool,
        total_rounds=25,
    )


def main(args: argparse.Namespace) -> None:
    plan = make_plan(args)
    dispatch(plan, args, make_cfg)


if __name__ == "__main__":
    parser = make_runner_parser("Exp 5 — Initial pool size ablation", add_extra_args)
    main(parser.parse_args())
