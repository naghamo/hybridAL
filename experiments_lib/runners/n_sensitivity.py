"""Exp 0 — random_subset_size (N) sensitivity for the EntropyOnRandomSubsetSampler.

Methods : RetrainStrategy
Sampler : EntropyOnRandomSubsetSampler with N ∈ {100, 500, 1000, 5000}
Backbone: DistilBERT
Datasets: imdb, agnews
Seeds   : [42, 43, 44]
Rounds  : 15
Total   : 4 × 1 × 2 × 3 = 24 runs
"""
import argparse
from pathlib import Path

from ..shared import constants as C
from ..shared.config_factory import build_cfg
from ..shared.plan import ExperimentPlan, make_plan_from_grid
from ._base import dispatch, make_runner_parser

NAME = "n_sensitivity"
SAVE_ROOT = Path(f"experiments/{NAME}")


def make_plan(args: argparse.Namespace) -> ExperimentPlan:
    return make_plan_from_grid(
        name=NAME,
        save_root=SAVE_ROOT,
        grid={
            "n":       C.N_GRID_DEFAULT,
            "dataset": C.DATASETS_TUNING_2,
            "seed":    C.SEEDS_TUNING,
        },
    )


def make_cfg(args: argparse.Namespace, entry: dict):
    n = entry["n"]
    dataset = entry["dataset"]
    seed = entry["seed"]
    name = f"NSens_n{n}_{dataset}_seed{seed}"
    return build_cfg(
        experiment_name=name,
        save_dir=SAVE_ROOT,
        data=dataset,
        seed=seed,
        strategy_class="RetrainStrategy",
        strategy_kwargs={},
        sampler_class="EntropyOnRandomSubsetSampler",
        sampler_kwargs={"random_subset_size": n, "seed": seed},
        random_subset_size=n,
        total_rounds=15,
    )


def main(args: argparse.Namespace) -> None:
    plan = make_plan(args)
    dispatch(plan, args, make_cfg)


if __name__ == "__main__":
    parser = make_runner_parser("Exp 0 — N (random_subset_size) sensitivity")
    main(parser.parse_args())
