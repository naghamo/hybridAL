"""Exp 7 — Early stopping validation.

Configs:
  (A) Fixed 5 epochs, no ES → epochs=5, early_stopping_patience=999.
      Note: best-state restore is still applied by BaseStrategy. Future
      `restore_best_state=False` flag will give a perfectly faithful "no
      ES" comparison.
  (B) Current setup → epochs=10, patience=2.

Methods : Retrain + FineTune
Backbone: DistilBERT
Datasets: imdb, agnews
Seeds   : [42, 43, 44]
Rounds  : 15
Total   : 2 × 2 × 2 × 3 = 24 runs.
"""
import argparse
from pathlib import Path

from ..shared import constants as C
from ..shared.config_factory import build_cfg
from ..shared.plan import ExperimentPlan, make_plan_from_grid
from ._base import dispatch, make_runner_parser

NAME = "early_stopping_validation"
SAVE_ROOT = Path(f"experiments/{NAME}")


def make_plan(args: argparse.Namespace) -> ExperimentPlan:
    return make_plan_from_grid(
        name=NAME,
        save_root=SAVE_ROOT,
        grid={
            "config":  ["A_fixed5", "B_es_max10"],
            "method":  ["Retrain", "FineTune"],
            "dataset": C.DATASETS_TUNING_2,
            "seed":    C.SEEDS_TUNING,
        },
    )


def make_cfg(args: argparse.Namespace, entry: dict):
    cfg_label = entry["config"]
    method = entry["method"]
    dataset = entry["dataset"]
    seed = entry["seed"]

    if cfg_label == "A_fixed5":
        epochs, patience = 5, 999
    else:
        epochs, patience = 10, 2

    if method == "Retrain":
        strategy_class = "RetrainStrategy"
    else:
        strategy_class = "FineTuneStrategy"

    name = f"ES_{cfg_label}_{method}_{dataset}_seed{seed}"
    return build_cfg(
        experiment_name=name,
        save_dir=SAVE_ROOT,
        data=dataset,
        seed=seed,
        strategy_class=strategy_class,
        strategy_kwargs={},
        epochs=epochs,
        early_stopping_patience=patience,
        total_rounds=15,
    )


def main(args: argparse.Namespace) -> None:
    plan = make_plan(args)
    dispatch(plan, args, make_cfg)


if __name__ == "__main__":
    parser = make_runner_parser("Exp 7 — Early stopping validation")
    main(parser.parse_args())
