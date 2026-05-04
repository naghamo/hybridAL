"""Exp 3 — Main results.

Methods : Retrain, FineTune, NewOnly,
          HybridAL_alpha (delta_spectral_alpha, ε_alpha, k_alpha),
          HybridAL_acc   (delta_accuracy,       ε_acc,   k_acc),
          FixedSwitch@{3,5,7,10}                              = 9 methods
Backbones: distilbert/bert/roberta-base                       = 3 backbones
Datasets : all 6
Seeds    : [42-46]
Rounds   : 25
Total    : 9 × 3 × 6 × 5 = 810 runs.

GPU split: standard 3-vs-3 dataset split. Each GPU runs ~405 configs.

CLI args:
  --signal-alpha     name of the primary HybridAL signal (e.g. delta_spectral_alpha)
  --epsilon-alpha    raw ε for the primary signal
  --k-alpha          k for the primary signal
  --signal-acc       name of the secondary HybridAL signal (e.g. delta_accuracy)
  --epsilon-acc      raw ε for the secondary signal
  --k-acc            k for the secondary signal
"""
import argparse
from pathlib import Path

from ..shared import constants as C
from ..shared.config_factory import build_cfg
from ..shared.plan import ExperimentPlan, make_plan_from_grid
from ._base import dispatch, make_runner_parser

NAME = "main_results"
SAVE_ROOT = Path(f"experiments/{NAME}")

METHODS = [
    "Retrain",
    "FineTune",
    "NewOnly",
    "HybridAL_alpha",
    "HybridAL_acc",
    "FixedSwitch_3",
    "FixedSwitch_5",
    "FixedSwitch_7",
    "FixedSwitch_10",
]


def add_extra_args(p: argparse.ArgumentParser) -> None:
    # Primary HybridAL signal
    p.add_argument("--signal-alpha", required=True,
                   help="Primary HybridAL signal name (e.g. delta_spectral_alpha).")
    p.add_argument("--epsilon-alpha", type=float, required=True,
                   help="Raw ε for the primary signal (no calibration normalizer).")
    p.add_argument("--k-alpha", type=int, required=True,
                   help="k for the primary signal.")
    # Secondary HybridAL signal
    p.add_argument("--signal-acc", required=True,
                   help="Secondary HybridAL signal name (e.g. delta_accuracy).")
    p.add_argument("--epsilon-acc", type=float, required=True,
                   help="Raw ε for the secondary signal.")
    p.add_argument("--k-acc", type=int, required=True,
                   help="k for the secondary signal.")


def make_plan(args: argparse.Namespace) -> ExperimentPlan:
    return make_plan_from_grid(
        name=NAME,
        save_root=SAVE_ROOT,
        grid={
            "method":   METHODS,
            "backbone": C.BACKBONES_ALL,
            "dataset":  C.ALL_DATASETS,
            "seed":     C.SEEDS_FULL,
        },
    )


def _strategy_for(method: str, args: argparse.Namespace):
    """Return (strategy_class, strategy_kwargs) for the given method."""
    if method == "Retrain":
        return "RetrainStrategy", {}
    if method == "FineTune":
        return "FineTuneStrategy", {}
    if method == "NewOnly":
        return "NewOnlyStrategy", {}
    if method == "HybridAL_alpha":
        # Phase-2: ε at signal's raw scale, no calibration normalizer.
        return "DeltaF1Strategy", {
            "epsilon": args.epsilon_alpha,
            "k": args.k_alpha,
            "signal": args.signal_alpha,
            "signal_normalizer": None,
        }
    if method == "HybridAL_acc":
        return "DeltaF1Strategy", {
            "epsilon": args.epsilon_acc,
            "k": args.k_acc,
            "signal": args.signal_acc,
            "signal_normalizer": None,
        }
    if method.startswith("FixedSwitch_"):
        sw = int(method.split("_")[1])
        return "FixedSwitchStrategy", {"switch_round": sw}
    raise ValueError(f"unknown method {method!r}")


def make_cfg(args: argparse.Namespace, entry: dict):
    method = entry["method"]
    backbone = entry["backbone"]
    dataset = entry["dataset"]
    seed = entry["seed"]
    strategy_class, strategy_kwargs = _strategy_for(method, args)
    backbone_short = backbone.replace("/", "_").replace("-", "_")
    name = f"Main_{method}_{backbone_short}_{dataset}_seed{seed}"
    return build_cfg(
        experiment_name=name,
        save_dir=SAVE_ROOT,
        data=dataset,
        seed=seed,
        strategy_class=strategy_class,
        strategy_kwargs=strategy_kwargs,
        model_name_or_path=backbone,
        total_rounds=25,
    )


def main(args: argparse.Namespace) -> None:
    plan = make_plan(args)
    dispatch(plan, args, make_cfg)


if __name__ == "__main__":
    parser = make_runner_parser("Exp 3 — Main results", add_extra_args)
    main(parser.parse_args())
