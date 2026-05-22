"""Exp — Calibration re-evaluation with logit saving.

Re-runs a subset of the main-results grid (1 backbone, 5 methods) with
the patched evaluation code that writes test/val logits to .npy
sidecars. Used by `scratch/_compute_ece_and_temp_nll.py` to compute
Expected Calibration Error (ECE) and post-temperature-scaling NLL
without retraining further.

Grid: 5 methods x 1 backbone (DistilBERT) x 6 datasets x 5 seeds = 150
runs. Same hyperparameters as `main_results.py` so the saved logits
correspond to the same models reported in the main paper.

CLI args (same as main_results):
  --signal-alpha     name of the primary HybridAL signal
  --epsilon-alpha    raw eps for the primary signal
  --k-alpha          k for the primary signal
  --signal-acc       name of the secondary HybridAL signal
  --epsilon-acc      raw eps for the secondary signal
  --k-acc            k for the secondary signal
"""
import argparse
from pathlib import Path

from ..shared import constants as C
from ..shared.config_factory import build_cfg
from ..shared.plan import ExperimentPlan, make_plan_from_grid
from ._base import dispatch, make_runner_parser

NAME = "calibration_eval"
SAVE_ROOT = Path(f"experiments/{NAME}")

METHODS = [
    "Retrain",
    "FineTune",
    "NewOnly",
    "HybridAL_alpha",
    "HybridAL_acc",
]

BACKBONE = "distilbert-base-uncased"


def add_extra_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--signal-alpha", required=True)
    p.add_argument("--epsilon-alpha", type=float, required=True)
    p.add_argument("--k-alpha", type=int, required=True)
    p.add_argument("--signal-acc", required=True)
    p.add_argument("--epsilon-acc", type=float, required=True)
    p.add_argument("--k-acc", type=int, required=True)


def make_plan(args: argparse.Namespace) -> ExperimentPlan:
    return make_plan_from_grid(
        name=NAME,
        save_root=SAVE_ROOT,
        grid={
            "method":   METHODS,
            "backbone": [BACKBONE],
            "dataset":  C.ALL_DATASETS,
            "seed":     C.SEEDS_FULL,
        },
    )


def _strategy_for(method: str, args: argparse.Namespace):
    if method == "Retrain":
        return "RetrainStrategy", {}
    if method == "FineTune":
        return "FineTuneStrategy", {}
    if method == "NewOnly":
        return "NewOnlyStrategy", {}
    if method == "HybridAL_alpha":
        return "DeltaF1Strategy", {
            "epsilon": args.epsilon_alpha, "k": args.k_alpha,
            "signal": args.signal_alpha, "signal_normalizer": None,
        }
    if method == "HybridAL_acc":
        return "DeltaF1Strategy", {
            "epsilon": args.epsilon_acc, "k": args.k_acc,
            "signal": args.signal_acc, "signal_normalizer": None,
        }
    raise ValueError(f"unknown method {method!r}")


def make_cfg(args: argparse.Namespace, entry: dict):
    method = entry["method"]
    backbone = entry["backbone"]
    dataset = entry["dataset"]
    seed = entry["seed"]
    strategy_class, strategy_kwargs = _strategy_for(method, args)
    backbone_short = backbone.replace("/", "_").replace("-", "_")
    name = f"Cal_{method}_{backbone_short}_{dataset}_seed{seed}"
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
    parser = make_runner_parser("Calibration re-eval (logit saving)", add_extra_args)
    main(parser.parse_args())
