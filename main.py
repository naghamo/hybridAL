#!/usr/bin/env python3
"""Single CLI entry point for every Phase-2 experiment.

Replaces the per-experiment scripts in cli/ with one unified command-line:

    python main.py run <experiment> [experiment-specific args]
    python main.py aggregate <experiment> [--dry-run]
    python main.py list

`run` dispatches to `experiments_lib.runners.<experiment>.main(args)`,
`aggregate` to `experiments_lib.aggregators.<experiment>.main(args)`. The
runners themselves own their CLI arguments (`add_extra_args`); this script
only injects the shared flags (`--gpu`, `--dry-run`, `--rebalance-from-csv`)
and routes.

Code layout:
  adaptive_al/                — model, AL strategies, samplers, evaluation
  adaptive_al/utils/          — data loaders, common helpers
  experiments_lib/runners/    — one module per experiment that builds a plan
  experiments_lib/aggregators/— one module per experiment that aggregates results
  experiments_lib/shared/     — utility helpers shared by the above
                                (config factory, constants, signals, csv/plot/table I/O,
                                 plan dataclass, generic plan runner, ntfy push)
  experiments/                — output directory (one subdirectory per experiment)

Examples:
    # Phase-1 hyperparameter tuning of (ε, k) for the primary signal
    python main.py run hyperparameter \
        --signal delta_spectral_alpha --confirm-grid \
        --epsilon-grid 0.00005 0.00010 0.00015 0.00020

    # Phase-2 main results table (with both winning HybridAL signals)
    python main.py run main_results \
        --signal-alpha delta_spectral_alpha --epsilon-alpha 0.00015 --k-alpha 3 \
        --signal-acc   delta_accuracy       --epsilon-acc   0.015     --k-acc   3 \
        --gpu 0

    # Aggregate the results once runs finish
    python main.py aggregate main_results
"""
from __future__ import annotations

import argparse
import importlib
import sys

from experiments_lib.shared.runner import add_common_runner_args


# Experiment registry — single source of truth for which experiments exist
# and how they wire into the unified CLI. Each tuple is:
#   (cli_name, runner_module, optional_aggregator_module, description).
# Aggregator can be None if not yet wired (Exp 0 has no aggregator).
EXPERIMENTS = [
    ("n_sensitivity",          "n_sensitivity",          "n_sensitivity",
     "Exp 0 — N (random_subset_size) sensitivity"),
    ("hyperparameter",         "hyperparameter",         "hyperparameter",
     "Exp 2 — Hyperparameter tuning of (ε, k) for one HybridAL signal"),
    ("main_results",           "main_results",           "main_results",
     "Exp 3 — Main results table across 9 methods × 3 backbones × 6 datasets"),
    ("samplers",               "samplers",               "samplers",
     "Exp 4 — Sampler ablation (Random / Entropy / BADGE)"),
    ("pool_size",              "pool_size",              "pool_size",
     "Exp 5 — Initial pool size sensitivity"),
    ("batch_size",             "batch_size",             "batch_size",
     "Exp 6 — Acquisition batch size sensitivity"),
    ("early_stopping",         "early_stopping",         "early_stopping",
     "Exp 7 — Early stopping validation"),
    ("normalizer_sensitivity", "normalizer_sensitivity", "normalizer_sensitivity",
     "Exp 8 — Normalization sensitivity (appendix)"),
    ("calibration_sensitivity","calibration_sensitivity","calibration_sensitivity",
     "Exp 9 — Calibration dataset sensitivity (appendix)"),
]


def _build_run_parser(subparsers: argparse._SubParsersAction) -> None:
    """Attach `run <experiment> ...` subparsers, one per registered experiment."""
    run_p = subparsers.add_parser(
        "run",
        help="Launch one experiment (or its dry-run plan).",
        description="Launch a Phase-2 experiment runner. Each experiment "
                    "subcommand has its own flags — see e.g. `python main.py run "
                    "hyperparameter --help`.",
    )
    exp_sub = run_p.add_subparsers(dest="experiment", required=True,
                                   metavar="EXPERIMENT")
    for cli_name, runner_mod, _agg_mod, desc in EXPERIMENTS:
        sp = exp_sub.add_parser(cli_name, help=desc, description=desc)
        add_common_runner_args(sp)
        # Pull experiment-specific flags from the runner module.
        mod = importlib.import_module(f"experiments_lib.runners.{runner_mod}")
        if hasattr(mod, "add_extra_args"):
            mod.add_extra_args(sp)
        sp.set_defaults(_runner_module=runner_mod)


def _build_aggregate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Attach `aggregate <experiment>` subparsers."""
    agg_p = subparsers.add_parser(
        "aggregate",
        help="Aggregate one experiment's results into CSVs / plots / report.",
        description="Run the aggregator for a finished experiment (or its dry-run).",
    )
    agg_p.add_argument("experiment",
                       choices=[e[0] for e in EXPERIMENTS if e[2] is not None],
                       help="Which experiment to aggregate.")
    agg_p.add_argument("--dry-run", action="store_true",
                       help="Print how many runs would be aggregated; don't write outputs.")


def _build_list_parser(subparsers: argparse._SubParsersAction) -> None:
    """Attach `list` subcommand — prints the experiment registry."""
    subparsers.add_parser(
        "list",
        help="Print the experiment registry.",
        description="Print every available experiment and its short description.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse tree for `main.py`."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Unified entry point for HybridAL Phase-2 experiments.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    _build_run_parser(sub)
    _build_aggregate_parser(sub)
    _build_list_parser(sub)
    return parser


def cmd_list() -> int:
    """Print one row per registered experiment."""
    print(f"{'name':<26}  description")
    print("-" * 78)
    for cli_name, _runner, _agg, desc in EXPERIMENTS:
        print(f"{cli_name:<26}  {desc}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Dispatch `run <experiment>` to that experiment's runner module."""
    mod = importlib.import_module(f"experiments_lib.runners.{args._runner_module}")
    mod.main(args)
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    """Dispatch `aggregate <experiment>` to that experiment's aggregator module."""
    mod = importlib.import_module(f"experiments_lib.aggregators.{args.experiment}")
    mod.main(args)
    return 0


def main() -> int:
    """Entry point — parse CLI args, dispatch to the right handler."""
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "list":
        return cmd_list()
    if args.command == "run":
        return cmd_run(args)
    if args.command == "aggregate":
        return cmd_aggregate(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
