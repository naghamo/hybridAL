"""Exp 2 — hyperparameter tuning of (ε, k) for the winning signal.

Algorithm:
  1. Read the signal-ablation `_per_round.csv`, compute the winning signal's
     mean normalized trajectory across (dataset × seed).
  2. Auto-derive 4 ε values that bracket switching at rounds {5, 10, 18, 25}.
     Print the rationale.
  3. Unless `--confirm-grid` is passed, STOP and ask the user to confirm/edit.
  4. Run the 16-cell grid (ε × k) on IMDb + AG News × 3 seeds × 15 rounds.
"""
import argparse
import json
import sys
from pathlib import Path

from ..shared import constants as C
from ..shared.config_factory import build_cfg
from ..shared.plan import ExperimentPlan, make_plan_from_grid
from ..shared.signals import (
    derive_epsilon_grid,
    load_normalizers,
)
from ._base import dispatch, make_runner_parser

NAME = "hyperparameter_tuning"
SAVE_ROOT = Path(f"experiments/{NAME}")
PER_ROUND_CSV = Path(f"{C.SIGNAL_ABLATION_RESULTS_DIR}/_per_round.csv")


def add_extra_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--signal", required=True,
                   help="Winning signal name from the signal ablation.")
    p.add_argument("--confirm-grid", action="store_true",
                   help="Skip the human-in-the-loop pause and use the auto-derived ε grid.")
    p.add_argument("--epsilon-grid", type=float, nargs="+", default=None,
                   help="Override: 4 explicit ε values. Skips trajectory analysis.")
    p.add_argument("--per-round-csv", default=str(PER_ROUND_CSV),
                   help="Path to signal-ablation _per_round.csv used for ε auto-derivation.")


def _derive_or_accept_epsilon_grid(args: argparse.Namespace) -> list:
    if args.epsilon_grid is not None and len(args.epsilon_grid) == 4:
        print(f"[hp-tuning] using user-supplied ε grid: {args.epsilon_grid}")
        return args.epsilon_grid

    normalizers = load_normalizers()
    if args.signal not in normalizers:
        sys.exit(f"signal {args.signal!r} not in {C.CALIBRATION_NORMALIZERS_PATH}; "
                 f"available: {list(normalizers.keys())}")
    normalizer = normalizers[args.signal]

    csv_path = Path(args.per_round_csv)
    if not csv_path.exists():
        sys.exit(f"missing {csv_path} — signal ablation aggregator must run first.")

    info = derive_epsilon_grid(
        per_round_csv=csv_path,
        signal=args.signal,
        normalizer=normalizer,
        target_rounds=[5, 10, 18, 25],
        k=3,
    )
    print(f"\n[hp-tuning] auto-derived ε grid for signal={args.signal!r}, "
          f"normalizer={normalizer:g}:")
    print(info["rationale"])
    grid = info["epsilon_grid"]
    print(f"\n[hp-tuning] proposed ε grid: {grid}")

    if not args.confirm_grid and not args.dry_run:
        print("\n[hp-tuning] NOT confirmed — re-run with --confirm-grid to launch, or "
              "pass --epsilon-grid e1 e2 e3 e4 to override.")
        sys.exit(0)
    return grid


def make_plan(args: argparse.Namespace) -> ExperimentPlan:
    if args.dry_run and (args.epsilon_grid is None and not args.confirm_grid):


        eps_grid = [0.05, 0.1, 0.2, 0.5]
        print("[hp-tuning] DRY RUN with placeholder ε grid:", eps_grid)
    else:
        eps_grid = _derive_or_accept_epsilon_grid(args)

    return make_plan_from_grid(
        name=NAME,
        save_root=SAVE_ROOT,
        grid={
            "epsilon": eps_grid,
            "k":       C.K_GRID_DEFAULT,
            "dataset": C.DATASETS_TUNING_2,
            "seed":    C.SEEDS_TUNING,
        },
    )


def make_cfg(args: argparse.Namespace, entry: dict):
    eps = entry["epsilon"]
    k = entry["k"]
    dataset = entry["dataset"]
    seed = entry["seed"]


    name = f"HP_{args.signal}_eps{eps}_k{k}_{dataset}_seed{seed}"
    return build_cfg(
        experiment_name=name,
        save_dir=SAVE_ROOT,
        data=dataset,
        seed=seed,
        strategy_class="DeltaF1Strategy",
        strategy_kwargs={
            "epsilon": eps,
            "k": k,
            "signal": args.signal,
            "signal_normalizer": None,
        },
        total_rounds=15,
    )


def main(args: argparse.Namespace) -> None:
    plan = make_plan(args)
    dispatch(plan, args, make_cfg)


if __name__ == "__main__":
    parser = make_runner_parser("Exp 2 — Hyperparameter tuning of (ε, k)", add_extra_args)
    main(parser.parse_args())
