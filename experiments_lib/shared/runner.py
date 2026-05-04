"""Generic plan executor for Phase-2 experiments.

Replaces the bespoke loop in `_ablation_runner.py` with a reusable function
each experiment runner calls with its own `make_cfg(args, entry)` callback.
"""
import argparse
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .plan import ExperimentPlan, already_done, split_for_gpu


def add_common_runner_args(parser: argparse.ArgumentParser) -> None:
    """Add the flags every Phase-2 runner CLI script uses."""
    parser.add_argument("--gpu", type=int, choices=[0, 1], default=None,
                        help="Which GPU's slice to run (default = whole plan).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned configs and exit. No GPU.")
    parser.add_argument("--rebalance-from-csv", type=str, default=None,
                        help="Path to a previous experiment's _summary.csv. "
                             "If given, rebalance which datasets go on which GPU "
                             "by mean training time. Default = use GPU_DATASETS_DEFAULT.")
    parser.add_argument("--split-mode", choices=["dataset", "index"], default="dataset",
                        help="How to split the plan across GPUs. 'dataset' assigns "
                             "by dataset (default; balanced when the experiment uses "
                             "all 6). 'index' round-robins by entry index — use for "
                             "small experiments (Exp 5/6/7) whose 2-3 datasets don't "
                             "split evenly across the default GPU partition.")


def _print_plan_dry_run(plan: ExperimentPlan,
                        slice_entries: List[Dict[str, Any]],
                        args: argparse.Namespace,
                        cfg_preview: Callable[[Dict[str, Any]], str],
                        ) -> None:
    print(f"\n=== DRY RUN: {plan.name} ===")
    print(f"GPU: {args.gpu if args.gpu is not None else 'both'}")
    print(f"plan size: {len(plan.entries)}")
    print(f"this slice: {len(slice_entries)}")
    print(f"save_root:  {plan.save_root}")
    print()
    print("first 5 (or fewer) entries:")
    for i, e in enumerate(slice_entries[:5], 1):
        print(f"  [{i}] {e}")
        print(f"      → {cfg_preview(e)}")
    if len(slice_entries) > 5:
        print(f"  ... ({len(slice_entries) - 5} more)")
    print()
    est_minutes = len(slice_entries) * 6  # ~6 min/run heuristic
    print(f"estimated wall-clock @ 6 min/run on this GPU: ~{est_minutes / 60:.1f} h")


def run_plan_on_gpu(plan: ExperimentPlan,
                    args: argparse.Namespace,
                    make_cfg: Callable[[argparse.Namespace, Dict[str, Any]], Any],
                    ) -> None:
    """Execute the plan slice that belongs to `args.gpu`.

    `make_cfg(args, entry)` must return an `ExperimentConfig`.
    Already-done runs (per `already_done(name, save_root)`) are skipped.
    Honors `args.dry_run` (prints, never imports torch).
    """
    from . import constants as C

    gpu_datasets = C.GPU_DATASETS_DEFAULT
    if args.rebalance_from_csv:
        gpu_datasets = _rebalance_gpus_from_summary_csv(
            args.rebalance_from_csv, default=gpu_datasets,
        )

    if args.gpu is None:
        slice_entries = list(plan.entries)
        gpu_label = "whole-plan"
    else:
        split_mode = getattr(args, "split_mode", "dataset")
        slice_entries = split_for_gpu(plan, args.gpu, gpu_datasets,
                                       mode=split_mode)
        gpu_label = f"gpu{args.gpu}"

    def _cfg_preview(entry: Dict[str, Any]) -> str:
        # Defer to the runner — we don't want to construct ExperimentConfig
        # in dry-run because that pulls in torch via adaptive_al imports.
        # Just describe the entry's experiment name (lazy import).
        try:
            cfg = make_cfg(args, entry)
            return f"experiment_name={cfg.experiment_name}  data={cfg.data}  seed={cfg.seed}"
        except Exception as e:
            return f"<cfg-build-error: {e}>"

    if args.dry_run:
        _print_plan_dry_run(plan, slice_entries, args, _cfg_preview)
        return

    # Actual execution path — imports torch only here.
    from adaptive_al.active_learning import ActiveLearning  # noqa: E402

    plan.save_root.mkdir(parents=True, exist_ok=True)

    print(f"[{gpu_label}] {len(slice_entries)} runs from plan {plan.name!r}")
    t_total = time.time()
    for idx, entry in enumerate(slice_entries, 1):
        cfg = make_cfg(args, entry)
        if already_done(cfg.experiment_name, plan.save_root):
            print(f"[{gpu_label}] [{idx}/{len(slice_entries)}] SKIP "
                  f"(already done): {cfg.experiment_name}", flush=True)
            continue
        t0 = time.time()
        print(f"[{gpu_label}] [{idx}/{len(slice_entries)}] START {cfg.experiment_name}",
              flush=True)
        try:
            al = ActiveLearning(cfg)
            metrics = al.run_full_pipeline()
            al.save_experiment()
            sw = getattr(al.strategy, "switch_round", None)
            print(f"[{gpu_label}] [{idx}/{len(slice_entries)}] DONE "
                  f"{cfg.experiment_name}  test_F1={metrics['f1_score']:.4f}  "
                  f"switch_round={sw}  elapsed={time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            print(f"[{gpu_label}] [{idx}/{len(slice_entries)}] FAILED "
                  f"{cfg.experiment_name}: {e}", flush=True)
            traceback.print_exc()
    print(f"[{gpu_label}] all done in {(time.time() - t_total) / 3600:.2f}h")


def _rebalance_gpus_from_summary_csv(csv_path: str,
                                     default: Dict[int, List[str]]) -> Dict[int, List[str]]:
    """Read a previous run's `_summary.csv` and re-partition the 6 datasets
    into two halves of similar total wall-clock time.

    If imbalance ≤ 20%, returns the default split.
    """
    from .csv_io import read_csv, to_float
    rows = read_csv(Path(csv_path))
    if not rows:
        print(f"[runner] {csv_path} empty/missing — keeping default GPU split.")
        return default

    # Sum training_time per dataset
    by_ds: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for r in rows:
        ds = r.get("dataset")
        t = to_float(r.get("training_time_total"))
        if ds is None or t is None:
            continue
        by_ds[ds] = by_ds.get(ds, 0.0) + t
        counts[ds] = counts.get(ds, 0) + 1
    means = {ds: by_ds[ds] / counts[ds] for ds in by_ds if counts[ds] > 0}

    # If the existing default already balances within 20%, keep it.
    cur_g0 = sum(means.get(d, 0.0) for d in default[0])
    cur_g1 = sum(means.get(d, 0.0) for d in default[1])
    total = cur_g0 + cur_g1
    if total <= 0:
        return default
    imbalance = abs(cur_g0 - cur_g1) / total
    if imbalance <= 0.20:
        print(f"[runner] default split is balanced (imbalance={imbalance:.0%}). "
              f"GPU0={default[0]}  GPU1={default[1]}.")
        return default

    # Greedy rebalance: assign each dataset (descending mean time) to the
    # currently-lighter GPU.
    sorted_ds = sorted(means.items(), key=lambda x: -x[1])
    g0: List[str] = []
    g1: List[str] = []
    s0 = 0.0
    s1 = 0.0
    for ds, t in sorted_ds:
        if s0 <= s1:
            g0.append(ds)
            s0 += t
        else:
            g1.append(ds)
            s1 += t
    new_split = {0: g0, 1: g1}
    new_imbalance = abs(s0 - s1) / (s0 + s1) if (s0 + s1) > 0 else 0
    print(f"[runner] rebalanced GPU split (default imbalance={imbalance:.0%}, "
          f"new imbalance={new_imbalance:.0%}). GPU0={g0}  GPU1={g1}.")
    return new_split
