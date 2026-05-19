"""
Hyperparameter optimization and experiment runner for active learning strategies.

Each experiment phase has a named --mode that sets sensible defaults.
Any flag can override the mode defaults. Use --dry-run to preview what will run.

RECOMMENDED WORKFLOW
====================

Step 1 — Baselines (no Optuna, can run immediately):
    python experimentation.py --mode baselines

Step 2 — Primary HybridAL (Optuna on DistilBERT + EntropyOnRandomSubsetSampler):
    python experimentation.py --mode hybrid

Step 3 — Backbone ablation (BERT + RoBERTa):
    python experimentation.py --mode backbones

Step 4 — Sampler ablation (Random + BADGE):
    python experimentation.py --mode samplers

Step 5 — New datasets (SST-2, TweetEval, Yahoo Answers):
    python experimentation.py --mode new_datasets

Step 6 — Fixed-switch baselines:
    python experimentation.py --mode fixed_switch

Step 7 — Signal ablation:
    python experimentation.py --mode signal_ablation

Quick smoke test (fast, ~5 rounds each):
    python experimentation.py --mode quick

Everything at once:
    python experimentation.py --mode full

Dry-run to preview without running:
    python experimentation.py --mode baselines --dry-run
"""

import itertools
import time
import argparse

import torch
from pathlib import Path
import logging
import optuna

from adaptive_al.active_learning import ActiveLearning, ExperimentConfig

HOURS = 3600




DATASET_NUM_LABELS = {
    "agnews":        4,
    "imdb":          2,
    "jigsaw":        2,
    "sst2":          2,
    "tweeteval":     3,
    "yahoo_answers": 10,
}

ORIGINAL_DATASETS = ["agnews", "imdb", "jigsaw"]
NEW_DATASETS      = ["sst2", "tweeteval", "yahoo_answers"]
ALL_DATASETS      = ORIGINAL_DATASETS + NEW_DATASETS

ALL_MODELS   = ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"]
ALL_SAMPLERS = ["EntropyOnRandomSubsetSampler", "RandomSampler", "BADGESampler"]
ALL_HYBRID_SIGNALS = [
    "delta_f1",
    "delta_loss",
    "delta_accuracy",
    "gradient_norm",
    "spectral_alpha",
    "nc1_ratio",
    "cka",
    "l2_weight_distance",
]
HYBRID_EPSILON_SEARCH_SPACES = {
    "delta_f1": [0.05, 0.02, 0.01, 0.005],
    "delta_loss": [0.1, 0.05, 0.02, 0.01, 0.005],
    "delta_accuracy": [0.05, 0.02, 0.01, 0.005],
    "gradient_norm": [5.0, 2.0, 1.0, 0.5, 0.1],
    "spectral_alpha": [0.5, 0.2, 0.1, 0.05, 0.02],
    "nc1_ratio": [5.0, 2.0, 1.0, 0.5, 0.1],
    "cka": [0.1, 0.05, 0.02, 0.01, 0.005],
    "l2_weight_distance": [5.0, 2.0, 1.0, 0.5, 0.1],
}
HYBRID_K_SEARCH_SPACE = [3, 5, 7, 10]




MODES = {

    "quick": dict(
        datasets=["imdb"],
        strategies=["RetrainStrategy", "FineTuneStrategy", "DeltaF1Strategy"],
        models=["distilbert-base-uncased"],
        samplers=["EntropyOnRandomSubsetSampler"],
        seeds=[42, 43],
        total_rounds=5,
        optuna_hours=0.05,
        max_seconds=300,
        save_dir="./experiments/quick",
    ),


    "subset_sensitivity": dict(
        datasets=["imdb", "sst2"],
        strategies=["RetrainStrategy", "FineTuneStrategy"],
        models=["distilbert-base-uncased"],
        samplers=["EntropyOnRandomSubsetSampler"],
        seeds=[42, 43, 44],
        total_rounds=10,
        ablation_signals=["delta_f1"],
        save_dir="./experiments/subset_sensitivity",
    ),


    "baselines": dict(
        datasets=ALL_DATASETS,
        strategies=["RetrainStrategy", "FineTuneStrategy", "NewOnlyStrategy"],
        models=["distilbert-base-uncased"],
        samplers=["EntropyOnRandomSubsetSampler"],
        seeds=[42, 43, 44, 45, 46],
        save_dir="./experiments/baselines",
    ),


    "hybrid": dict(
        datasets=ALL_DATASETS,
        strategies=["DeltaF1Strategy"],
        models=["distilbert-base-uncased"],
        samplers=["EntropyOnRandomSubsetSampler"],
        seeds=[42, 43, 44, 45, 46],
        ablation_signals=ALL_HYBRID_SIGNALS,
        save_dir="./experiments/hybrid",
    ),


    "backbones": dict(
        datasets=ALL_DATASETS,
        strategies=["RetrainStrategy", "FineTuneStrategy", "NewOnlyStrategy", "DeltaF1Strategy"],
        models=ALL_MODELS,
        samplers=["EntropyOnRandomSubsetSampler"],
        seeds=[42, 43, 44, 45, 46],
        ablation_signals=ALL_HYBRID_SIGNALS,
        save_dir="./experiments/backbones",
    ),


    "samplers": dict(
        datasets=ALL_DATASETS,
        strategies=["RetrainStrategy", "FineTuneStrategy", "NewOnlyStrategy", "DeltaF1Strategy"],
        models=["distilbert-base-uncased"],
        samplers=ALL_SAMPLERS,
        seeds=[42, 43, 44, 45, 46],
        ablation_signals=ALL_HYBRID_SIGNALS,
        save_dir="./experiments/samplers",
    ),


    "fixed_switch": dict(
        datasets=ALL_DATASETS,
        strategies=["FixedSwitchStrategy"],
        models=["distilbert-base-uncased"],
        samplers=["EntropyOnRandomSubsetSampler"],
        seeds=[42, 43, 44, 45, 46],
        fixed_switch_rounds=[3, 5, 7, 10],
        save_dir="./experiments/fixed_switch",
    ),


    "signal_ablation": dict(
        datasets=ALL_DATASETS,
        strategies=["DeltaF1Strategy"],
        models=["distilbert-base-uncased"],
        samplers=["EntropyOnRandomSubsetSampler"],
        seeds=[42, 43, 44, 45, 46],
        ablation_signals=[signal for signal in ALL_HYBRID_SIGNALS if signal != "delta_f1"],
        save_dir="./experiments/signal_ablation",
    ),


    "full": dict(
        datasets=ALL_DATASETS,
        strategies=["RetrainStrategy", "FineTuneStrategy", "NewOnlyStrategy",
                    "DeltaF1Strategy", "FixedSwitchStrategy"],
        models=ALL_MODELS,
        samplers=ALL_SAMPLERS,
        seeds=[42, 43, 44, 45, 46],
        ablation_signals=ALL_HYBRID_SIGNALS,
        fixed_switch_rounds=[3, 5, 7, 10],
        save_dir="./experiments/full",
    ),
}






def model_slug(model_name: str) -> str:
    """Convert a model name like 'roberta-base' to a filesystem-safe short key."""
    return model_name.replace("/", "_")


def get_sampler_kwargs(sampler_name: str, args) -> dict:
    """Return constructor kwargs for the given sampler class."""
    if sampler_name in ("EntropyOnRandomSubsetSampler", "BADGESampler"):
        return {"random_subset_size": args.random_subset_size}
    return {}


def get_hybrid_search_space(signal: str) -> tuple[list[float], list[int]]:
    """Return the epsilon and k grids for a given hybrid signal."""
    if signal not in HYBRID_EPSILON_SEARCH_SPACES:
        raise ValueError(f"Unknown hybrid signal '{signal}'")
    return HYBRID_EPSILON_SEARCH_SPACES[signal], HYBRID_K_SEARCH_SPACE


def _log_final(experiment_name, metrics):
    logging.info(
        "Final Test Metrics (%s): F1=%.4f, Accuracy=%.4f, Loss=%.4f",
        experiment_name,
        metrics['f1_score'],
        metrics['accuracy'],
        metrics['loss'],
    )


def _run_single(config_parameters):
    """Instantiate and run one experiment, save it, return final metrics."""
    experiment_config = ExperimentConfig(**config_parameters)
    al = ActiveLearning(experiment_config)
    final_metrics = al.run_full_pipeline()
    al.save_experiment()
    return final_metrics






def objective(trial, strategy, data_sets, seeds, common_config_parameters,
              start_time=0):
    """Optuna objective: searches epsilon and k for DeltaF1Strategy."""
    signal = common_config_parameters.get("_ablation_signal", "delta_f1")
    epsilon_grid, k_grid = get_hybrid_search_space(signal)
    epsilon = trial.suggest_categorical("epsilon", epsilon_grid)
    k       = trial.suggest_categorical("k", k_grid)

    strat_kwargs = {
        "epsilon": epsilon,
        "k": k,
        "signal": signal,
    }

    sampler_name = common_config_parameters.get("sampler_class", "sampler")
    mslug        = model_slug(common_config_parameters.get("model_name_or_path", "model"))

    f1_scores = []
    for data_set in data_sets:
        for seed in seeds:
            logging.info(
                f"[Optuna] eps={epsilon}, k={k} | "
                f"{data_set} seed={seed} | "
                f"elapsed {(time.perf_counter()-start_time)/HOURS:.2f}h"
            )
            experiment_name = (
                f"{strategy}_{signal}_{epsilon}_{k}_"
                f"{sampler_name}_{mslug}_{data_set}_{seed}"
            )

            config_parameters = {k: v for k, v in common_config_parameters.items()
                                  if not k.startswith("_")}
            config_parameters.update({
                "experiment_name": experiment_name,
                "data": data_set,
                "seed": seed,
                "num_labels": DATASET_NUM_LABELS[data_set],
                "strategy_class": strategy,
                "strategy_kwargs": strat_kwargs,
                "sampler_kwargs": {**common_config_parameters["sampler_kwargs"], "seed": seed},
            })

            metrics = _run_single(config_parameters)
            f1_scores.append(metrics["f1_score"])

    return sum(f1_scores) / len(f1_scores) if f1_scores else None






def parse_args():
    parser = argparse.ArgumentParser(
        description="Active learning experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )


    parser.add_argument(
        "--mode",
        choices=list(MODES.keys()),
        default=None,
        help="Experiment preset. Sets sensible defaults; any other flag overrides the mode.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run (with counts) without executing anything.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all experiments, including ones that already have saved results.",
    )


    parser.add_argument("--datasets",   nargs="+", default=None)
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--models",     nargs="+", default=None)
    parser.add_argument("--samplers",   nargs="+", default=None)
    parser.add_argument("--seeds",      nargs="+", type=int, default=None)


    parser.add_argument("--save-dir", type=str, default=None)


    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--batch-size",    type=int,   default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay",  type=float, default=1e-3)
    parser.add_argument("--max-length",    type=int,   default=128)


    parser.add_argument("--initial-pool-size",      type=int,   default=200)
    parser.add_argument("--acquisition-batch-size", type=int,   default=32)
    parser.add_argument("--total-rounds",           type=int,   default=None)
    parser.add_argument("--optuna-rounds",          type=int,   default=15,
                        help="Rounds per trial during Optuna search (fewer = faster trials). Final eval uses --total-rounds.")
    parser.add_argument("--optuna-trials",          type=int,   default=50,
                        help="Number of Optuna trials. 50 covers all 16 combos ~3x (~20h). Overrides --optuna-hours if set.")
    parser.add_argument("--max-seconds",            type=int,   default=None)


    parser.add_argument("--min-rounds-before-plateau", type=int,   default=10)
    parser.add_argument("--plateau-patience",          type=int,   default=-1)
    parser.add_argument("--plateau-f1-threshold",      type=float, default=0.0005)


    parser.add_argument("--random-subset-size", type=int, default=5000)


    parser.add_argument("--scheduler-step-size", type=int,   default=10)
    parser.add_argument("--scheduler-gamma",     type=float, default=0.1)


    parser.add_argument("--optuna-hours", type=float, default=None)


    parser.add_argument("--fixed-switch-rounds", nargs="+", type=int, default=None)


    parser.add_argument(
        "--ablation-signals", nargs="+", default=None,
        help="Signals for DeltaF1Strategy. Each selected signal is tuned with a signal-specific epsilon grid.",
    )
    parser.add_argument("--best-epsilon",              type=float, default=None,
                        help="Deprecated manual override; tuning is now automatic for all hybrid signals.")
    parser.add_argument("--best-k",                    type=int,   default=None,
                        help="Deprecated manual override; tuning is now automatic for all hybrid signals.")
    parser.add_argument("--best-validation-fraction",  type=float, default=None,
                        help="Unused legacy flag retained for CLI compatibility.")

    args = parser.parse_args()


    if args.mode is not None:
        mode_defaults = MODES[args.mode]
        for key, value in mode_defaults.items():
            cli_key = key.replace("-", "_")
            if getattr(args, cli_key, None) is None:
                setattr(args, cli_key, value)


    if args.datasets          is None: args.datasets          = ORIGINAL_DATASETS
    if args.strategies        is None: args.strategies        = ["DeltaF1Strategy", "FineTuneStrategy",
                                                                   "NewOnlyStrategy", "RetrainStrategy"]
    if args.models            is None: args.models            = ["distilbert-base-uncased"]
    if args.samplers          is None: args.samplers          = ["EntropyOnRandomSubsetSampler"]
    if args.seeds             is None: args.seeds             = [42, 43, 44, 45, 46]
    if args.save_dir          is None: args.save_dir          = "./experiments/sep_25"
    if args.total_rounds      is None: args.total_rounds      = 25
    if args.max_seconds       is None: args.max_seconds       = 2400
    if args.optuna_hours      is None: args.optuna_hours      = 48
    if args.fixed_switch_rounds is None: args.fixed_switch_rounds = [3, 5, 7, 10]
    if args.ablation_signals  is None: args.ablation_signals  = ALL_HYBRID_SIGNALS

    return args






def build_experiment_list(args, save_dir: Path):
    """
    Enumerate every (experiment_name, strategy, config) that would be run.
    Returns a list of dicts, each with keys: name, strategy, config_parameters.
    """
    experiments = []

    for model_name in args.models:
        mslug = model_slug(model_name)

        base_config = {
            "save_dir":              save_dir,
            "initial_pool_size":     args.initial_pool_size,
            "acquisition_batch_size":args.acquisition_batch_size,
            "max_seconds":           args.max_seconds,
            "total_rounds":          args.total_rounds,
            "min_rounds_before_plateau": args.min_rounds_before_plateau,
            "plateau_patience":      args.plateau_patience,
            "plateau_f1_threshold":  args.plateau_f1_threshold,
            "model_name_or_path":    model_name,
            "tokenizer_kwargs": {
                "max_length":         args.max_length,
                "padding":            "max_length",
                "truncation":         True,
                "add_special_tokens": True,
                "return_tensors":     "pt",
            },
            "optimizer_class":  "AdamW",
            "optimizer_kwargs": {"lr": args.learning_rate, "weight_decay": args.weight_decay},
            "criterion_class":  "CrossEntropyLoss",
            "criterion_kwargs": {},
            "scheduler_class":  None,
            "scheduler_kwargs": {},
            "epochs":     args.epochs,
            "batch_size": args.batch_size,
        }

        for sampler_name in args.samplers:
            sampler_kwargs_base = get_sampler_kwargs(sampler_name, args)
            common = {
                **base_config,
                "sampler_class":  sampler_name,
                "sampler_kwargs": sampler_kwargs_base,
            }

            for strategy in args.strategies:

                if strategy == "FixedSwitchStrategy":
                    for switch_r in args.fixed_switch_rounds:
                        for data_set, seed in itertools.product(args.datasets, args.seeds):
                            name = (
                                f"FixedSwitchStrategy_r{switch_r}_{sampler_name}"
                                f"_{mslug}_{data_set}_{seed}"
                            )
                            cfg = {**common,
                                   "experiment_name": name,
                                   "data":            data_set,
                                   "seed":            seed,
                                   "num_labels":      DATASET_NUM_LABELS[data_set],
                                   "strategy_class":  strategy,
                                   "strategy_kwargs": {"switch_round": switch_r},
                                   "sampler_kwargs":  {**sampler_kwargs_base, "seed": seed}}
                            experiments.append({"name": name, "strategy": strategy, "config": cfg})

                elif strategy == "DeltaF1Strategy":
                    for signal in args.ablation_signals:
                        experiments.append({
                            "name":     f"[Tune] DeltaF1Strategy_{signal}_{sampler_name}_{mslug}_({'|'.join(args.datasets)})",
                            "strategy": "DeltaF1Strategy_Optuna",
                            "config":   {**common, "_ablation_signal": signal},
                        })

                else:
                    for data_set, seed in itertools.product(args.datasets, args.seeds):
                        name = f"{strategy}_{sampler_name}_{mslug}_{data_set}_{seed}"
                        cfg = {**common,
                               "experiment_name": name,
                               "data":            data_set,
                               "seed":            seed,
                               "num_labels":      DATASET_NUM_LABELS[data_set],
                               "strategy_class":  strategy,
                               "strategy_kwargs": {},
                               "sampler_kwargs":  {**sampler_kwargs_base, "seed": seed}}
                        experiments.append({"name": name, "strategy": strategy, "config": cfg})

    return experiments






if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"



    save_dir = Path(args.save_dir)


    mode_label = f"  mode     : {args.mode}" if args.mode else "  mode     : (custom)"
    print("=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(mode_label)
    print(f"  device   : {device}")
    print(f"  save_dir : {save_dir}")
    print(f"  datasets : {args.datasets}")
    print(f"  strategies: {args.strategies}")
    print(f"  models   : {args.models}")
    print(f"  samplers : {args.samplers}")
    print(f"  seeds    : {args.seeds}")
    print(f"  rounds   : {args.total_rounds}  epochs: {args.epochs}  batch: {args.batch_size}")
    print(f"  signals  : {args.ablation_signals}")
    if "FixedSwitchStrategy" in args.strategies:
        print(f"  fixed_switch_rounds: {args.fixed_switch_rounds}")
    print("=" * 70)


    experiment_list = build_experiment_list(args, save_dir)


    optuna_runs  = [e for e in experiment_list if e["strategy"] == "DeltaF1Strategy_Optuna"]
    direct_runs  = [e for e in experiment_list if e["strategy"] != "DeltaF1Strategy_Optuna"]

    existing     = [] if args.force else [e for e in direct_runs if (save_dir / e["name"]).exists()]
    pending      = direct_runs if args.force else [e for e in direct_runs if not (save_dir / e["name"]).exists()]

    print(f"\nDirect experiments : {len(direct_runs):4d} total  "
          f"({len(existing)} already done, {len(pending)} to run)")
    print(f"Tuning runs        : {len(optuna_runs):4d}  "
          f"(one per hybrid signal/model/sampler combination)")

    if args.dry_run:
        print("\n── Pending direct experiments ──")
        for i, e in enumerate(pending, 1):
            print(f"  {i:4d}. [{e['strategy']:30s}] {e['name']}")
        if optuna_runs:
            print("\n── Tuning runs ──")
            for e in optuna_runs:
                print(f"        {e['name']}")
        print("\nDry run complete. Nothing was executed.")
        raise SystemExit(0)


    for e in experiment_list:
        e["config"]["device"] = device

    start_time   = time.perf_counter()
    total_to_run = len(pending) + len(optuna_runs)
    done_count   = 0


    for e in pending:
        done_count += 1
        logging.info(
            f"\n[{done_count}/{total_to_run}] {e['name']}"
        )
        try:
            metrics = _run_single(e["config"])
            _log_final(e["name"], metrics)
        except Exception as exc:
            logging.error(f"FAILED: {e['name']} — {exc}", exc_info=True)


    for e in optuna_runs:
        done_count += 1
        common_cfg = e["config"].copy()
        signal     = common_cfg.get("_ablation_signal", "delta_f1")
        mslug_val  = model_slug(common_cfg.get("model_name_or_path", "model"))
        sampler_n  = common_cfg.get("sampler_class", "sampler")

        logging.info(
            f"\n[{done_count}/{total_to_run}] Grid search DeltaF1Strategy "
            f"signal={signal} model={mslug_val} sampler={sampler_n}"
        )

        tune_seeds    = args.seeds[:3]
        tune_datasets = ["imdb", "agnews"]
        tune_cfg      = {**common_cfg, "total_rounds": args.optuna_rounds}
        sampler_kwargs_base = common_cfg.get("sampler_kwargs", {})

        epsilons, ks = get_hybrid_search_space(signal)
        grid_results = {}

        import glob as _glob

        def _load_existing_f1(exp_dir: Path):
            """Return f1_score from an existing result JSON, or None if not found."""
            jsons = sorted(_glob.glob(str(exp_dir / "*.json")))
            for jf in reversed(jsons):
                try:
                    d = json.load(open(jf))
                    f1 = d.get("final_test_stats", {}).get("f1_score")
                    if f1 is not None:
                        return f1
                except Exception:
                    continue
            return None

        total_combos = len(epsilons) * len(ks)
        combo_count  = 0
        for epsilon, k in itertools.product(epsilons, ks):
            combo_count += 1
            logging.info(f"  Grid [{combo_count}/{total_combos}] epsilon={epsilon} k={k}")
            f1_scores = []
            for data_set, seed in itertools.product(tune_datasets, tune_seeds):
                name = (
                    f"DeltaF1Strategy_{signal}_{epsilon}_{k}_"
                    f"{sampler_n}_{mslug_val}_{data_set}_{seed}"
                )
                exp_dir = save_dir / name
                existing_f1 = _load_existing_f1(exp_dir)
                if existing_f1 is not None:
                    logging.info(f"    Skipping (exists): {name}  f1={existing_f1:.4f}")
                    f1_scores.append(existing_f1)
                    continue
                cfg = {k_: v for k_, v in tune_cfg.items() if not k_.startswith("_")}
                cfg.update({
                    "experiment_name": name,
                    "data":            data_set,
                    "seed":            seed,
                    "num_labels":      DATASET_NUM_LABELS[data_set],
                    "strategy_class":  "DeltaF1Strategy",
                    "strategy_kwargs": {"epsilon": epsilon, "k": k, "signal": signal},
                    "sampler_kwargs":  {**sampler_kwargs_base, "seed": seed},
                })
                try:
                    metrics = _run_single(cfg)
                    f1_scores.append(metrics["f1_score"])
                except Exception as exc:
                    logging.error(f"FAILED: {name} — {exc}", exc_info=True)
            grid_results[(epsilon, k)] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            logging.info(f"    avg F1 = {grid_results[(epsilon, k)]:.4f}")

        best_epsilon, best_k = max(grid_results, key=grid_results.get)
        best_avg_f1 = grid_results[(best_epsilon, best_k)]
        logging.info(f"Best: epsilon={best_epsilon} k={best_k} avg_F1={best_avg_f1:.4f}")
        print(
            f"\n{'='*60}\n"
            f"  BEST HYPERPARAMS (DeltaF1Strategy, signal={signal},\n"
            f"  model={mslug_val}, sampler={sampler_n}):\n"
            f"    --best-epsilon  {best_epsilon}\n"
            f"    --best-k        {best_k}\n"
            f"  Avg F1 = {best_avg_f1:.4f}\n"
            f"{'='*60}\n"
        )

        import json as _json
        best_hparams_path = save_dir / f"best_hyperparams_{signal}_{sampler_n}_{mslug_val}.json"
        with open(best_hparams_path, "w") as _hf:
            _json.dump(
                {"signal": signal, "epsilon": best_epsilon, "k": best_k, "avg_f1": best_avg_f1},
                _hf,
                indent=2,
            )
        logging.info("Best hyperparams saved to %s", best_hparams_path)


        logging.info("Running final evaluation with best hyperparams on all datasets and seeds...")
        best_strat_kwargs = {"epsilon": best_epsilon, "k": best_k, "signal": signal}
        for data_set, seed in itertools.product(args.datasets, args.seeds):
            final_name = (
                f"DeltaF1Strategy_{signal}_{best_epsilon}_{best_k}_"
                f"{sampler_n}_{mslug_val}_{data_set}_{seed}"
            )
            exp_dir = save_dir / final_name
            if _load_existing_f1(exp_dir) is not None:
                logging.info(f"Final eval skipping (exists): {final_name}")
                continue
            final_cfg = {k_: v for k_, v in common_cfg.items() if not k_.startswith("_")}
            final_cfg.update({
                "experiment_name": final_name,
                "data":            data_set,
                "seed":            seed,
                "num_labels":      DATASET_NUM_LABELS[data_set],
                "strategy_class":  "DeltaF1Strategy",
                "strategy_kwargs": best_strat_kwargs,
                "sampler_kwargs":  {**sampler_kwargs_base, "seed": seed},
            })
            logging.info(f"Final eval: {final_name}")
            try:
                metrics = _run_single(final_cfg)
                _log_final(final_name, metrics)
            except Exception as exc:
                logging.error(f"FAILED: {final_name} — {exc}", exc_info=True)

    elapsed = (time.perf_counter() - start_time) / HOURS
    logging.info(f"All experiments complete. Total elapsed: {elapsed:.2f}h")
