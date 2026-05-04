"""Constants shared across all Phase-2 experiment runners.

Same conventions as the in-flight signal ablation so results stay comparable.
"""
from typing import Dict, List

# ----------------------------------------------------------------- Datasets --
DATASET_NUM_LABELS: Dict[str, int] = {
    "imdb":          2,
    "agnews":        4,
    "jigsaw":        2,
    "sst2":          2,
    "tweeteval":     3,
    "yahoo_answers": 10,
}
ALL_DATASETS: List[str] = list(DATASET_NUM_LABELS.keys())

# ----------------------------------------------------------------- Seeds ----
SEEDS_FULL:    List[int] = [42, 43, 44, 45, 46]   # 5-seed evaluation (Exp 2, 3)
SEEDS_TUNING:  List[int] = [42, 43, 44]           # 3-seed tuning / appendix (Exp 1, 4-8)

# --------------------------------------------------------------- Backbones --
BACKBONES_ALL: List[str] = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
]
BACKBONE_DEFAULT: str = "distilbert-base-uncased"

# ------------------------------------------------------------ GPU dataset split
# Default split mirrors the signal ablation; can be overridden via
# experiments_lib.shared.runner.split_for_gpus(plan, gpu_id, override=...)
# or rebuilt by training-time balance via --rebalance-from-csv.
GPU_DATASETS_DEFAULT: Dict[int, List[str]] = {
    0: ["imdb", "agnews", "jigsaw"],
    1: ["sst2", "tweeteval", "yahoo_answers"],
}

# ------------------------------------------------------------- Run defaults --
DEFAULTS = dict(
    initial_pool_size       = 200,
    acquisition_batch_size  = 32,
    total_rounds            = 25,
    epochs                  = 10,
    batch_size              = 16,
    early_stopping_patience = 2,
    early_stopping_min_delta= 1e-4,
    log_all_signals         = True,
    random_subset_size      = 1000,
    sampler_class           = "EntropyOnRandomSubsetSampler",
    optimizer_class         = "AdamW",
    optimizer_kwargs        = {"lr": 2e-5, "weight_decay": 1e-3},
    criterion_class         = "CrossEntropyLoss",
    criterion_kwargs        = {},
    scheduler_class         = None,
    scheduler_kwargs        = {},
    device                  = "cuda",
)

TOKENIZER_KWARGS_DEFAULT = {
    "max_length":         128,
    "padding":             "max_length",
    "truncation":          True,
    "add_special_tokens":  True,
    "return_tensors":      "pt",
}

# Phase-2 hyperparameter tuning uses raw ε at each signal's natural scale
# (no calibration normalizer); the k grid is fixed.
K_GRID_DEFAULT: List[int] = [2, 3, 5, 7]

# Pareto / comparison plot ε used by the in-flight signal ablation:
SIGNAL_ABLATION_EPSILON: float = 0.5
SIGNAL_ABLATION_K:       int   = 3

# --------------------------------------------------------- Calibration paths
CALIBRATION_NORMALIZERS_PATH = "_calibration_normalizers.json"
SIGNAL_ABLATION_RESULTS_DIR  = "experiments/signal_ablation"

# ---------------------------------------------- Per-experiment dataset picks
# Tuning / appendix experiments that use 2 datasets — IMDb (binary) + AG News
# (4-class) for breadth without inflating run count.
DATASETS_TUNING_2: List[str] = ["imdb", "agnews"]

# Exp 5 + Exp 6 use 3 datasets covering the full label-cardinality range:
#   IMDb           (2 classes, balanced)
#   AG News        (4 classes, balanced)
#   Yahoo Answers (10 classes, balanced — extreme case where |L_0|=50
#                  gives only 5 examples per class, stress-testing the
#                  initial-pool ablation in Exp 5).
# SST-2 was previously here but offered no diversity over IMDb (both
# binary, both sentiment) so it was swapped for Yahoo.
DATASETS_3:        List[str] = ["imdb", "agnews", "yahoo_answers"]

# Exp 9 (calibration sensitivity) — 2 datasets seen in calibration sources +
# 2 unseen, to test generalization of the normalizer constants.
DATASETS_CALIB_4:  List[str] = ["imdb", "agnews", "jigsaw", "sst2"]

# ---------------------------------------------- N (random_subset_size) grid
# Exp 0 — N sensitivity (Retrain + Entropy). 100 = small subset (entropy
# samples from a tiny pool, more random-like), 1000 = current default,
# 5000 = much larger subset (closer to full-pool entropy).
N_GRID_DEFAULT: List[int] = [100, 500, 1000, 5000]

# ---------------------------------------------- Initial pool size grid
POOL_SIZE_GRID: List[int] = [50, 100, 200, 500]

# ---------------------------------------------- Acquisition batch size grid
BATCH_SIZE_GRID: List[int] = [16, 32, 64, 128]
