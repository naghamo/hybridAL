"""Constants shared across all Phase-2 experiment runners.

Same conventions as the in-flight signal ablation so results stay comparable.
"""
from typing import Dict, List


DATASET_NUM_LABELS: Dict[str, int] = {
    "imdb":          2,
    "agnews":        4,
    "jigsaw":        2,
    "sst2":          2,
    "tweeteval":     3,
    "yahoo_answers": 10,
}
ALL_DATASETS: List[str] = list(DATASET_NUM_LABELS.keys())


SEEDS_FULL:    List[int] = [42, 43, 44, 45, 46]
SEEDS_TUNING:  List[int] = [42, 43, 44]


BACKBONES_ALL: List[str] = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
]
BACKBONE_DEFAULT: str = "distilbert-base-uncased"





GPU_DATASETS_DEFAULT: Dict[int, List[str]] = {
    0: ["imdb", "agnews", "jigsaw"],
    1: ["sst2", "tweeteval", "yahoo_answers"],
}


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



K_GRID_DEFAULT: List[int] = [2, 3, 5, 7]


SIGNAL_ABLATION_EPSILON: float = 0.5
SIGNAL_ABLATION_K:       int   = 3


CALIBRATION_NORMALIZERS_PATH = "_calibration_normalizers.json"
SIGNAL_ABLATION_RESULTS_DIR  = "experiments/signal_ablation"




DATASETS_TUNING_2: List[str] = ["imdb", "agnews"]









DATASETS_3:        List[str] = ["imdb", "agnews", "yahoo_answers"]



DATASETS_CALIB_4:  List[str] = ["imdb", "agnews", "jigsaw", "sst2"]





N_GRID_DEFAULT: List[int] = [100, 500, 1000, 5000]


POOL_SIZE_GRID: List[int] = [50, 100, 200, 500]


BATCH_SIZE_GRID: List[int] = [16, 32, 64, 128]
