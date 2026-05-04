"""build_cfg(...) — produce a fully-populated ExperimentConfig with the
defaults shared across all Phase-2 experiments. Each runner only overrides
the axes that make its experiment different.
"""
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from adaptive_al.config import ExperimentConfig

from . import constants as C


def build_cfg(
    *,
    experiment_name: str,
    save_dir: Path,
    data: str,
    seed: int,
    strategy_class: str,
    strategy_kwargs: Optional[Dict[str, Any]] = None,
    sampler_class: str = C.DEFAULTS["sampler_class"],
    sampler_kwargs: Optional[Dict[str, Any]] = None,
    model_name_or_path: str = C.BACKBONE_DEFAULT,
    total_rounds: int = C.DEFAULTS["total_rounds"],
    initial_pool_size: int = C.DEFAULTS["initial_pool_size"],
    acquisition_batch_size: int = C.DEFAULTS["acquisition_batch_size"],
    epochs: int = C.DEFAULTS["epochs"],
    batch_size: int = C.DEFAULTS["batch_size"],
    early_stopping_patience: int = C.DEFAULTS["early_stopping_patience"],
    early_stopping_min_delta: float = C.DEFAULTS["early_stopping_min_delta"],
    log_all_signals: bool = C.DEFAULTS["log_all_signals"],
    random_subset_size: int = C.DEFAULTS["random_subset_size"],
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    """Return an ExperimentConfig with all defaults filled in.

    Notes:
      - sampler_kwargs defaults to {"random_subset_size": <random_subset_size>, "seed": <seed>}
        for the EntropyOnRandomSubsetSampler / BADGESampler. Random sampler should
        override sampler_kwargs to {"seed": <seed>} only.
      - strategy_kwargs is passed through unchanged; the runner picks (epsilon, k,
        signal, signal_normalizer) etc. as appropriate for each experiment.
      - `extra` is a dict of any additional ExperimentConfig fields to override
        (e.g., switching_signal). Applied last.
    """
    if strategy_kwargs is None:
        strategy_kwargs = {}
    if sampler_kwargs is None:
        sampler_kwargs = {"random_subset_size": random_subset_size, "seed": seed}
    if optimizer_kwargs is None:
        optimizer_kwargs = deepcopy(C.DEFAULTS["optimizer_kwargs"])

    if data not in C.DATASET_NUM_LABELS:
        raise ValueError(f"Unknown dataset {data!r}; expected one of {C.ALL_DATASETS}")
    num_labels = C.DATASET_NUM_LABELS[data]

    cfg_kwargs = dict(
        seed=seed,
        total_rounds=total_rounds,
        initial_pool_size=initial_pool_size,
        acquisition_batch_size=acquisition_batch_size,
        sampler_class=sampler_class,
        sampler_kwargs=sampler_kwargs,
        strategy_class=strategy_class,
        strategy_kwargs=strategy_kwargs,
        model_name_or_path=model_name_or_path,
        num_labels=num_labels,
        tokenizer_kwargs=deepcopy(C.TOKENIZER_KWARGS_DEFAULT),
        optimizer_class=C.DEFAULTS["optimizer_class"],
        optimizer_kwargs=optimizer_kwargs,
        criterion_class=C.DEFAULTS["criterion_class"],
        criterion_kwargs=deepcopy(C.DEFAULTS["criterion_kwargs"]),
        scheduler_class=C.DEFAULTS["scheduler_class"],
        scheduler_kwargs=deepcopy(C.DEFAULTS["scheduler_kwargs"]),
        device=C.DEFAULTS["device"],
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        log_all_signals=log_all_signals,
        data=data,
        save_dir=save_dir,
        experiment_name=experiment_name,
    )
    if extra:
        cfg_kwargs.update(extra)

    return ExperimentConfig(**cfg_kwargs)
