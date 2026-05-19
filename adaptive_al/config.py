"""
Configuration dataclass for active learning experiments.

This module defines ExperimentConfig, which contains all parameters needed
to configure and run an active learning experiment, including model settings,
training hyperparameters, sampling strategies, and stopping criteria.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set






@dataclass
class ExperimentConfig:
    """
    Configuration for active learning experiments.

    This dataclass encapsulates all settings required for running an active
    learning pipeline, including reproducibility seeds, pool management,
    training parameters, model configuration, and logging options.
    """

    seed: int = 42
    total_rounds: int = -1



    initial_pool_size: int = 200
    acquisition_batch_size: int = 32


    min_rounds_before_plateau: int = -1
    plateau_patience: int = -1
    plateau_f1_threshold: float = 0.5

    max_seconds: int = None

    pool_proportion_threshold: float = -1

    sampler_class: str = field(default=None)
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)



    strategy_class: str = field(default=None)
    strategy_kwargs: Dict[str, Any] = field(default_factory=dict)


    model_name_or_path: str = field(default='None')
    num_labels: int = field(default=None)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "max_length": 128,
        "padding": "max_length",
        "truncation": True,
        "add_special_tokens": True,
        "return_tensors": "pt"
    })


    optimizer_class: str = field(default=None)
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    criterion_class: str = field(default=None)
    criterion_kwargs: Dict[str, Any] = field(default_factory=dict)

    scheduler_class: str = field(default=None)
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)

    device: str = 'cuda'
    epochs: int = 10
    batch_size: int = 16


    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 1e-4






    switching_signal: str = "delta_f1"




    log_all_signals: bool = False


    data: str = field(default=None)


    save_dir: Path = Path("./results")
    experiment_name: str = "dummy"

    def __post_init__(self):
        """
        Validate required configuration fields after initialization.

        Ensures that all required fields are set and converts None kwargs
        to empty dictionaries for consistent handling throughout the pipeline.

        Raises:
            ValueError: If any required field (model_name_or_path, num_labels,
                       strategy_class, sampler_class, optimizer_class, or
                       criterion_class) is None.
        """
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path is required in ExperimentConfig")
        if self.num_labels is None:
            raise ValueError("num_labels is required in ExperimentConfig")
        if self.strategy_class is None:
            raise ValueError("strategy_class is required in ExperimentConfig")
        if self.sampler_class is None:
            raise ValueError("sampler_class is required in ExperimentConfig")
        if self.optimizer_class is None:
            raise ValueError("optimizer_class is required in ExperimentConfig")
        if self.criterion_class is None:
            raise ValueError("criterion_class is required in ExperimentConfig")


        self.strategy_kwargs = self.strategy_kwargs or {}
        self.sampler_kwargs = self.sampler_kwargs or {}
        self.optimizer_kwargs = self.optimizer_kwargs or {}
        self.criterion_kwargs = self.criterion_kwargs or {}
        self.scheduler_kwargs = self.scheduler_kwargs or {}


