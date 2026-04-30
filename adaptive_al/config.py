"""
Configuration dataclass for active learning experiments.

This module defines ExperimentConfig, which contains all parameters needed
to configure and run an active learning experiment, including model settings,
training hyperparameters, sampling strategies, and stopping criteria.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     pass


@dataclass
class ExperimentConfig:
    """
    Configuration for active learning experiments.

    This dataclass encapsulates all settings required for running an active
    learning pipeline, including reproducibility seeds, pool management,
    training parameters, model configuration, and logging options.
    """
    # --- Reproducibility
    seed: int = 42
    total_rounds: int = -1 # Total rounds to run active learning (model training + sampling new data),
                           # -1 means until we run out of data

    # --- Pool config
    initial_pool_size: int = 200 # Initial size of the pool for training
    acquisition_batch_size: int = 32 # New labels per round

    # Plateau checking:
    min_rounds_before_plateau: int = -1
    plateau_patience: int = -1
    plateau_f1_threshold: float = 0.5

    max_seconds: int = None # i.e. no timeout

    pool_proportion_threshold: float = -1

    sampler_class: str = field(default=None)
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)

    # --- Training config
    # Strategy and Sampler configuration
    strategy_class: str = field(default=None)
    strategy_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Model configuration (via transformers)
    model_name_or_path: str = field(default='None')
    num_labels: int = field(default=None)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "max_length": 128,
        "padding": "max_length",
        "truncation": True,
        "add_special_tokens": True,
        "return_tensors": "pt"
    })

    # Optimizer / Criterion / Scheduler configuration
    optimizer_class: str = field(default=None)
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    criterion_class: str = field(default=None)
    criterion_kwargs: Dict[str, Any] = field(default_factory=dict)

    scheduler_class: str = field(default=None)
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)

    device: str = 'cuda'
    epochs: int = 10
    batch_size: int = 16

    # Early stopping (applied uniformly to all strategies; controls within-round epoch loop)
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 1e-4

    # HybridAL switching signal. Selects which of the per-round signals drives
    # the Retrain → FineTune switch in DeltaF1Strategy. All signals are still
    # logged every round regardless of which one is active.
    # Options: "delta_f1", "delta_accuracy", "delta_loss",
    #          "gradient_norm", "l2_weight_distance", "cka".
    switching_signal: str = "delta_f1"

    # Dataset for the model
    data: str = field(default=None)

    # Logging
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

        # Ensure kwargs are always dicts
        self.strategy_kwargs = self.strategy_kwargs or {}
        self.sampler_kwargs = self.sampler_kwargs or {}
        self.optimizer_kwargs = self.optimizer_kwargs or {}
        self.criterion_kwargs = self.criterion_kwargs or {}
        self.scheduler_kwargs = self.scheduler_kwargs or {}


