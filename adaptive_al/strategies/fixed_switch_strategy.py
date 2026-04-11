"""
Fixed-switch baseline strategy for active learning.

This module implements a training strategy that switches from full retraining
to fine-tuning at a predetermined round number. Used as a baseline to determine
whether the ΔF1-based adaptive switch is better than an arbitrary fixed switch point.
"""

import logging
from typing import List, Dict

from .base_strategy import BaseStrategy
from .fine_tuning_strategy import FineTuneStrategy
from .retrain_strategy import RetrainStrategy
from ..pool import DataPool


class FixedSwitchStrategy(BaseStrategy):
    """
    Baseline strategy that switches from retraining to fine-tuning at a fixed round.

    Unlike DeltaF1Strategy (which adapts when to switch based on validation F1),
    this strategy switches unconditionally at a hardcoded round number. Used to
    show that the adaptive ΔF1 switching signal outperforms arbitrary switching.

    Attributes:
        switch_round (int): The round at which to switch from retraining to fine-tuning.
        switched (bool): Whether the switch has occurred.
    """

    def __init__(self, switch_round: int, **kwargs):
        """
        Initialize the fixed-switch strategy.

        Args:
            switch_round (int): Round number at which to switch to fine-tuning
                               (1-indexed; switch happens when _internal_round >= switch_round).
            **kwargs: Additional arguments passed to BaseStrategy.
        """
        super().__init__(**kwargs)
        self.switch_round = switch_round
        self._internal_round = 0
        self.switched = False

        self.fine_tune = FineTuneStrategy(strategy=self)
        self.retrain = RetrainStrategy(strategy=self)

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        """
        Train using retraining until switch_round, then fine-tuning thereafter.

        Args:
            pool (DataPool): Current data pool with labeled/unlabeled splits.
            new_indices (List[int]): Newly sampled indices (not yet in pool).

        Returns:
            Dict: Training statistics from the underlying strategy.
        """
        self._internal_round += 1

        if not self.switched and self._internal_round >= self.switch_round:
            self.switched = True
            logging.info(
                f"FixedSwitchStrategy: switching to fine-tuning at round {self._internal_round} "
                f"(configured switch_round={self.switch_round})"
            )

        if self.switched:
            return self.fine_tune._train_implementation(pool, new_indices)
        return self.retrain._train_implementation(pool, new_indices)
