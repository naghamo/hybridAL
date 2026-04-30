"""
Full retraining strategy for active learning.

This module implements a training strategy that resets the model to initial
weights and retrains from scratch on all labeled data each round. This
approach ensures the model learns from all data consistently but is
computationally expensive.
"""

import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple

from .base_strategy import BaseStrategy
from ..pool import DataPool


class RetrainStrategy(BaseStrategy):
    """
    Full retraining strategy that resets model to initial weights each round.

    This strategy resets the model to its initial state and trains from scratch
    on all accumulated labeled data each round.
    """
    def __init__(self, **kwargs):
        """
        Initialize the full retraining strategy.

        Args:
            **kwargs: Arguments passed to BaseStrategy (model, optimizer, etc.).
        """
        super().__init__(**kwargs)

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        """
        Reset model to initial weights and retrain on all labeled data.

        Adds new indices to the labeled pool, resets the model to its initial
        state, and trains from scratch for the configured number of epochs on
        all accumulated labeled samples.

        Args:
            pool (DataPool): Current data pool with labeled/unlabeled splits.
            new_indices (List[int]): Newly sampled indices to add to training pool.

        Returns:
            Dict: Training statistics including avg_loss, epochs, total_samples,
                  and new_samples.
        """

        if new_indices:
            pool.add_labeled_samples(new_indices)
        labeled_subset = pool.get_labeled_subset()
        dataloader = DataLoader(labeled_subset, batch_size=self.batch_size, shuffle=True)

        self.reset()

        total_loss, num_batches, actual_epochs = self.train_epochs(dataloader)

        return self.get_stats(total_loss, num_batches, labeled_subset, new_indices,
                              actual_epochs=actual_epochs)
