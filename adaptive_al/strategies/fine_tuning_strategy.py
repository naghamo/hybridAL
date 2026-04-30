"""
Fine-tuning training strategy for active learning.

This module implements an incremental training strategy that fine-tunes
the model on all labeled data accumulated so far, without resetting to
initial weights. This approach is efficient for later rounds when the
model has already learned useful representations.
"""

import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple

from .base_strategy import BaseStrategy
from ..pool import DataPool


class FineTuneStrategy(BaseStrategy):
    """
    Incremental fine-tuning strategy that continues training from current weights.

    This strategy trains the model on all labeled data without resetting to
    initial weights, allowing the model to incrementally adapt to new samples.
    """
    def __init__(self, **kwargs):
        """
        Initialize the fine-tuning strategy.

        Args:
            **kwargs: Arguments passed to BaseStrategy (model, optimizer, etc.).
        """
        super().__init__(**kwargs)

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        """
        Fine-tune the model on all labeled data without resetting weights.

        Adds new indices to the labeled pool and trains for the configured
        number of epochs on all accumulated labeled samples. The model continues
        from its current state rather than resetting to initial weights.

        Args:
            pool (DataPool): Current data pool with labeled/unlabeled splits.
            new_indices (List[int]): Newly sampled indices to add to training pool.

        Returns:
            Dict: Training statistics including avg_loss, epochs, total_samples,
                  and new_samples.
        """
        self.model.train()  # Set the model to training mode
        self.model.to(self.device)

        if new_indices:
            pool.add_labeled_samples(new_indices)

        # Get all labeled data from the pool
        labeled_subset = pool.get_labeled_subset()
        dataloader = DataLoader(labeled_subset, batch_size=self.batch_size, shuffle=True)
        total_loss, num_batches, actual_epochs = self.train_epochs(dataloader)
        return self.get_stats(total_loss, num_batches, labeled_subset, new_indices,
                              actual_epochs=actual_epochs)
