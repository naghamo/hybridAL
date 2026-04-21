"""
New-only training strategy for active learning.

This module implements a training strategy that trains the model exclusively
on newly sampled data each round, without incorporating previously labeled
samples. This approach is highly efficient but may lead to catastrophic
forgetting of earlier knowledge.
"""

import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple

from .base_strategy import BaseStrategy
from ..pool import DataPool


class NewOnlyStrategy(BaseStrategy):
    """
    Training strategy that uses only newly acquired samples each round.

    This strategy trains the model exclusively on the newly sampled data from
    each round, ignoring all previously labeled samples.
    """
    def __init__(self, **kwargs):
        """
        Initialize the new-only training strategy.

        Args:
            **kwargs: Arguments passed to BaseStrategy (model, optimizer, etc.).
        """
        super().__init__(**kwargs)

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        """
        Train the model exclusively on newly sampled data.

        Adds new indices to the labeled pool, then trains only on these new
        samples for the configured number of epochs. All previously labeled
        data is ignored in this training round.

        If no new indices are provided, returns zero statistics without training.
        Uses batch size equal to the number of new samples.

        Args:
            pool (DataPool): Current data pool with labeled/unlabeled splits.
            new_indices (List[int]): Newly sampled indices to train on.

        Returns:
            Dict: Training statistics including avg_loss, epochs, total_samples,
                  and new_samples.
        """
        if not new_indices:
            # Initial round: train on the full labeled pool (same as other strategies)
            labeled_subset = pool.get_labeled_subset()
            dataloader = DataLoader(labeled_subset, batch_size=self.batch_size, shuffle=True)
            total_loss, num_batches = self.train_epochs(dataloader)
            return self.get_stats(total_loss, num_batches, labeled_subset, [])

        pool.add_labeled_samples(new_indices)

        labeled_subset = pool.get_subset(new_indices)
        dataloader = DataLoader(labeled_subset, batch_size=self.batch_size, shuffle=True)


        total_loss, num_batches = self.train_epochs(dataloader)

        return self.get_stats(total_loss, num_batches, labeled_subset, new_indices)

