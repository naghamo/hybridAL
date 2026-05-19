"""
Entropy sampling on a random subset for computational efficiency.

This module implements a two-stage sampling approach: first randomly selecting
a subset of unlabeled data, then applying entropy-based selection within that
subset. This reduces computational cost when the unlabeled pool is very large.
"""

from .entropy_sampler import EntropySampler
from ..pool import DataPool
import random


class EntropyOnRandomSubsetSampler(EntropySampler):
    """
    Selects high-entropy samples from a random subset of the unlabeled pool.

    This sampler improves efficiency for large unlabeled pools by first
    randomly sampling a subset, then applying entropy-based selection within
    that subset. This two-stage approach reduces the number of model forward
    passes required while still leveraging uncertainty-based acquisition.

    Attributes:
        random_subset_size (int): Maximum size of random subset to consider
                                 before applying entropy selection.
    """

    def __init__(self, random_subset_size: int, **kwargs):
        """
        Initialize the entropy-on-random-subset sampler.

        Args:
            random_subset_size (int): Size of random subset to sample from before
                                     applying entropy selection. If unlabeled pool
                                     is smaller, uses all unlabeled samples.
            **kwargs: Arguments passed to EntropySampler (show_progress, model,
                     batch_size, device, seed).
        """
        super().__init__(**kwargs)
        self.random_subset_size = random_subset_size








        self._rng = random.Random(self.seed)

    def get_unlabeled_indices(self, pool: DataPool):
        """
        Get a random subset of unlabeled indices for entropy computation.

        Returns a random sample of unlabeled indices up to random_subset_size.
        If the unlabeled pool is smaller than random_subset_size, returns all
        unlabeled indices.

        Args:
            pool (DataPool): Data pool containing labeled and unlabeled samples.

        Returns:
            List[int]: Random subset of unlabeled indices.
        """
        unlabeled_indices = pool.get_unlabeled_indices()
        if len(unlabeled_indices) > self.random_subset_size:
            unlabeled_indices = self._rng.sample(unlabeled_indices, self.random_subset_size)
        return unlabeled_indices
