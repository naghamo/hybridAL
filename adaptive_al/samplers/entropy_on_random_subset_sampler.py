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
        # Private RNG owned by this sampler. Without this, get_unlabeled_indices
        # would call global random.sample(), so any code that consumes Python's
        # global RNG between rounds (model forward passes, gradient norm in
        # train mode, signal computation) would shift the subset selection.
        # That made HybridAL pre-switch and pure Retrain produce different
        # sample sequences for the same seed even though the training code is
        # identical pre-switch. Owning a separate Random(seed) makes the
        # subset selection reproducible regardless of in-between RNG activity.
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
