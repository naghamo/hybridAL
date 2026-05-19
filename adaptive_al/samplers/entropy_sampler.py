"""
Entropy-based uncertainty sampling for active learning.

This module implements an acquisition function that selects samples based
on predictive entropy, choosing examples where the model is most uncertain.
Optionally supports hybrid selection with a random component.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import List, Any

from tqdm import tqdm

from .base_sampler import BaseSampler
from ..pool import DataPool


class EntropySampler(BaseSampler):
    """
    Selects samples with the highest predictive entropy (uncertainty).

    This sampler computes the entropy of the model's predicted probability
    distribution for each unlabeled sample and selects those with the highest
    entropy values. High entropy indicates the model is uncertain about the
    prediction, making these samples potentially valuable for improving
    model performance.

    Attributes:
        show_progress (bool): Whether to display a progress bar during selection, since it may take a lot of time.
    """

    def __init__(self, show_progress=False, **kwargs):
        """
        Initialize the entropy sampler.

        Args:
            show_progress (bool): Whether to show progress bar during sampling (default: False).
            **kwargs: Arguments passed to BaseSampler (model, batch_size, device, seed).
        """
        super().__init__(**kwargs)
        self.show_progress = show_progress

    def get_unlabeled_indices(self, pool: DataPool):
        """
        Get indices of all unlabeled samples from the pool.

        Args:
            pool (DataPool): Data pool containing labeled and unlabeled samples.

        Returns:
            List[int]: Indices of unlabeled samples.
        """
        return pool.get_unlabeled_indices()

    def _are_unlabeled(self,pool, suspicious_indices):
        """
        Verify that all given indices are in the unlabeled pool.

        Args:
            pool (DataPool): Data pool to check against.
            suspicious_indices: Indices to verify.

        Returns:
            bool: True if all indices are unlabeled, False otherwise.
        """
        unlabeled = pool.get_unlabeled_indices()
        return all(i in unlabeled for i in suspicious_indices)

    def select(self, pool: DataPool, num_samples: int, random_indices_fraction: float = 0) -> List[int]:
        """
        Select samples with highest predictive entropy from the unlabeled pool.

        Computes entropy of model predictions for all unlabeled samples and
        selects those with highest uncertainty.

        Args:
            pool (DataPool): Data pool containing labeled and unlabeled samples.
            num_samples (int): Total number of samples to select.
            random_indices_fraction (float): Fraction of samples to select randomly
                                            instead of by entropy (default: 0).

        Returns:
            List[int]: Selected indices, combining high-entropy and random samples.
        """
        self.model.eval()
        self.model.to(self.device)

        unlabeled_indices = self.get_unlabeled_indices(pool)


        if len(unlabeled_indices) < num_samples:
            return unlabeled_indices
        random_indices_len = int(num_samples * random_indices_fraction)
        high_entropy_indices_len = num_samples - random_indices_len
        selected_indices = random.sample(unlabeled_indices, random_indices_len)
        unlabeled_indices = [i for i in unlabeled_indices if i not in selected_indices]

        if not unlabeled_indices:
            return []

        unlabeled_subset = pool.get_subset(unlabeled_indices)
        dataloader = DataLoader(
            unlabeled_subset,
            batch_size=self.batch_size,
            shuffle=False
        )

        all_entropies = []


        iterator = dataloader
        if self.show_progress:
            iterator = tqdm(dataloader, desc="Entropy Sampling", leave=False)

        with torch.no_grad():
            for batch in iterator:
                inputs, _ = batch
                inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

                outputs = self.model(**inputs)
                logits = outputs['logits']

                probs = F.softmax(logits, dim=-1)

                log_probs = torch.log(probs)
                entropy_vals = -torch.sum(probs * log_probs, dim=1)

                all_entropies.append(entropy_vals.cpu())

        all_entropies = np.concatenate([t.numpy() for t in all_entropies])


        top_k_indices = np.argpartition(all_entropies, -high_entropy_indices_len)[-high_entropy_indices_len:]


        selected_indices.extend([unlabeled_indices[i] for i in list(top_k_indices)])

        return selected_indices
