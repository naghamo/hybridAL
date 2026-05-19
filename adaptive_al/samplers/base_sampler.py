"""
Base sampler class for active learning sample selection.

This module defines the abstract BaseSampler class that provides common
functionality for selecting samples from the unlabeled pool. Samplers
implement different acquisition functions to determine which samples
are most valuable to label next.
"""

import logging
import random
from abc import ABC, abstractmethod
from typing import List, Set, Any
from ..pool import DataPool

import torch.nn as nn


class BaseSampler(ABC):
    """
    Abstract base class for active learning sample selection strategies.

    Samplers implement acquisition functions that score and select the most
    informative samples from the unlabeled pool. Different samplers use
    different criteria such as uncertainty, diversity, or representativeness.

    Attributes:
        seed (int): Random seed for reproducible sampling.
        model (nn.Module): Model used for making predictions/computing scores.
        batch_size (int): Batch size for processing data during selection.
        device (str): Device for computations ('cuda' or 'cpu').
    """

    def __init__(self, model: nn.Module, batch_size: int, device: str, seed: int) -> None:
        """
        Initialize the base sampler.

        Args:
            model (nn.Module): Model to use for sample selection.
            batch_size (int): Batch size for data processing.
            device (str): Device for computations ('cuda' or 'cpu').
            seed (int): Random seed for reproducibility (None for no seeding).
        """
        self.seed = seed

        if seed is not None:
            random.seed(seed)
        else:
            logging.warning("No seed was provided for RandomSampler.")

        self.model = model.to(device)
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def select(self,
                   pool: DataPool,
                   num_samples: int) -> List[int]:
        """
        Select the most informative samples from the unlabeled pool (must be overridden).

        Subclasses must implement this method to define their specific
        acquisition function (e.g., uncertainty sampling, random sampling, ...).

        Args:
            pool (DataPool): Data pool containing labeled and unlabeled samples.
            num_samples (int): Number of samples to select from unlabeled pool.

        Returns:
            List[int]: Indices of selected samples from the unlabeled pool.
        """
        pass

    def get_random_unlabeled(self, pool: DataPool, num_of_indices: int) -> List[int]:
        """
        Randomly select samples from the unlabeled pool.

        This helper method provides random sampling functionality that can be
        used by samplers for baseline comparison or as part of hybrid strategies.
        Returns fewer samples if the requested number exceeds available unlabeled data.

        Args:
            pool (DataPool): Data pool containing labeled and unlabeled samples.
            num_of_indices (int): Number of samples to randomly select.

        Returns:
            List[int]: Randomly selected indices from unlabeled pool (empty list
                       if no unlabeled samples remain).
        """
        unlabeled_indices = pool.get_unlabeled_indices()

        if len(unlabeled_indices) == 0:
            return []


        actual_batch_size = min(num_of_indices, len(unlabeled_indices))


        selected_indices = random.sample(unlabeled_indices, actual_batch_size)

        return selected_indices