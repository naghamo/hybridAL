"""
Data pool management for active learning.

This module provides the DataPool class for managing labeled and unlabeled
data splits during active learning rounds, including efficient tracking of
which samples have been labeled and utilities for creating data subsets.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
from torch.utils.data import Subset
from scipy.stats import entropy

class DataPool:
    """
    Manages labeled and unlabeled data pools for active learning.

    This class tracks which training samples are labeled vs unlabeled,
    provides methods to move samples from unlabeled to labeled pools,
    and creates PyTorch Subset objects for efficient data loading.

    Attributes:
        train_dataset: Full training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        labeled_indices (Set[int]): Set of indices for labeled training samples.
        unlabeled_indices (Set[int]): Set of indices for unlabeled training samples.
        history (List[Dict]): Historical tracking of pool changes (currently unused).
    """

    def __init__(self, train_dataset, val_dataset, test_dataset, initial_labeled_indices: List[int]):
        """
        Initialize the data pool with initial labeled samples.

        Args:
            train_dataset: Full training dataset.
            val_dataset: Validation dataset.
            test_dataset: Test dataset.
            initial_labeled_indices (List[int]): Indices of initially labeled samples.
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset


        self.labeled_indices: Set[int] = set(initial_labeled_indices)
        self.unlabeled_indices: Set[int] = set(range(len(train_dataset))) - self.labeled_indices


        self.history: List[Dict] = []

    def get_subset(self, indices: List[int]):
        """
        Create a Subset of the training dataset for given indices.

        Args:
            indices (List[int]): Indices to include in the subset.

        Returns:
            Subset: PyTorch Subset containing only the specified indices.
        """
        return Subset(self.train_dataset, list(indices))

    def add_labeled_samples(self, new_indices: List[int]) -> None:
        """
        Move samples from unlabeled to labeled pool.

        Args:
            new_indices (List[int]): Indices to move to labeled pool.

        Raises:
            ValueError: If any index is not currently in the unlabeled pool.
        """
        new_indices_set = set(new_indices)


        invalid_indices = new_indices_set - self.unlabeled_indices
        if invalid_indices:
            raise ValueError(f"Indices {invalid_indices} are not in unlabeled pool")


        self.labeled_indices.update(new_indices_set)
        self.unlabeled_indices -= new_indices_set

    def get_labeled_subset(self):
        """
        Get Subset containing all labeled training data.

        Returns:
            Subset: PyTorch Subset of labeled samples.
        """
        return Subset(self.train_dataset, list(self.labeled_indices))

    def get_unlabeled_subset(self):
        """
        Get Subset containing all unlabeled training data.

        Returns:
            Subset: PyTorch Subset of unlabeled samples.
        """
        return Subset(self.train_dataset, list(self.unlabeled_indices))

    def get_unlabeled_indices(self) -> List[int]:
        """
        Get list of all unlabeled sample indices.

        Returns:
            List[int]: Indices of unlabeled samples.
        """
        return list(self.unlabeled_indices)

    def get_labeled_indices(self) -> List[int]:
        """
        Get list of all labeled sample indices.

        Returns:
            List[int]: Indices of labeled samples.
        """
        return list(self.labeled_indices)

    def get_pool_stats(self) -> Dict[str, int]:
        """
        Get current statistics about the data pool.

        Returns:
            Dict[str, int]: Dictionary containing 'labeled_count',
                           'unlabeled_count', and 'total_count'.
        """
        return {
            "labeled_count": len(self.labeled_indices),
            "unlabeled_count": len(self.unlabeled_indices),
            "total_count": len(self.train_dataset)
        }