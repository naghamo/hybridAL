"""
Base strategy class for active learning training approaches.

This module defines the abstract BaseStrategy class that provides common
training functionality and defines the interface that all training strategies
must implement. Strategies control how models are trained each round.
"""

import copy
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple

from torch import nn
from tqdm import tqdm

from ..config import ExperimentConfig
from ..pool import DataPool

from transformers import AutoModelForSequenceClassification


class BaseStrategy(ABC):
    """
    Abstract base class for active learning training strategies.

    This class provides common training infrastructure including model management,
    optimizer/criterion/scheduler initialization, epoch-based training loops,
    and timing tracking. Subclasses must implement _train_implementation to
    define strategy-specific training logic.

    Strategies can be initialized either from scratch with all components, or
    by copying configuration from an existing strategy instance.
    """

    def __init__(self, *,
                 strategy: "BaseStrategy" = None,
                 model: nn.Module = None,
                 optimizer_cls=None, optimizer_kwargs=None,
                 criterion_cls=None, criterion_kwargs=None,
                 scheduler_cls=None, scheduler_kwargs=None,
                 device: str = None, epochs: int = None, batch_size: int = None):
        """Initialize the strategy either from scratch or by copying another strategy."""
        if strategy is not None:
            # Initialize from another strategy
            self.model = strategy.model
            self.optimizer = strategy.optimizer
            self.criterion = strategy.criterion
            self.scheduler = strategy.scheduler

            self.initial_model_state_dict = strategy.initial_model_state_dict

            self.optimizer_cls = strategy.optimizer_cls
            self.optimizer_kwargs = strategy.optimizer_kwargs
            self.criterion_cls = strategy.criterion_cls
            self.criterion_kwargs = strategy.criterion_kwargs
            self.scheduler_cls = strategy.scheduler_cls
            self.scheduler_kwargs = strategy.scheduler_kwargs

            self.device = strategy.device
            self.epochs = strategy.epochs
            self.batch_size = strategy.batch_size
        else:
            # Store the passed-in class/kwargs
            self.model = model
            self.initial_model_state_dict = copy.deepcopy(model.state_dict()) # Store initial weights for reset

            self.optimizer_cls = optimizer_cls
            self.optimizer_kwargs = optimizer_kwargs
            self.criterion_cls = criterion_cls
            self.criterion_kwargs = criterion_kwargs
            self.scheduler_cls = scheduler_cls
            self.scheduler_kwargs = scheduler_kwargs

            self.device = device
            self.epochs = epochs
            self.batch_size = batch_size
            # self.round_history: List[Dict] = None

            self._initialize_components()

    def train(self, pool: DataPool, new_indices: List[int]) -> Dict:
        """
        Train model for one round with automatic timing.

        This method wraps the strategy-specific _train_implementation with
        timing logic and returns combined statistics.

        Args:
            pool (DataPool): Current data pool with labeled/unlabeled splits.
            new_indices (List[int]): Newly sampled indices to add to training
                                     (not yet added to pool).

        Returns:
            Dict: Training statistics including 'training_time' and any
                  strategy-specific metrics.
        """
        start_time = time.time()

        # Call the strategy-specific training logic
        custom_stats = self._train_implementation(pool, new_indices)

        training_time = time.time() - start_time

        # Add any base statistics...
        base_stats = {
            "training_time": training_time,
        }

        # Merge with strategy-specific stats
        final_stats = {**base_stats, **custom_stats}

        return final_stats

    def _train_batch(self, batch):
        """
        Train on a single batch.

        Args:
            batch: Tuple of (inputs, targets) from DataLoader.

        Returns:
            float: Loss value for this batch.
        """
        inputs, targets = batch
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
        targets = targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(**inputs)

        logits = outputs['logits']
        loss = self.criterion(logits, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epochs(self, dataloader) -> Tuple[float, int]:
        """
        Train the model for the configured number of epochs.

        Args:
            dataloader: DataLoader providing training batches.

        Returns:
            tuple: (total_loss, num_batches) across all epochs.
        """
        start_time = time.time()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_batches = 0

            batch_bar = tqdm(
                dataloader,
                desc=f"    Epoch {epoch + 1}/{self.epochs}",
                leave=False,
                unit="batch",
                dynamic_ncols=True,
            )
            for batch in batch_bar:
                loss = self._train_batch(batch)
                epoch_loss += loss
                epoch_batches += 1
                batch_bar.set_postfix(loss=f"{epoch_loss / epoch_batches:.4f}")

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0
            tqdm.write(
                f"    Epoch {epoch + 1}/{self.epochs}  |  "
                f"Avg Loss: {avg_epoch_loss:.4f}  |  Time: {epoch_time:.1f}s"
            )

            total_loss += epoch_loss
            num_batches += epoch_batches

        total_time = time.time() - start_time
        logging.info(f"Training completed in {total_time:.2f}s")

        return total_loss, num_batches

    def get_stats(self, total_loss, num_batches, tot_samples, new_samples):
        """
        Compute training statistics for the current round.

        Args:
            total_loss (float): Cumulative loss across all batches.
            num_batches (int): Total number of batches processed.
            tot_samples: All samples used in training this round.
            new_samples: Newly added samples this round.

        Returns:
            Dict: Statistics including avg_loss, epochs, total_samples, new_samples.
        """
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Return model and training statistics
        return {
            "avg_loss": avg_loss,
            "epochs": self.epochs,
            "total_samples": len(tot_samples),
            "new_samples": len(new_samples) if new_samples is not None else 0,
        }

    def _initialize_components(self):
        """
        Initialize or reinitialize training components.

        Resets model to initial weights, moves it to device, and creates
        new instances of optimizer, criterion, and scheduler.
        """
        self.model.load_state_dict(self.initial_model_state_dict)
        self.model.to(self.device)
        self.optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)
        self.criterion = self.criterion_cls(**self.criterion_kwargs)

        self.scheduler = None
        if self.scheduler_cls is not None:
            self.scheduler = self.scheduler_cls(self.optimizer, **self.scheduler_kwargs)

    def reset(self):
        """
        Reset model and training components to initial state.

        Useful for strategies that need to retrain from scratch each round.
        """
        logging.info("Resetting model to initial state . . .")
        self._initialize_components()

    @abstractmethod
    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        """
        Strategy-specific training implementation (must be overridden).

        Subclasses must implement this method to define their specific
        training approach (e.g., incremental training, full retraining,
        weighted sampling, etc.).

        Args:
            pool (DataPool): Current data pool with labeled/unlabeled splits.
            new_indices (List[int]): Newly sampled indices to train on
                                     (not yet added to pool).

        Returns:
            Dict: Strategy-specific training statistics.
        """
        pass

    def pass_args_to_sampler(self) -> Dict[str, Any]:
        """
        Provide arguments to be passed to the sampler.

        Strategies can override this to pass additional information
        (e.g., model predictions, uncertainties) to samplers that need it.

        Returns:
            Dict[str, Any]: Arguments to pass to sampler's select method.
        """
        return {}