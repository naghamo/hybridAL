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
from ..evaluation import evaluate_model

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
                 device: str = None, epochs: int = None, batch_size: int = None,
                 val_dataset=None,
                 early_stopping_patience: int = 2,
                 early_stopping_min_delta: float = 1e-4):
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

            # Inherit early-stopping configuration from parent strategy so that
            # nested strategies (DeltaF1Strategy, FixedSwitchStrategy) use the
            # exact same val set, patience, and min_delta as everyone else.
            self.val_dataset = strategy.val_dataset
            self.early_stopping_patience = strategy.early_stopping_patience
            self.early_stopping_min_delta = strategy.early_stopping_min_delta
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

            self.val_dataset = val_dataset
            self.early_stopping_patience = early_stopping_patience
            self.early_stopping_min_delta = early_stopping_min_delta

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

    def train_epochs(self, dataloader) -> Tuple[float, int, int]:
        """
        Train the model for up to self.epochs epochs with validation-loss early stopping.

        After every epoch, validation loss is measured on self.val_dataset using the
        same evaluate_model entry point used elsewhere in the pipeline. If validation
        loss fails to improve by at least self.early_stopping_min_delta for
        self.early_stopping_patience consecutive epochs, the loop breaks early. The
        weights with the lowest observed validation loss are restored before returning.

        Args:
            dataloader: DataLoader providing training batches.

        Returns:
            tuple: (total_loss, num_batches, actual_epochs) — total_loss/num_batches
            cover only the epochs that actually ran; actual_epochs is the number
            of epochs executed before early stopping (or self.epochs if no break).
        """
        start_time = time.time()
        total_loss = 0.0
        num_batches = 0
        actual_epochs = 0

        # evaluate_model toggles model.eval(); remember the mode train_epochs was
        # called in so we can restore it for the next epoch's gradient updates.
        initial_training_mode = self.model.training

        best_val_loss = float('inf')
        best_state = None
        epochs_since_improve = 0

        for epoch in range(self.epochs):
            self.model.train(initial_training_mode)

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

            total_loss += epoch_loss
            num_batches += epoch_batches
            actual_epochs += 1

            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0
            epoch_time = time.time() - epoch_start_time

            val_loss = self._compute_val_loss()

            if val_loss is None:
                # No validation set available; behave like the pre-ES loop.
                tqdm.write(
                    f"    Epoch {epoch + 1}/{self.epochs}  |  "
                    f"Avg Loss: {avg_epoch_loss:.4f}  |  Time: {epoch_time:.1f}s"
                )
                continue

            improved = (best_val_loss - val_loss) > self.early_stopping_min_delta
            if improved:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            tqdm.write(
                f"    Epoch {epoch + 1}/{self.epochs}  |  "
                f"Avg Loss: {avg_epoch_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  "
                f"Best: {best_val_loss:.4f}  |  "
                f"Patience: {epochs_since_improve}/{self.early_stopping_patience}  |  "
                f"Time: {epoch_time:.1f}s"
            )

            if epochs_since_improve >= self.early_stopping_patience:
                tqdm.write(
                    f"    [EarlyStop] No val-loss improvement >= {self.early_stopping_min_delta} "
                    f"for {self.early_stopping_patience} epochs — stopping at epoch "
                    f"{actual_epochs}/{self.epochs}"
                )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.train(initial_training_mode)

        total_time = time.time() - start_time
        logging.info(
            f"Training completed in {total_time:.2f}s "
            f"(actual_epochs={actual_epochs}/{self.epochs})"
        )

        return total_loss, num_batches, actual_epochs

    def _compute_val_loss(self):
        """Return validation loss on self.val_dataset, or None if not available."""
        if self.val_dataset is None:
            return None
        metrics = evaluate_model(
            self.model, self.criterion, self.batch_size,
            dataset=self.val_dataset, device=self.device,
        )
        return metrics["loss"]

    def get_stats(self, total_loss, num_batches, tot_samples, new_samples,
                  actual_epochs=None):
        """
        Compute training statistics for the current round.

        Args:
            total_loss (float): Cumulative loss across all batches.
            num_batches (int): Total number of batches processed.
            tot_samples: All samples used in training this round.
            new_samples: Newly added samples this round.
            actual_epochs (int, optional): Number of epochs actually trained
                (after early stopping). Falls back to self.epochs when not
                supplied so older callers do not break.

        Returns:
            Dict: Statistics including avg_loss, epochs (max), actual_epochs,
                  total_samples, new_samples.
        """
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {
            "avg_loss": avg_loss,
            "epochs": self.epochs,
            "actual_epochs": actual_epochs if actual_epochs is not None else self.epochs,
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