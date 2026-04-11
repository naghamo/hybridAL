"""
Delta F1 adaptive training strategy.

This module implements our proposed hybrid strategy that adaptively switches from full retraining
to fine-tuning based on a monitoring signal on a small validation subset. When the signal
stops improving for k consecutive rounds, the strategy switches to fine-tuning for efficiency.

The monitoring signal is configurable via the `signal` parameter:
  - "delta_f1"       : change in macro F1 score (default, original behavior)
  - "delta_loss"     : change in validation loss (negated so larger = better)
  - "delta_accuracy" : change in validation accuracy
  - "gradient_norm"  : L2 norm of parameter gradients on the validation subset
"""

from typing import List, Dict, Any, Optional

import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_strategy import BaseStrategy
from ..pool import DataPool

from ..evaluation import evaluate_model

from .fine_tuning_strategy import FineTuneStrategy
from .retrain_strategy import RetrainStrategy

class DeltaF1Strategy(BaseStrategy):
    """
    Adaptive strategy that switches from retraining to fine-tuning based on a monitoring signal.

    This strategy monitors a chosen signal on a validation subset. When the absolute
    change in the signal remains below epsilon for k consecutive rounds, it switches
    from full retraining to incremental fine-tuning for improved efficiency.

    Attributes:
        epsilon (float): Signal change threshold for switching to fine-tuning.
        k (int): Number of consecutive rounds below epsilon before switching.
        validation_fraction (float): Fraction of new samples to use for validation monitoring.
        signal (str): Which signal to monitor. One of: "delta_f1", "delta_loss",
                      "delta_accuracy", "gradient_norm".
        count (int): Current count of consecutive rounds below epsilon threshold.
        switched (bool): Whether strategy has switched to fine-tuning mode.
        switch_round (Optional[int]): The round at which the switch occurred (None if not yet).
        prev_signal (float): Previous round's signal value on validation set.
        validation_indices (List[int]): Indices used for validation monitoring.
        fine_tune (FineTuneStrategy): Fine-tuning strategy instance.
        retrain (RetrainStrategy): Retraining strategy instance.
    """
    def __init__(self, epsilon: float, k: int, validation_fraction: float = 0,
                 signal: str = "delta_f1", **kwargs):
        """
        Initialize the Delta F1 adaptive strategy.

        Args:
            epsilon (float): Threshold for absolute signal change. When |Δsignal| < epsilon
                            for k consecutive rounds, switch to fine-tuning.
            k (int): Number of consecutive rounds below epsilon before switching.
            validation_fraction (float): Fraction of each new batch to reserve for
                                        validation monitoring (default: 0).
            signal (str): Monitoring signal to use for the switching decision.
                         One of "delta_f1" (default), "delta_loss", "delta_accuracy",
                         "gradient_norm".
            **kwargs: Additional arguments passed to BaseStrategy (model, optimizer, etc.).
        """
        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.k = k
        self.signal = signal

        self.count = 0
        self.switched = False
        self.prev_signal = None
        self.switch_round: Optional[int] = None
        self._internal_round = 0

        self.validation_indices = []
        self.validation_fraction = validation_fraction

        self.fine_tune = FineTuneStrategy(strategy=self)
        self.retrain = RetrainStrategy(strategy=self)

    def pass_args_to_sampler(self) -> Dict[str, Any]:
        """
        Provide validation fraction to sampler for random selection.

        Returns:
            Dict[str, Any]: Dictionary with 'random_indices_fraction' set to
                           validation_fraction before switch, 0 after switch.
        """
        return {"random_indices_fraction": 0 if self.switched else self.validation_fraction}

    def _calc_signal(self, subset_to_evaluate) -> float:
        """
        Compute the monitoring signal on the validation subset.

        For metric-based signals (delta_f1, delta_loss, delta_accuracy), evaluates
        the model on the given subset. For gradient_norm, computes the L2 norm of
        parameter gradients after a forward+backward pass on the subset.

        Args:
            subset_to_evaluate: Dataset subset to evaluate on.

        Returns:
            float: Signal value (higher = better for all signals; loss is negated).
        """
        if self.signal in ("delta_f1", "delta_loss", "delta_accuracy"):
            stats = evaluate_model(self.model, self.criterion, self.batch_size,
                                   subset_to_evaluate, self.device)
            if self.signal == "delta_f1":
                return stats['f1_score']
            elif self.signal == "delta_loss":
                return -stats['loss']   # negate so that improvement = increasing value
            else:
                return stats['accuracy']
        elif self.signal == "gradient_norm":
            return self._calc_gradient_norm(subset_to_evaluate)
        else:
            raise ValueError(
                f"Unknown signal '{self.signal}'. "
                "Choose from: 'delta_f1', 'delta_loss', 'delta_accuracy', 'gradient_norm'."
            )

    def _calc_gradient_norm(self, subset_to_evaluate) -> float:
        """
        Compute the L2 norm of model parameter gradients on the validation subset.

        Runs a forward+backward pass on the subset to obtain gradients, computes
        the overall gradient norm, then zeroes the gradients without updating weights.

        Args:
            subset_to_evaluate: Dataset subset to evaluate on.

        Returns:
            float: L2 norm of all parameter gradients.
        """
        self.model.train()
        self.optimizer.zero_grad()

        loader = DataLoader(subset_to_evaluate, batch_size=self.batch_size, shuffle=False)
        total_loss = torch.tensor(0.0, device=self.device)

        for inputs, targets in loader:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            targets = targets.to(self.device)
            outputs = self.model(**inputs)
            total_loss = total_loss + self.criterion(outputs.logits, targets)

        total_loss.backward()

        norm = sum(
            p.grad.norm().item() ** 2
            for p in self.model.parameters() if p.grad is not None
        ) ** 0.5

        self.optimizer.zero_grad()
        self.model.eval()
        return norm

    def _separate_validation_and_train(self, new_indices: List[int]):
        """
        Split new indices into validation and training sets.

        Allocates the first validation_fraction of indices for validation monitoring,
        with the remainder for training.

        Args:
            new_indices (List[int]): Newly sampled indices to split.

        Returns:
            tuple: (validation_indices, training_indices). Returns ([], None) if
                   new_indices is empty.
        """
        if not new_indices:
            return [], None
        else:
            new_validation_size = int(len(new_indices)*self.validation_fraction)
            return new_indices[:new_validation_size], new_indices[new_validation_size:]

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        """
        Train using retrain or fine-tune strategy based on signal improvement tracking.

        If switched to fine-tuning mode, delegates to FineTuneStrategy. Otherwise,
        separates new indices into validation and training sets, trains using
        RetrainStrategy, evaluates the monitoring signal on the validation set, and
        tracks consecutive rounds below epsilon threshold.

        Args:
            pool (DataPool): Current data pool with labeled/unlabeled splits.
            new_indices (List[int]): Newly sampled indices (not yet in pool).

        Returns:
            Dict: Training statistics from the underlying strategy.
        """
        self._internal_round += 1

        if self.switched:
            return self.fine_tune._train_implementation(pool, new_indices)

        new_validation_indices, new_train_indices = self._separate_validation_and_train(new_indices)
        self.validation_indices.extend(new_validation_indices)
        stats = self.retrain._train_implementation(pool, new_train_indices)

        if not self.validation_indices:
            return stats

        pool.add_labeled_samples(new_validation_indices)
        cur_signal = self._calc_signal(pool.get_subset(self.validation_indices))
        delta = cur_signal - self.prev_signal if self.prev_signal is not None else float('inf')
        self.prev_signal = cur_signal

        if abs(delta) < self.epsilon:
            self.count += 1
        else:
            self.count = 0

        signal_label = {"delta_f1": "ΔF1", "delta_loss": "ΔLoss",
                        "delta_accuracy": "ΔAcc", "gradient_norm": "GradNorm"}.get(self.signal, self.signal)
        delta_str = f"{delta:+.6f}" if self.prev_signal is not None else "—"

        if self.count >= self.k and not self.switched:
            self.switch_round = self._internal_round
            self.switched = True
            tqdm.write(
                f"  [HybridAL] *** SWITCHING to FineTune at round {self.switch_round} ***\n"
                f"             {signal_label} = {cur_signal:.6f}  Δ = {delta_str}  "
                f"({self.count} consecutive rounds below ε={self.epsilon})"
            )
        else:
            status = f"count {self.count}/{self.k}" if self.count > 0 else "reset (improvement detected)"
            tqdm.write(
                f"  [HybridAL] {signal_label} = {cur_signal:.6f}  Δ = {delta_str}  "
                f"ε={self.epsilon}  →  {status}"
            )

        return stats
