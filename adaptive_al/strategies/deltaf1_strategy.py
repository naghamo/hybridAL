"""
Delta F1 adaptive training strategy.

This module implements our proposed hybrid strategy that adaptively switches from full retraining
to fine-tuning based on a monitoring signal computed on the fixed validation set. When the signal
stops improving for k consecutive rounds, the strategy switches to fine-tuning for efficiency.

The monitoring signal is configurable via the `signal` parameter:
  - "delta_f1"       : change in macro F1 score (default, original behavior)
  - "delta_loss"     : change in validation loss (negated so larger = better)
  - "delta_accuracy" : change in validation accuracy
  - "gradient_norm"  : L2 norm of parameter gradients on the validation set
  - "spectral_alpha" : WeightWatcher-based spectral alpha
  - "nc1_ratio"      : neural-collapse NC1 ratio
  - "cka"            : centered kernel alignment to previous round
  - "l2_weight_distance" : L2 distance from the initial model weights
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

    Monitors a chosen signal on the fixed validation set each round. When the absolute
    change in the signal remains below epsilon for k consecutive rounds, it switches
    from full retraining to incremental fine-tuning for improved efficiency.

    Attributes:
        epsilon (float): Signal change threshold for switching to fine-tuning.
        k (int): Number of consecutive rounds below epsilon before switching.
        signal (str): Which signal to monitor.
        count (int): Current count of consecutive rounds below epsilon threshold.
        switched (bool): Whether strategy has switched to fine-tuning mode.
        switch_round (Optional[int]): The round at which the switch occurred.
        prev_signal (float): Previous round's signal value.
        fine_tune (FineTuneStrategy): Fine-tuning strategy instance.
        retrain (RetrainStrategy): Retraining strategy instance.
    """
    def __init__(self, epsilon: float, k: int, signal: str = "delta_f1",
                 signal_normalizer: Optional[float] = None, **kwargs):
        """
        Initialize the Delta F1 adaptive strategy.

        Args:
            epsilon (float): Threshold for the (normalized) switching value. When
                value_t < epsilon for k consecutive rounds, switch to fine-tuning.
                Values are mapped per-signal into "small=stabilized" form (see
                _compute_active_signal_value); if signal_normalizer is set, the
                value is divided by it before the comparison so a single ε works
                across signals with different raw scales.
            k (int): Number of consecutive rounds below epsilon before switching.
            signal (str): Monitoring signal to use. One of "delta_f1" (default),
                         "delta_loss", "delta_accuracy", "gradient_norm",
                         "spectral_alpha", "nc1_ratio", "cka", "l2_weight_distance".
            signal_normalizer (Optional[float]): Per-signal divisor obtained from
                the calibration run (the round-1-or-first-finite raw value). If
                None, the raw "small=stabilized" value is compared to epsilon
                directly.
            **kwargs: Additional arguments passed to BaseStrategy.
        """
        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.k = k
        self.signal = signal
        self.signal_normalizer = signal_normalizer

        self.count = 0
        self.switched = False
        # Tracks the previous round's raw metric for delta-style signals
        # (delta_f1/delta_accuracy/delta_loss); None at round 1.
        self._prev_metric: Optional[float] = None
        self.switch_round: Optional[int] = None
        self._internal_round = 0

        self.fine_tune = FineTuneStrategy(strategy=self)
        self.retrain = RetrainStrategy(strategy=self)

    def _calc_signal(self, val_dataset) -> float:
        """
        Compute the monitoring signal on the fixed validation set.

        Args:
            val_dataset: The fixed validation dataset.

        Returns:
            float: Signal value (higher = better for all signals; loss is negated).
        """
        if self.signal in ("delta_f1", "delta_loss", "delta_accuracy"):
            stats = evaluate_model(self.model, self.criterion, self.batch_size,
                                   val_dataset, self.device)
            if self.signal == "delta_f1":
                return stats['f1_score']
            elif self.signal == "delta_loss":
                return -stats['loss']
            else:
                return stats['accuracy']
        elif self.signal in ("spectral_alpha", "nc1_ratio", "cka", "l2_weight_distance"):
            stats = evaluate_model(
                self.model,
                self.criterion,
                self.batch_size,
                val_dataset,
                self.device,
                include_advanced_metrics=True,
                # active_learning.train_one_round is the canonical state-updater
                # (it calls evaluate_model with update_state defaulting to True
                # AFTER this call), so we read prev state without disturbing it.
                update_advanced_metric_state=False,
            )
            return stats[self.signal]
        elif self.signal == "gradient_norm":
            return self._calc_gradient_norm(val_dataset)
        else:
            raise ValueError(
                f"Unknown signal '{self.signal}'. "
                "Choose from: 'delta_f1', 'delta_loss', 'delta_accuracy', "
                "'gradient_norm', 'spectral_alpha', 'nc1_ratio', 'cka', "
                "'l2_weight_distance'."
            )

    def _calc_gradient_norm(self, val_dataset) -> float:
        """
        Compute the L2 norm of model parameter gradients on the validation set.

        Args:
            val_dataset: The fixed validation dataset.

        Returns:
            float: L2 norm of all parameter gradients.
        """
        self.model.train()
        self.optimizer.zero_grad()

        loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Backward per-batch and accumulate into .grad. Summing losses across
        # all val batches and backwarding once retains every batch's autograd
        # graph in parallel, which OOMs on val sets of even a few thousand.
        for inputs, targets in loader:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            targets = targets.to(self.device)
            outputs = self.model(**inputs)
            loss = self.criterion(outputs.logits, targets)
            loss.backward()

        norm = sum(
            p.grad.norm().item() ** 2
            for p in self.model.parameters() if p.grad is not None
        ) ** 0.5

        self.optimizer.zero_grad()
        self.model.eval()
        return norm

    def _compute_active_signal_value(self, val_dataset) -> float:
        """
        Map _calc_signal into a "small = stabilized" scalar comparable to
        epsilon. Round 1 returns +inf for any signal that needs a previous
        round, so switching can never trigger on the first round.

        - delta_f1/delta_accuracy/delta_loss : |metric_t - metric_{t-1}|
        - cka                                 : 1 - CKA_t   (raw CKA → 1 at stable)
        - gradient_norm, l2_weight_distance,
          spectral_alpha, nc1_ratio           : raw value (already small-=-stable)
        """
        raw = self._calc_signal(val_dataset)

        if self.signal in ("delta_f1", "delta_accuracy", "delta_loss"):
            # _calc_signal returns the metric (F1 / accuracy / -loss);
            # switching value is the absolute round-to-round change.
            if self._prev_metric is None:
                self._prev_metric = raw
                return float("inf")
            value = abs(raw - self._prev_metric)
            self._prev_metric = raw
            return value

        if self.signal == "cka":
            # NaN at round 1 (no previous representations); treat as +inf so
            # the first round does not count as stabilized.
            if not (raw == raw):  # NaN check without importing math
                return float("inf")
            return 1.0 - raw

        # gradient_norm, l2_weight_distance, spectral_alpha, nc1_ratio:
        # raw is already in "small = stabilized" form. l2 may legitimately be
        # +inf at round 1 (no previous weight snapshot).
        if not (raw == raw):
            return float("inf")
        return raw

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        """
        Train using retrain or fine-tune strategy based on signal stabilization.

        After each retraining round we compute the active signal in
        "small = stabilized" form, optionally normalize by self.signal_normalizer
        (a calibration constant), and switch to fine-tuning once the normalized
        value stays below epsilon for k consecutive rounds.

        Args:
            pool (DataPool): Current data pool with labeled/unlabeled splits.
            new_indices (List[int]): Newly sampled indices.

        Returns:
            Dict: Training statistics from the underlying strategy.
        """
        self._internal_round += 1

        if self.switched:
            return self.fine_tune._train_implementation(pool, new_indices)

        stats = self.retrain._train_implementation(pool, new_indices)

        switching_value = self._compute_active_signal_value(pool.val_dataset)

        normalized = switching_value
        if (self.signal_normalizer is not None
                and self.signal_normalizer > 0
                and switching_value != float("inf")):
            normalized = switching_value / self.signal_normalizer

        if normalized < self.epsilon:
            self.count += 1
        else:
            self.count = 0

        signal_label = {
            "delta_f1": "ΔF1",
            "delta_loss": "ΔLoss",
            "delta_accuracy": "ΔAcc",
            "gradient_norm": "GradNorm",
            "spectral_alpha": "Alpha",
            "nc1_ratio": "NC1",
            "cka": "1-CKA",
            "l2_weight_distance": "dW",
        }.get(self.signal, self.signal)
        raw_str = "inf" if switching_value == float("inf") else f"{switching_value:.6f}"
        norm_str = "inf" if normalized == float("inf") else f"{normalized:.6f}"

        if self.count >= self.k and not self.switched:
            self.switch_round = self._internal_round
            self.switched = True
            tqdm.write(
                f"  [HybridAL] *** SWITCHING to FineTune at round {self.switch_round} ***\n"
                f"             {signal_label}={raw_str}  norm={norm_str}  "
                f"({self.count} consecutive rounds below ε={self.epsilon})"
            )
        else:
            status = (
                f"count {self.count}/{self.k}" if self.count > 0
                else "reset (signal above ε)"
            )
            tqdm.write(
                f"  [HybridAL] {signal_label}={raw_str}  norm={norm_str}  "
                f"ε={self.epsilon}  →  {status}"
            )

        return stats
