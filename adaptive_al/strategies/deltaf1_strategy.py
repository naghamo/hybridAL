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
    def __init__(self, epsilon: float, k: int, signal: str = "delta_f1", **kwargs):
        """
        Initialize the Delta F1 adaptive strategy.

        Args:
            epsilon (float): Threshold for absolute signal change. When |Δsignal| < epsilon
                            for k consecutive rounds, switch to fine-tuning.
            k (int): Number of consecutive rounds below epsilon before switching.
            signal (str): Monitoring signal to use. One of "delta_f1" (default),
                         "delta_loss", "delta_accuracy", "gradient_norm",
                         "spectral_alpha", "nc1_ratio", "cka", "l2_weight_distance".
            **kwargs: Additional arguments passed to BaseStrategy.
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

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        """
        Train using retrain or fine-tune strategy based on signal improvement tracking.

        Uses the fixed validation set (pool.val_dataset) to monitor the signal —
        no acquisition budget is wasted on monitoring.

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

        cur_signal = self._calc_signal(pool.val_dataset)
        delta = cur_signal - self.prev_signal if self.prev_signal is not None else float('inf')
        self.prev_signal = cur_signal

        if abs(delta) < self.epsilon:
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
            "cka": "CKA",
            "l2_weight_distance": "dW",
        }.get(self.signal, self.signal)
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
