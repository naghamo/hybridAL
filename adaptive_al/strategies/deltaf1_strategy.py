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

from ..evaluation import (
    evaluate_model,
    calculate_user_spectral_alpha,
    calculate_within_class_variance,
)

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


        self._prev_metric: Optional[float] = None
        self.switch_round: Optional[int] = None
        self._internal_round = 0

        self.fine_tune = FineTuneStrategy(strategy=self)
        self.retrain = RetrainStrategy(strategy=self)

    def _calc_signal(self, val_dataset, pool=None) -> float:
        """
        Compute the raw quantity behind the active monitoring signal. The
        caller (_compute_active_signal_value) handles delta semantics where
        applicable.

        Args:
            val_dataset: The fixed validation dataset.
            pool: Optional DataPool; only used by signals that need labeled
                training data (e.g. delta_nc).
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
        elif self.signal == "delta_spectral_alpha":
            return calculate_user_spectral_alpha(self.model)
        elif self.signal == "delta_nc":
            if pool is None:
                raise ValueError("delta_nc requires the DataPool (for labeled data)")
            labeled_subset = pool.get_labeled_subset()
            return calculate_within_class_variance(
                self.model, labeled_subset, self.batch_size, self.device,
            )
        else:
            raise ValueError(
                f"Unknown signal '{self.signal}'. "
                "Choose from: 'delta_f1', 'delta_loss', 'delta_accuracy', "
                "'gradient_norm', 'spectral_alpha', 'nc1_ratio', 'cka', "
                "'l2_weight_distance', 'delta_spectral_alpha', 'delta_nc'."
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


    _DELTA_STYLE_SIGNALS = (
        "delta_f1", "delta_accuracy", "delta_loss",
        "delta_spectral_alpha", "delta_nc",
    )

    def _compute_active_signal_value(self, val_dataset, pool=None) -> float:
        """
        Map _calc_signal into a "small = stabilized" scalar comparable to
        epsilon. Round 1 returns +inf for any signal that needs a previous
        round, so switching can never trigger on the first round.

        - delta_f1/delta_accuracy/delta_loss   : |metric_t - metric_{t-1}|
        - delta_spectral_alpha                  : |α_t - α_{t-1}|
        - delta_nc                              : |NC_t - NC_{t-1}|
        - cka                                   : 1 - CKA_t   (raw CKA → 1 at stable)
        - gradient_norm, l2_weight_distance,
          spectral_alpha, nc1_ratio             : raw value (already small=stable)
        """
        raw = self._calc_signal(val_dataset, pool=pool)

        if self.signal in self._DELTA_STYLE_SIGNALS:
            if raw is None or not (raw == raw):
                return float("inf")
            if self._prev_metric is None:
                self._prev_metric = raw
                return float("inf")
            value = abs(raw - self._prev_metric)
            self._prev_metric = raw
            return value

        if self.signal == "cka":
            if not (raw == raw):
                return float("inf")
            return 1.0 - raw



        if not (raw == raw):
            return float("inf")
        return raw

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        """
        Run this round's training only — pure retrain pre-switch, fine-tune post-switch.

        The switching decision is taken later in `update_switching_state(...)`,
        which the AL loop calls after `_compute_round_signals` has run. Doing it
        there lets HybridAL reuse the signal value already computed by
        `log_all_signals` instead of triggering an extra forward (and, for
        gradient_norm, an extra backward) pass that would advance the global
        CUDA RNG and silently desynchronize the next round's dropout from a
        Retrain-only run.

        Args:
            pool (DataPool): Current data pool with labeled/unlabeled splits.
            new_indices (List[int]): Newly sampled indices.

        Returns:
            Dict: Training statistics from the underlying strategy.
        """
        self._internal_round += 1
        if self.switched:
            return self.fine_tune._train_implementation(pool, new_indices)
        return self.retrain._train_implementation(pool, new_indices)




    _SIGNAL_FROM_DICT = {
        "delta_f1": "delta_f1",
        "delta_accuracy": "delta_accuracy",
        "delta_loss": "delta_loss",
        "gradient_norm": "gradient_norm",
        "l2_weight_distance": "l2_weight_distance",
        "cka": "cka",
        "delta_spectral_alpha": "delta_spectral_alpha",
        "delta_nc": "delta_nc",
    }

    def _signal_value_from_dict(self, signals: Dict[str, float]) -> float:
        """Pull the active signal's switching value from a precomputed dict.

        Returns +inf if the signal is missing or NaN; applies the cka-specific
        1-x transform so the result is comparable to epsilon (small = stable).
        """
        key = self._SIGNAL_FROM_DICT.get(self.signal)
        if key is None or signals is None or key not in signals:
            return float("inf")
        raw = signals[key]
        try:
            if raw is None or raw != raw:
                return float("inf")
        except TypeError:
            return float("inf")
        if self.signal == "cka":
            return 1.0 - raw
        return raw

    def update_switching_state(
        self,
        val_dataset=None,
        signals: Optional[Dict[str, float]] = None,
        pool: Optional[DataPool] = None,
    ) -> None:
        """Decide whether to switch to FineTune for this round.

        Called by the AL loop after the round's val stats and per-round signals
        have been computed. Prefers the precomputed `signals` dict (zero extra
        compute → identical CUDA RNG state to a Retrain-only run); falls back
        to its own forward pass when log_all_signals=False.
        """
        if self.switched:
            return

        if signals is not None and self.signal in self._SIGNAL_FROM_DICT:
            switching_value = self._signal_value_from_dict(signals)
        else:
            switching_value = self._compute_active_signal_value(val_dataset, pool=pool)

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
            "delta_spectral_alpha": "Δα",
            "delta_nc": "ΔNC",
        }.get(self.signal, self.signal)
        raw_str = "inf" if switching_value == float("inf") else f"{switching_value:.6f}"
        norm_str = "inf" if normalized == float("inf") else f"{normalized:.6f}"

        if self.count >= self.k:
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
