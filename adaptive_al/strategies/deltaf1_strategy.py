"""
Delta F1 adaptive training strategy.

This module implements our proposed hybrid strategy that adaptively switches from full retraining
to fine-tuning based on a monitoring signal computed on the fixed validation set. When the signal
stays below epsilon for k consecutive rounds, the strategy switches to fine-tuning for efficiency.

Every round, ALL of the following signals are computed and logged in the round stats under
`"signals"`, regardless of which one is the active switching signal:

  - "delta_f1"            : |F1_t - F1_{t-1}|     on the fixed validation set
  - "delta_accuracy"      : |Acc_t - Acc_{t-1}|   on the fixed validation set
  - "delta_loss"          : |Loss_t - Loss_{t-1}| on the fixed validation set
  - "gradient_norm"       : ||∇_θ L_val(θ_t)||_2  (L2 norm of param grads on val set)
  - "l2_weight_distance"  : ||θ_t - θ_{t-1}||_2 / sqrt(num_params)
  - "cka"                 : 1 - linear_CKA(repr_{t-1}, repr_t) on val [CLS] features

For all signals we use the same convention: "value < ε means stabilized". At t=1 every signal
that requires a previous round (everything except gradient_norm) is set to +inf so the
strategy cannot switch on the very first round.
"""

import copy
from typing import Dict, List, Optional

import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_strategy import BaseStrategy
from ..pool import DataPool

from ..evaluation import evaluate_model

from .fine_tuning_strategy import FineTuneStrategy
from .retrain_strategy import RetrainStrategy


SUPPORTED_SIGNALS = (
    "delta_f1",
    "delta_loss",
    "delta_accuracy",
    "gradient_norm",
    "l2_weight_distance",
    "cka",
)


class DeltaF1Strategy(BaseStrategy):
    """
    Adaptive strategy that switches from retraining to fine-tuning based on a monitoring signal.

    Monitors a chosen signal on the fixed validation set each round. When the signal value
    stays below epsilon for k consecutive rounds, it switches from full retraining to
    incremental fine-tuning. All signals are computed every round and exposed in the
    returned stats dict under "signals", so a single run yields per-round values for every
    candidate signal.
    """

    def __init__(self, epsilon: float, k: int, signal: str = "delta_f1", **kwargs):
        """
        Args:
            epsilon (float): Threshold for the active switching signal. When
                signal_t < epsilon for k consecutive rounds, switch to fine-tuning.
            k (int): Number of consecutive rounds below epsilon before switching.
            signal (str): Active switching signal. One of SUPPORTED_SIGNALS.
            **kwargs: Additional arguments passed to BaseStrategy.
        """
        super().__init__(**kwargs)

        if signal not in SUPPORTED_SIGNALS:
            raise ValueError(
                f"Unknown switching signal '{signal}'. "
                f"Choose from: {SUPPORTED_SIGNALS}."
            )

        self.epsilon = epsilon
        self.k = k
        self.signal = signal

        self.count = 0
        self.switched = False
        self.switch_round: Optional[int] = None
        self._internal_round = 0

        # State carried between rounds for the "model-change" signals.
        self._prev_metrics: Optional[Dict[str, float]] = None  # f1, acc, loss
        self._prev_state_dict_cpu: Optional[Dict[str, torch.Tensor]] = None
        self._prev_repr: Optional[torch.Tensor] = None  # (n_val, dim) on CPU

        self.fine_tune = FineTuneStrategy(strategy=self)
        self.retrain = RetrainStrategy(strategy=self)

    # ------------------------------------------------------------------ #
    # Per-signal primitives                                                #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _snapshot_state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        """Return a CPU clone of the current model state_dict for next-round comparison."""
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    @torch.no_grad()
    def _l2_weight_distance(self, prev_state_dict_cpu: Dict[str, torch.Tensor]) -> float:
        """
        Compute ||θ_t - θ_{t-1}||_2 / sqrt(num_params) over all floating-point parameters.

        Buffers (e.g. running stats with non-floating dtypes) are skipped; the comparison
        uses CPU tensors so no GPU memory is held for the previous snapshot.
        """
        cur = self.model.state_dict()
        sum_sq = 0.0
        n = 0
        for key, prev_v in prev_state_dict_cpu.items():
            if not torch.is_floating_point(prev_v):
                continue
            cur_v = cur[key].detach().to("cpu", dtype=prev_v.dtype, copy=False)
            diff = (cur_v - prev_v).flatten()
            sum_sq += float((diff * diff).sum())
            n += diff.numel()
        if n == 0:
            return 0.0
        return (sum_sq / n) ** 0.5

    @torch.no_grad()
    def _extract_representations(self, val_dataset) -> torch.Tensor:
        """
        Run val_dataset through the body of the HF model (no classification head)
        and return the [CLS] hidden state for each example as a (n, dim) CPU tensor.

        Works uniformly for distilbert / bert / roberta because all
        AutoModelForSequenceClassification variants expose `model.base_model`,
        and their base models return last_hidden_state of shape (batch, seq, dim).
        """
        self.model.eval()
        loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        chunks = []
        for inputs, _targets in loader:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.base_model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]
            chunks.append(cls.detach().to("cpu", dtype=torch.float32))
        return torch.cat(chunks, dim=0)

    @staticmethod
    @torch.no_grad()
    def _linear_cka(prev_repr: torch.Tensor, cur_repr: torch.Tensor) -> float:
        """
        Linear Centered Kernel Alignment between two representation matrices of
        shape (n, d). Both inputs are centered column-wise before computing:

            CKA = ||Y_c^T X_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)
        """
        X = prev_repr.float()
        Y = cur_repr.float()
        Xc = X - X.mean(dim=0, keepdim=True)
        Yc = Y - Y.mean(dim=0, keepdim=True)

        cross = Yc.t() @ Xc
        XtX = Xc.t() @ Xc
        YtY = Yc.t() @ Yc

        num = torch.linalg.norm(cross) ** 2
        den = torch.linalg.norm(XtX) * torch.linalg.norm(YtY)
        if float(den) <= 0.0:
            return 0.0
        return float((num / den).clamp(min=0.0, max=1.0))

    def _calc_gradient_norm(self, val_dataset) -> float:
        """L2 norm of model parameter gradients accumulated over the val set."""
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

    # ------------------------------------------------------------------ #
    # Combined signal computation                                          #
    # ------------------------------------------------------------------ #

    def _compute_all_signals_and_update(self, pool: DataPool) -> Dict[str, float]:
        """
        Evaluate every candidate switching signal once and update the running state
        used to compute next-round signals. Returns a dict of scalars; the dict is
        log-friendly (no tensors) and uses the convention "value < ε ⇒ stabilized"
        for every signal.
        """
        val_dataset = pool.val_dataset

        # 1) Validation metrics (single forward pass, also reused by delta_f1/acc/loss).
        val_stats = evaluate_model(
            self.model, self.criterion, self.batch_size, val_dataset, self.device,
        )
        f1_t = val_stats["f1_score"]
        acc_t = val_stats["accuracy"]
        loss_t = val_stats["loss"]

        if self._prev_metrics is None:
            delta_f1 = float("inf")
            delta_acc = float("inf")
            delta_loss = float("inf")
        else:
            delta_f1 = abs(f1_t - self._prev_metrics["f1"])
            delta_acc = abs(acc_t - self._prev_metrics["acc"])
            delta_loss = abs(loss_t - self._prev_metrics["loss"])

        # 2) Penultimate-layer representations for CKA.
        cur_repr = self._extract_representations(val_dataset)
        if self._prev_repr is None:
            cka_signal = float("inf")
        else:
            cka_signal = 1.0 - self._linear_cka(self._prev_repr, cur_repr)

        # 3) L2 weight distance to previous snapshot.
        if self._prev_state_dict_cpu is None:
            l2_dist = float("inf")
        else:
            l2_dist = self._l2_weight_distance(self._prev_state_dict_cpu)

        # 4) Gradient norm on val set (always computable; not delta-style).
        grad_norm_t = self._calc_gradient_norm(val_dataset)

        signals = {
            "delta_f1": float(delta_f1),
            "delta_accuracy": float(delta_acc),
            "delta_loss": float(delta_loss),
            "gradient_norm": float(grad_norm_t),
            "l2_weight_distance": float(l2_dist),
            "cka": float(cka_signal),
        }

        # Persist state for the next round.
        self._prev_metrics = {"f1": f1_t, "acc": acc_t, "loss": loss_t}
        self._prev_state_dict_cpu = self._snapshot_state_dict_cpu()
        self._prev_repr = cur_repr

        return signals

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #

    def _train_implementation(self, pool: DataPool, new_indices: List[int]) -> Dict:
        """
        Train using retrain or fine-tune; then compute all signals and decide whether to switch.

        Args:
            pool (DataPool): Current data pool with labeled/unlabeled splits.
            new_indices (List[int]): Newly sampled indices.

        Returns:
            Dict: Training stats from the underlying strategy, plus a "signals" key
                  with the full per-round signal dict.
        """
        self._internal_round += 1

        if self.switched:
            stats = self.fine_tune._train_implementation(pool, new_indices)
            # Keep logging signals in fine-tune phase too, for full per-round trajectories.
            stats["signals"] = self._compute_all_signals_and_update(pool)
            return stats

        stats = self.retrain._train_implementation(pool, new_indices)
        signals = self._compute_all_signals_and_update(pool)
        stats["signals"] = signals

        active_value = signals[self.signal]

        if active_value < self.epsilon:
            self.count += 1
        else:
            self.count = 0

        signal_label = {
            "delta_f1": "ΔF1", "delta_loss": "ΔLoss", "delta_accuracy": "ΔAcc",
            "gradient_norm": "GradNorm", "l2_weight_distance": "L2Δθ", "cka": "1-CKA",
        }.get(self.signal, self.signal)
        active_str = "inf" if active_value == float("inf") else f"{active_value:.6f}"

        if self.count >= self.k and not self.switched:
            self.switch_round = self._internal_round
            self.switched = True
            tqdm.write(
                f"  [HybridAL] *** SWITCHING to FineTune at round {self.switch_round} ***\n"
                f"             {signal_label} = {active_str}  "
                f"({self.count} consecutive rounds below ε={self.epsilon})"
            )
        else:
            status = (
                f"count {self.count}/{self.k}" if self.count > 0
                else "reset (signal above ε)"
            )
            tqdm.write(
                f"  [HybridAL] {signal_label} = {active_str}  "
                f"ε={self.epsilon}  →  {status}"
            )

        return stats
