"""
Active learning orchestration and experiment management.

This module contains the main ActiveLearning class that manages the complete
active learning pipeline, including model initialization, training rounds,
sampling strategies, and result tracking.
"""

from datetime import datetime
import json
import random
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

import numpy as np
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

import torch.nn as nn
import torch.optim as optim

import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .config import ExperimentConfig
from .pool import DataPool

import adaptive_al.strategies as strategies
import adaptive_al.samplers as samplers


from .utils.data_loader import load_agnews, load_imdb, load_jigsaw, load_sst2, load_tweeteval, load_yahoo_answers

from .evaluation import (
    evaluate_model,
    compute_confusion_matrix,
    initialize_evaluation_state,
    calculate_user_spectral_alpha,
    calculate_within_class_variance,
    calculate_linear_cka,
    _collect_representations,
    _flatten_trainable_parameters,
)


class ActiveLearning:
    """Manages active learning round by round."""

    def __init__(self, cfg: ExperimentConfig):
        self.final_test_stats = None
        self._no_improve_rounds = 0
        self.last_stats = {"f1_score": 0.0, "loss": float("inf"), "accuracy": 0.0}
        self.cfg = cfg
        self.start_time = 0
        self._initialize()
        self.confusion_matrix = None

    def _initialize(self):
        """
        Initialize all components for active learning.

        Sets up seeds, model, tokenizer, datasets, data pool, training classes
        (optimizer, criterion, scheduler), strategy, sampler, and tracking variables.
        """
        self.set_seeds(self.cfg.seed)

        self._initialize_model_and_tokenizer()

        self._initialize_data()
        self._initialize_pool()

        self._initialize_classes()
        self._initialize_round_tracking()
        self._initialize_timer()

    def _initialize_timer(self):
        """Initializing starting time"""
        self.start_time = time.perf_counter()

    def _initialize_model_and_tokenizer(self):
        """Loads the model and tokenizer from Hugging Face."""
        tqdm.write(f"  [Init] Loading model: {self.cfg.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name_or_path,
            num_labels=self.cfg.num_labels
        )
        initialize_evaluation_state(self.model)
        tqdm.write(f"  [Init] Model loaded  ({self.cfg.num_labels} labels, device={self.cfg.device})")

    def _initialize_data(self):
        """Initialize datasets"""
        tqdm.write(f"  [Init] Loading dataset: {self.cfg.data}")
        self.train_dataset, self.val_dataset, self.test_dataset = self._load_data()
        tqdm.write(
            f"  [Init] Dataset ready  "
            f"train={len(self.train_dataset):,}  "
            f"val={len(self.val_dataset):,}  "
            f"test={len(self.test_dataset):,}"
        )
        logging.info(
            f"Train size: {len(self.train_dataset)}, Validation size: {len(self.val_dataset)}, Test size: {len(self.test_dataset)}")

    def _initialize_pool(self):
        """
        Initialize the data pool with random initial samples.

        Creates a DataPool object with a random subset of training data
        based on cfg.initial_pool_size.
        """

        from collections import defaultdict
        class_to_indices = defaultdict(list)
        for i, label in enumerate(self.train_dataset.labels):
            class_to_indices[int(label)].append(i)
        n_classes = len(class_to_indices)
        per_class = self.cfg.initial_pool_size // n_classes
        initial_indices = []
        for cls_indices in class_to_indices.values():
            initial_indices.extend(random.sample(cls_indices, min(per_class, len(cls_indices))))

        remaining = self.cfg.initial_pool_size - len(initial_indices)
        if remaining > 0:
            leftover = [i for i in range(len(self.train_dataset)) if i not in set(initial_indices)]
            initial_indices.extend(random.sample(leftover, remaining))
        self.pool = DataPool(self.train_dataset, self.val_dataset, self.test_dataset, initial_indices)
        tqdm.write(
            f"  [Init] Pool created   labeled={self.cfg.initial_pool_size}  "
            f"unlabeled={len(self.train_dataset) - self.cfg.initial_pool_size:,}"
        )

    def _initialize_classes(self):
        """
        Resolve and instantiate all training components.

        Resolves class names from config to actual Python classes for:
        optimizer, criterion, scheduler, strategy, and sampler. Then
        instantiates the strategy and sampler with appropriate parameters.
        """
        cfg = self.cfg


        self.optimizer_cls = resolve_class(cfg.optimizer_class, optim)
        self.optimizer_kwargs = cfg.optimizer_kwargs


        self.criterion_cls = resolve_class(cfg.criterion_class, nn)
        self.criterion_kwargs = cfg.criterion_kwargs


        self.scheduler_cls = None
        self.scheduler_kwargs = {}
        if cfg.scheduler_class:
            self.scheduler_cls = resolve_class(cfg.scheduler_class, optim.lr_scheduler)
            self.scheduler_kwargs = cfg.scheduler_kwargs


        self.strategy_cls = resolve_class(cfg.strategy_class, strategies)
        self.strategy_kwargs = dict(cfg.strategy_kwargs)


        if (cfg.strategy_class == "DeltaF1Strategy"
                and "signal" not in self.strategy_kwargs):
            self.strategy_kwargs["signal"] = cfg.switching_signal


        self.sampler_cls = resolve_class(cfg.sampler_class, samplers)
        self.sampler_kwargs = cfg.sampler_kwargs



        self.strategy = self.strategy_cls(
            model=self.model,
            optimizer_cls=self.optimizer_cls,
            optimizer_kwargs=self.optimizer_kwargs,
            criterion_cls=self.criterion_cls,
            criterion_kwargs=self.criterion_kwargs,
            scheduler_cls=self.scheduler_cls,
            scheduler_kwargs=self.scheduler_kwargs,
            device=cfg.device,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            val_dataset=self.val_dataset,
            early_stopping_patience=cfg.early_stopping_patience,
            early_stopping_min_delta=cfg.early_stopping_min_delta,
            **self.strategy_kwargs
        )


        self.sampler = self.sampler_cls(model=self.model, batch_size=cfg.batch_size, device=cfg.device,
                                        **self.sampler_kwargs)

    def _initialize_round_tracking(self):
        """
        Initialize tracking variables for active learning rounds.

        Sets up round_stats list, final_test_stats dict, and resets
        the current_round counter to 0.
        """
        self.round_stats: List[Dict] = []
        self.final_test_stats: Dict = {}

        self.current_round = 0


        self._prev_signal_metrics: Optional[Dict[str, float]] = None

        self._prev_alpha: Optional[float] = None
        self._prev_nc: Optional[float] = None




        self._theta_round1: Optional[torch.Tensor] = None
        self._reps_round1: Optional[torch.Tensor] = None
        self._alpha_round1: Optional[float] = None
        self._nc_round1: Optional[float] = None

    def _load_data(self):
        """
        Load dataset based on configuration.

        Uses eval() to dynamically call the appropriate load function
        (load_agnews, load_imdb, or load_jigsaw) based on cfg.data.

        Returns:
            tuple: (train_dataset, val_dataset, test_dataset) tokenized datasets.
        """
        dataset_name = self.cfg.data


        (train_dataset, val_dataset, test_dataset), (df_train, df_val,
                                                        df_test) = eval(
            f"load_{dataset_name}(path='data', seed={self.cfg.seed}, model_name_or_path='{self.cfg.model_name_or_path}', tokenizer_kwargs={self.cfg.tokenizer_kwargs})")

        return train_dataset, val_dataset, test_dataset

    def set_seeds(self, seed):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train_one_round(self, new_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Train one round of active learning.

        Args:
            new_indices: Indices of newly sampled data (None for first round)

        Returns:
            Dictionary with round statistics
        """
        round_num   = self.current_round + 1
        pool_stats  = self.pool.get_pool_stats()
        labeled_n   = pool_stats.get("labeled_count", 0)
        unlabeled_n = pool_stats.get("unlabeled_count", 0)
        total_n     = labeled_n + unlabeled_n
        pct_labeled = 100 * labeled_n / total_n if total_n > 0 else 0
        new_n       = len(new_indices) if new_indices else 0
        prev_f1     = self.last_stats["f1_score"]

        mode_str = self._current_mode_str()

        tqdm.write(
            f"\n{'─'*60}\n"
            f"  Round {round_num}  |  {mode_str}\n"
            f"  Pool  : {labeled_n:,} labeled ({pct_labeled:.1f}% of {total_n:,})  "
            f"|  {unlabeled_n:,} unlabeled  |  +{new_n} new this round\n"
            f"{'─'*60}"
        )


        training_stats = self.strategy.train(self.pool, new_indices)


        val_stats = evaluate_model(
            self.model,
            self.strategy.criterion,
            self.cfg.batch_size,
            dataset=self.val_dataset,
            device=self.cfg.device,
            include_advanced_metrics=True,
        )


        self._update_last_stats(val_stats)

        delta_f1 = val_stats['f1_score'] - prev_f1
        delta_str = f"{delta_f1:+.4f}" if prev_f1 > 0 else "  —   "
        trend     = "▲" if delta_f1 > 0.001 else ("▼" if delta_f1 < -0.001 else "▶")

        round_stats = {
            **training_stats,
            **val_stats,
            "pool_stats": self.pool.get_pool_stats()
        }

        if self.cfg.log_all_signals:
            round_stats["signals"] = self._compute_round_signals(val_stats)





        self.strategy.update_switching_state(
            val_dataset=self.val_dataset,
            signals=round_stats.get("signals"),
            pool=self.pool,
        )

        self.round_stats.append(round_stats)
        tqdm.write(
            f"  {trend} Val F1: {val_stats['f1_score']:.4f} ({delta_str})  |  "
            f"Acc: {val_stats['accuracy']:.4f}  |  "
            f"Loss: {val_stats['loss']:.4f}  |  "
            f"Time: {training_stats['training_time']:.1f}s  |  "
            f"Samples trained: {training_stats.get('total_samples', '?'):,}"
        )
        tqdm.write(
            "      "
            f"Alpha: {self._format_optional_metric(val_stats.get('spectral_alpha'))}  |  "
            f"NC1: {self._format_optional_metric(val_stats.get('nc1_ratio'))}  |  "
            f"CKA: {self._format_optional_metric(val_stats.get('cka'))}  |  "
            f"dW: {self._format_optional_metric(val_stats.get('l2_weight_distance'))}"
        )

        self.current_round += 1
        return round_stats

    def _update_last_stats(self, val_stats):
        """
        Update internal tracking of last validation statistics.

        Args:
            val_stats (dict): Dictionary containing validation metrics
                             (f1_score, loss, accuracy).
        """
        self.last_stats["f1_score"] =  val_stats["f1_score"]
        self.last_stats["loss"] = val_stats["loss"]
        self.last_stats["accuracy"] = val_stats["accuracy"]

    @staticmethod
    def _format_optional_metric(value) -> str:
        if value is None:
            return "n/a"
        try:
            if not np.isfinite(value):
                return "n/a"
        except TypeError:
            return "n/a"
        return f"{value:.4f}"

    def _compute_round_signals(self, val_stats: Dict[str, Any]) -> Dict[str, float]:
        """
        Build the per-round signals dict logged when cfg.log_all_signals=True.

        Convention: every value follows "value < ε ⇒ stabilized".
          - delta_f1 / delta_accuracy / delta_loss : |metric_t - metric_{t-1}|;
            +inf at round 1 (no previous metrics)
          - gradient_norm                          : ||∇_θ L_val(θ_t)||_2 on the
            full validation set
          - l2_weight_distance, cka                : taken straight from val_stats
            (already populated by evaluate_model with include_advanced_metrics=True)
        """
        f1_t = val_stats["f1_score"]
        acc_t = val_stats["accuracy"]
        loss_t = val_stats["loss"]

        if self._prev_signal_metrics is None:
            delta_f1 = float("inf")
            delta_accuracy = float("inf")
            delta_loss = float("inf")
        else:
            prev = self._prev_signal_metrics
            delta_f1 = abs(f1_t - prev["f1_score"])
            delta_accuracy = abs(acc_t - prev["accuracy"])
            delta_loss = abs(loss_t - prev["loss"])

        self._prev_signal_metrics = {"f1_score": f1_t, "accuracy": acc_t, "loss": loss_t}

        grad_norm = self._compute_gradient_norm(self.val_dataset)


        try:
            alpha_t = calculate_user_spectral_alpha(self.model)
        except Exception:
            alpha_t = float("nan")
        delta_alpha = float("inf")
        if self._prev_alpha is not None and alpha_t == alpha_t:
            delta_alpha = abs(alpha_t - self._prev_alpha)
        if alpha_t == alpha_t:
            self._prev_alpha = alpha_t


        try:
            labeled_subset = self.pool.get_labeled_subset()
            nc_t = calculate_within_class_variance(
                self.model, labeled_subset, self.cfg.batch_size, self.cfg.device,
            )
        except Exception:
            nc_t = float("nan")
        delta_nc = float("inf")
        if self._prev_nc is not None and nc_t == nc_t:
            delta_nc = abs(nc_t - self._prev_nc)
        if nc_t == nc_t:
            self._prev_nc = nc_t









        try:
            flat_theta = _flatten_trainable_parameters(self.model).cpu().float()
            weight_norm = float(torch.norm(flat_theta, p=2).item())
        except Exception:
            flat_theta = None
            weight_norm = float("nan")





        try:
            reps_t, _labels_t = _collect_representations(
                self.model, self.val_dataset, self.cfg.batch_size, self.cfg.device,
            )
            reps_t = reps_t.cpu().float()
        except Exception:
            reps_t = None

        if self._theta_round1 is None and flat_theta is not None:
            self._theta_round1 = flat_theta.clone()
            l2_vs_round1 = 0.0
        elif flat_theta is not None and self._theta_round1 is not None:
            try:
                l2_vs_round1 = float(
                    torch.norm(flat_theta - self._theta_round1, p=2).item()
                )
            except Exception:
                l2_vs_round1 = float("nan")
        else:
            l2_vs_round1 = float("nan")

        if self._reps_round1 is None and reps_t is not None:
            self._reps_round1 = reps_t.clone()
            cka_vs_round1 = 1.0
        elif reps_t is not None and self._reps_round1 is not None:
            try:
                cka_vs_round1 = float(calculate_linear_cka(reps_t, self._reps_round1))
            except Exception:
                cka_vs_round1 = float("nan")
        else:
            cka_vs_round1 = float("nan")

        if self._alpha_round1 is None and (alpha_t == alpha_t):
            self._alpha_round1 = alpha_t
            alpha_vs_round1 = 0.0
        elif self._alpha_round1 is not None and (alpha_t == alpha_t):
            alpha_vs_round1 = abs(alpha_t - self._alpha_round1)
        else:
            alpha_vs_round1 = float("nan")

        if self._nc_round1 is None and (nc_t == nc_t):
            self._nc_round1 = nc_t
            nc_vs_round1 = 0.0
        elif self._nc_round1 is not None and (nc_t == nc_t):
            nc_vs_round1 = abs(nc_t - self._nc_round1)
        else:
            nc_vs_round1 = float("nan")

        return {
            "delta_f1": float(delta_f1),
            "delta_accuracy": float(delta_accuracy),
            "delta_loss": float(delta_loss),
            "gradient_norm": float(grad_norm),
            "l2_weight_distance": float(val_stats.get("l2_weight_distance", float("nan"))),
            "cka": float(val_stats.get("cka", float("nan"))),
            "delta_spectral_alpha": float(delta_alpha),
            "delta_nc": float(delta_nc),

            "_raw_spectral_alpha": float(alpha_t),
            "_raw_nc": float(nc_t),

            "weight_norm": float(weight_norm),
            "l2_vs_round1": float(l2_vs_round1),
            "cka_vs_round1": float(cka_vs_round1),
            "alpha_vs_round1": float(alpha_vs_round1),
            "nc_vs_round1": float(nc_vs_round1),
        }

    def _compute_gradient_norm(self, val_dataset) -> float:
        """
        L2 norm of model parameter gradients accumulated over val_dataset.

        Computes ||∇_θ Σ_b L_val_b(θ)||_2. We backward per batch and let .grad
        accumulate (gradients are additive), instead of summing losses across
        all batches into a single graph and backwarding once — the latter keeps
        every batch's activations alive and OOMs on a 5k-example val set.
        """
        from torch.utils.data import DataLoader as _DL
        was_training = self.model.training
        self.model.train()
        self.strategy.optimizer.zero_grad()

        loader = _DL(val_dataset, batch_size=self.cfg.batch_size, shuffle=False)

        for inputs, targets in loader:
            inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
            targets = targets.to(self.cfg.device)
            outputs = self.model(**inputs)
            loss = self.strategy.criterion(outputs.logits, targets)
            loss.backward()

        norm = sum(
            p.grad.norm().item() ** 2
            for p in self.model.parameters() if p.grad is not None
        ) ** 0.5

        self.strategy.optimizer.zero_grad()
        self.model.train(was_training)
        return norm

    def sample_next_batch(self) -> List[int]:
        """
        Sample next batch of data to label.

        Returns:
            List of selected indices
        """
        if not self.pool.get_unlabeled_indices():
            logging.warning("No unlabeled data remaining!")
            return []

        sampler_name = type(self.sampler).__name__
        pool_stats   = self.pool.get_pool_stats()
        unlabeled_n  = pool_stats.get("unlabeled_count", 0)
        tqdm.write(
            f"  [Sample] {sampler_name} selecting {self.cfg.acquisition_batch_size} "
            f"from {unlabeled_n:,} unlabeled samples..."
        )

        start = time.perf_counter()

        valid_params = inspect.signature(self.sampler.select).parameters
        sampler_kwargs = {k: v for k, v in self.strategy.pass_args_to_sampler().items() if k in valid_params}
        selected_indices = self.sampler.select(self.pool, self.cfg.acquisition_batch_size, **sampler_kwargs)

        elapsed = round(time.perf_counter() - start, 2)
        tqdm.write(f"  [Sample] {len(selected_indices)} samples selected in {elapsed}s")
        return selected_indices

    def calculate_total_rounds(self):
        """
        Calculate the total number of rounds to run.

        Returns the configured total_rounds if set, otherwise calculates
        based on pool proportion threshold.

        Returns:
            int: Number of rounds to execute, or -1 for unlimited.
        """
        if self.cfg.total_rounds != -1:
            return self.cfg.total_rounds

        return self.calculate_total_rounds_pool_proportion()


    def calculate_total_rounds_pool_proportion(self):
        """
        Calculate rounds based on pool proportion threshold.

        Determines how many rounds are needed to reach the target proportion
        of the training pool, or uses the full pool if no threshold is set.

        Returns:
            int: Number of rounds needed to reach the target pool proportion.
        """
        if self.cfg.pool_proportion_threshold != -1:
            return int(len(self.pool.train_dataset)*self.cfg.pool_proportion_threshold/self.cfg.acquisition_batch_size)

        return int(len(self.pool.train_dataset)/self.cfg.acquisition_batch_size)

    def run_full_pipeline(self):
        """
        Execute the complete active learning pipeline.

        Runs the full cycle of training rounds with active sampling until
        one of the stopping conditions is met: reaching total_rounds,
        exhausting unlabeled data, detecting plateau, or timing out.
        Performs final evaluation on the test set.

        Returns:
            dict: Final test set metrics (loss, f1_score, accuracy).
        """
        self._initialize()
        total_rounds = self.calculate_total_rounds()
        unlimited = (total_rounds == -1)
        if unlimited:
            total_rounds = float('inf')

        tqdm.write(
            f"\n{'='*60}\n"
            f"  Experiment: {self.cfg.experiment_name}\n"
            f"  Dataset:  {self.cfg.data:<20}  Seed: {self.cfg.seed}\n"
            f"  Strategy: {self.cfg.strategy_class:<20}  Sampler: {self.cfg.sampler_class}\n"
            f"  Model:    {self.cfg.model_name_or_path}\n"
            f"  Rounds:   {'unlimited' if unlimited else total_rounds}  "
            f"Batch: {self.cfg.acquisition_batch_size}  Epochs/round: {self.cfg.epochs}\n"
            f"{'='*60}"
        )

        round_bar = tqdm(
            total=None if unlimited else int(total_rounds),
            desc="  Rounds",
            unit="round",
            dynamic_ncols=True,
            position=0,
        )

        with logging_redirect_tqdm():
            new_indices = []
            if total_rounds > 0:
                self.train_one_round(None)
                round_bar.update(1)
                round_bar.set_postfix(**self._round_postfix())
                new_indices = self.sample_next_batch()

            while new_indices and self.current_round < total_rounds and not self.has_plateaued() and not self._timedout():
                self.train_one_round(new_indices)
                round_bar.update(1)
                round_bar.set_postfix(**self._round_postfix())
                new_indices = self.sample_next_batch()

        round_bar.close()

        if self.current_round != total_rounds:
            tqdm.write(f"\n  Stopped at round {self.current_round} (early stopping / timeout / no unlabeled data)")

        tqdm.write(f"\n  [Eval] Running final evaluation on test set...")
        metrics = evaluate_model(
            self.model,
            self.strategy.criterion,
            self.cfg.batch_size,
            dataset=self.test_dataset,
            device=self.cfg.device,
            include_advanced_metrics=True,
        )
        self.final_test_stats = metrics
        self.confusion_matrix = compute_confusion_matrix(self.model, self.test_dataset, self.cfg.batch_size)
        tqdm.write(
            f"  [Eval] Test F1: {metrics['f1_score']:.4f}  |  "
            f"Test Acc: {metrics['accuracy']:.4f}  |  "
            f"Test Loss: {metrics['loss']:.4f}\n"
        )
        tqdm.write(
            "         "
            f"Alpha: {self._format_optional_metric(metrics.get('spectral_alpha'))}  |  "
            f"NC1: {self._format_optional_metric(metrics.get('nc1_ratio'))}  |  "
            f"CKA: {self._format_optional_metric(metrics.get('cka'))}  |  "
            f"dW: {self._format_optional_metric(metrics.get('l2_weight_distance'))}\n"
        )
        return metrics

    def _current_mode_str(self) -> str:
        """Return a human-readable description of the current training mode."""
        s = self.strategy
        strategy_name = type(s).__name__
        if strategy_name == "DeltaF1Strategy":
            if s.switched:
                return f"HybridAL  [FineTune — switched at round {s.switch_round}]"
            signal_label = {"delta_f1": "ΔF1", "delta_loss": "ΔLoss",
                            "delta_accuracy": "ΔAcc", "gradient_norm": "GradNorm"}.get(s.signal, s.signal)
            return f"HybridAL  [Retrain, monitoring {signal_label}, count={s.count}/{s.k}]"
        if strategy_name == "FixedSwitchStrategy":
            mode = "FineTune" if s.switched else "Retrain"
            return f"FixedSwitch-r{s.switch_round}  [{mode}]"
        return strategy_name

    def _round_postfix(self) -> dict:
        """Return tqdm postfix dict for the round bar."""
        last = self.round_stats[-1] if self.round_stats else {}
        return {
            "F1": f"{last.get('f1_score', 0):.4f}",
            "loss": f"{last.get('loss', 0):.4f}",
            "labeled": self.pool.get_pool_stats().get("labeled_count", 0),
        }

    def has_plateaued(self):
        """
        Check if model performance has plateaued.

        Returns:
            bool: True if plateau detected, False otherwise.
        """
        return self._has_plateaued_f1_based()

    def _has_plateaued_f1_based(self):
        """
        Check for plateau based on F1 score improvement.

        Tracks consecutive rounds with insufficient F1 improvement against
        the configured plateau_f1_threshold and plateau_patience.

        Returns:
            bool: True if F1 improvements have plateaued for patience rounds.
        """
        if self.current_round < self.cfg.min_rounds_before_plateau or self.cfg.plateau_patience == -1:
            return False
        current_f1 = self.round_stats[-1]["f1_score"]
        last_f1 = self.last_stats["f1_score"]
        if current_f1 - last_f1 < self.cfg.plateau_f1_threshold:
            self._no_improve_rounds += 1
        else:
            self._no_improve_rounds = 0
        if self._no_improve_rounds >= self.cfg.plateau_patience:
            logging.info("\n--- Plateau patience reached. Finishing up.")
            return True
        return False

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all rounds."""
        strategy_metadata = {}
        if hasattr(self.strategy, 'switched'):
            strategy_metadata['switched'] = self.strategy.switched
        if hasattr(self.strategy, 'switch_round'):
            strategy_metadata['switch_round'] = self.strategy.switch_round

        return {
            "cfg": self.cfg.__dict__,
            "total_rounds": len(self.round_stats),
            "round_val_stats": self.round_stats,
            "strategy_metadata": strategy_metadata,
            "final_pool_stats": self.pool.get_pool_stats(),
            "final_test_stats": self.final_test_stats,
            "confusion_matrix": self.confusion_matrix
        }

    def _timedout(self):
        """
        Check if the experiment has exceeded its time limit.

        Returns:
            bool: True if max_seconds limit exceeded, False otherwise.
        """
        limit = getattr(self.cfg, "max_seconds", None)
        if limit is None or limit < 0:
            return False
        timed_out = (time.perf_counter() - self.start_time) >= limit

        if timed_out:
            logging.info("\n--- Timed out. Finishing up.")
        return timed_out

    def save_experiment(self, filepath: Optional[Path] = None, timestamp: bool = True) -> None:
        """Save experiment results to a JSON file."""
        summary = self.get_experiment_summary()


        save_dir = self.cfg.save_dir / self.cfg.experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)

        if filepath is None:
            name = f"results"
            if timestamp:
                name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = save_dir / f"{name}.json"


        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logging.info(f"Experiment saved to {filepath}")


def resolve_class(name: str, module) -> Any:
    """
    Resolve a class name (string) to a Python class.

    Args:
        name: Name of the class (string)
        module: Module to look in (e.g., torch.nn, torch.optim)
    Returns:
        Python class object
    """
    if hasattr(module, name):
        return getattr(module, name)

    raise ValueError(f"Cannot resolve class '{name}' in module {module}")
