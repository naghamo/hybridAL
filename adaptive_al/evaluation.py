"""
Model evaluation utilities for active learning experiments.

This module provides functions for evaluating PyTorch models, including
full dataset evaluation, approximate subset-based evaluation for efficiency,
variance analysis across multiple evaluations, and confusion matrix computation.
"""

import logging
from typing import Optional, Dict, List, Tuple
import random

import torch
import numpy as np
import time
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Subset

try:
    import weightwatcher as ww
except ImportError:  # pragma: no cover - depends on runtime environment
    ww = None


_EVAL_STATE_ATTR = "_adaptive_al_eval_state"


def initialize_evaluation_state(model: torch.nn.Module) -> None:
    """
    Cache model state needed by advanced evaluation metrics.

    Tracks the previous round's trainable-parameter snapshot and the previous
    round's penultimate representations so CKA and L2 weight distance can be
    computed against round t-1. Both start as None; they are populated by
    calculate_additional_metrics(... update_state=True) at the end of each round.
    """
    if hasattr(model, _EVAL_STATE_ATTR):
        return

    setattr(model, _EVAL_STATE_ATTR, {
        "previous_weights": None,
        "previous_representations": None,
    })


def _get_evaluation_state(model: torch.nn.Module) -> Dict[str, Optional[torch.Tensor]]:
    initialize_evaluation_state(model)
    return getattr(model, _EVAL_STATE_ATTR)


def _flatten_trainable_parameters(model: torch.nn.Module) -> torch.Tensor:
    params = [
        parameter.detach().reshape(-1)
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not params:
        return torch.empty(0)
    return torch.cat(params)


def _select_metric_dataset(
        dataset: torch.utils.data.Dataset,
        max_samples: Optional[int]
) -> torch.utils.data.Dataset:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, list(range(max_samples)))


def _run_model_forward(
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        collect_hidden_states: bool = False
):
    if collect_hidden_states:
        try:
            return model(**inputs, output_hidden_states=True, return_dict=True)
        except TypeError:
            pass
    return model(**inputs)


def _extract_penultimate_representations(outputs, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        last_hidden = hidden_states[-1]
        if last_hidden.ndim == 3:
            return last_hidden[:, 0, :]
        return last_hidden.reshape(last_hidden.shape[0], -1)

    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    return logits.reshape(logits.shape[0], -1)


def _collect_representations(
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_representations, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
            outputs = _run_model_forward(model, inputs, collect_hidden_states=True)
            representations = _extract_penultimate_representations(outputs, inputs)

            all_representations.append(representations.detach().cpu())
            all_labels.append(targets.detach().cpu())

    return torch.cat(all_representations), torch.cat(all_labels)


def calculate_spectral_alpha(
        model: torch.nn.Module,
        max_layers: int = 6,
        tail_fraction: float = 0.2
) -> float:
    """
    Compute spectral alpha using WeightWatcher when available.

    Falls back to a local power-law tail approximation if WeightWatcher is not
    installed in the active Python environment.
    """
    if ww is not None:
        try:
            watcher = ww.WeightWatcher(model=model, framework="pytorch")
            details = watcher.analyze(
                plot=False,
                pool=True,
                randomize=False,
                mp_fit=False,
                savefig=None,
            )
            if "alpha" in details.columns:
                alpha_values = details["alpha"].replace([np.inf, -np.inf], np.nan).dropna()
                if not alpha_values.empty:
                    return float(alpha_values.tail(max_layers).mean())
        except Exception as exc:  # pragma: no cover - defensive fallback
            logging.warning("WeightWatcher spectral alpha failed, falling back to local approximation: %s", exc)

    candidate_matrices = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.ndim < 2 or "weight" not in name:
            continue
        candidate_matrices.append(parameter.detach().float().reshape(parameter.shape[0], -1).cpu())

    if not candidate_matrices:
        return float("nan")

    alphas = []
    for weight_matrix in candidate_matrices[-max_layers:]:
        singular_values = torch.linalg.svdvals(weight_matrix)
        eigenvalues = torch.square(singular_values).numpy()
        eigenvalues = eigenvalues[np.isfinite(eigenvalues) & (eigenvalues > 0)]
        if eigenvalues.size < 10:
            continue

        xmin = float(np.quantile(eigenvalues, max(0.0, 1.0 - tail_fraction)))
        tail = eigenvalues[eigenvalues >= xmin]
        if tail.size < 5:
            continue

        logs = np.log(np.maximum(tail / max(xmin, 1e-12), 1.0 + 1e-12))
        denom = float(np.sum(logs))
        if denom <= 0:
            continue

        alphas.append(1.0 + tail.size / denom)

    return float(np.mean(alphas)) if alphas else float("nan")


def calculate_nc1_ratio(
        representations: torch.Tensor,
        labels: torch.Tensor
) -> float:
    """
    Compute the neural-collapse NC1 ratio Tr(Sw Sb^-1).
    """
    if representations.numel() == 0 or labels.numel() == 0:
        return float("nan")

    features = representations.float()
    labels = labels.long()
    unique_labels = torch.unique(labels)
    if unique_labels.numel() < 2:
        return float("nan")

    global_mean = features.mean(dim=0)
    feature_dim = features.shape[1]
    sigma_w = torch.zeros((feature_dim, feature_dim), dtype=features.dtype)
    sigma_b = torch.zeros((feature_dim, feature_dim), dtype=features.dtype)

    valid_classes = 0
    for label in unique_labels:
        class_features = features[labels == label]
        if class_features.shape[0] < 2:
            continue
        class_mean = class_features.mean(dim=0)
        centered = class_features - class_mean
        sigma_w += centered.T @ centered / class_features.shape[0]
        mean_diff = (class_mean - global_mean).unsqueeze(1)
        sigma_b += mean_diff @ mean_diff.T
        valid_classes += 1

    if valid_classes < 2:
        return float("nan")

    sigma_w /= valid_classes
    sigma_b /= valid_classes

    sigma_b_pinv = torch.linalg.pinv(sigma_b)
    nc1_ratio = torch.trace(sigma_w @ sigma_b_pinv)
    return float(nc1_ratio.item())


def calculate_linear_cka(
        representations: torch.Tensor,
        previous_representations: Optional[torch.Tensor]
) -> float:
    """
    Compute linear CKA between current and previous representations.
    """
    if previous_representations is None:
        return float("nan")
    if representations.shape != previous_representations.shape:
        return float("nan")

    x = representations.float()
    y = previous_representations.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    x_gram = x @ x.T
    y_gram = y @ y.T

    n = x_gram.shape[0]
    if n < 2:
        return float("nan")

    h = torch.eye(n) - torch.full((n, n), 1.0 / n)
    x_centered = h @ x_gram @ h
    y_centered = h @ y_gram @ h

    hsic_xy = torch.sum(x_centered * y_centered)
    hsic_xx = torch.sum(x_centered * x_centered)
    hsic_yy = torch.sum(y_centered * y_centered)
    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom <= 0:
        return float("nan")

    return float((hsic_xy / denom).item())


def calculate_user_spectral_alpha(model: torch.nn.Module) -> float:
    """
    Per-Linear-layer power-law exponent of the singular-value spectrum, averaged.

    For each nn.Linear weight matrix W:
      - svals = torch.linalg.svdvals(W)  (descending after sort)
      - fit log(rank) -> log(svals) via linear regression
      - layer_alpha = -slope
    Return mean across layers. Returns nan if no layer yields a usable spectrum.

    Used as the raw value behind the "delta_spectral_alpha" switching signal,
    where the strategy compares |alpha_t - alpha_{t-1}|.
    """
    from scipy.stats import linregress

    alphas = []
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            W = module.weight.detach()
            try:
                sv = torch.linalg.svdvals(W).cpu().numpy()
            except Exception:
                continue
            sv = sv[sv > 1e-8]
            if sv.size < 3:
                continue
            sv = np.sort(sv)[::-1]
            ranks = np.arange(1, sv.size + 1)
            log_rank = np.log(ranks)
            log_sv = np.log(sv)
            try:
                slope = linregress(log_rank, log_sv).slope
            except Exception:
                continue
            if not np.isfinite(slope):
                continue
            alphas.append(-float(slope))
    if not alphas:
        return float("nan")
    return float(np.mean(alphas))


def calculate_within_class_variance(
        model: torch.nn.Module,
        labeled_dataset: torch.utils.data.Dataset,
        batch_size: int,
        device: str = "cuda",
) -> float:
    """
    Within-class variance of penultimate ([CLS]) representations on labeled data.

    For each class c with n_c samples and per-sample reps h_i:
      mu_c = (1/n_c) Σ h_i
      var_c = (1/n_c) Σ ||h_i - mu_c||^2
    Return mean of var_c across classes (skipping classes with < 2 samples).
    Used as the raw value behind the "delta_nc" switching signal.
    """
    representations, labels = _collect_representations(
        model, labeled_dataset, batch_size, device,
    )
    if representations.numel() == 0:
        return float("nan")

    H = representations.float()
    L = labels.long()
    classes = torch.unique(L)
    var_per_class = []
    for c in classes:
        mask = L == c
        if int(mask.sum().item()) < 2:
            continue
        Hc = H[mask]
        mu = Hc.mean(dim=0, keepdim=True)
        mse = ((Hc - mu) ** 2).sum(dim=1).mean().item()
        var_per_class.append(float(mse))
    if not var_per_class:
        return float("nan")
    return float(np.mean(var_per_class))


def calculate_l2_weight_distance(model: torch.nn.Module) -> float:
    """
    Compute ||θ_t - θ_{t-1}||_2: the L2 distance between current trainable
    weights and the previous round's snapshot. Returns +inf at round 1 (no
    previous snapshot yet); the snapshot is updated by calculate_additional_metrics
    when update_state=True.
    """
    eval_state = _get_evaluation_state(model)
    previous_weights = eval_state.get("previous_weights")
    if previous_weights is None:
        return float("inf")

    current_weights = _flatten_trainable_parameters(model).detach().cpu()
    if current_weights.shape != previous_weights.shape:
        return float("nan")

    return float(torch.norm(current_weights - previous_weights, p=2).item())


def calculate_additional_metrics(
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        device: str = "cuda",
        max_samples: int = 512,
        update_state: bool = True,
) -> Dict[str, float]:
    """
    Compute advanced structural metrics on a deterministic subset of the dataset.
    """
    eval_state = _get_evaluation_state(model)
    metric_dataset = _select_metric_dataset(dataset, max_samples)
    representations, labels = _collect_representations(model, metric_dataset, batch_size, device)

    metrics = {
        "spectral_alpha": calculate_spectral_alpha(model),
        "nc1_ratio": calculate_nc1_ratio(representations, labels),
        "cka": calculate_linear_cka(representations, eval_state["previous_representations"]),
        "l2_weight_distance": calculate_l2_weight_distance(model),
    }

    if update_state:
        eval_state["previous_representations"] = representations.clone()
        eval_state["previous_weights"] = _flatten_trainable_parameters(model).detach().cpu()
    return metrics


def _format_metric_value(metric_value: float) -> str:
    if metric_value is None or not np.isfinite(metric_value):
        return "n/a"
    return f"{metric_value:.4f}"


def _evaluate_model_core(
        model: torch.nn.Module,
        criterion,
        batch_size: int,
        dataset: torch.utils.data.Dataset,
        device: str = "cuda",
        subset_size: Optional[int] = None,
        random_seed: Optional[int] = None,
        include_advanced_metrics: bool = False,
        advanced_metric_subset_size: int = 512,
        update_advanced_metric_state: bool = True,
) -> Dict[str, float]:
    """
    Core evaluation function that handles both full and subset evaluation.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion: Loss function.
        batch_size (int): Batch size for evaluation.
        dataset (torch.utils.data.Dataset): Dataset to evaluate on.
        device (str): Device for evaluation.
        subset_size (Optional[int]): If provided, evaluate on random subset of this size.
        random_seed (Optional[int]): Seed for reproducible subset selection.

    Returns:
        Dict[str, float]: Evaluation metrics.
    """
    start = time.perf_counter()

    # Create subset if requested
    eval_dataset = dataset
    if subset_size is not None:
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)

        indices = random.sample(range(len(dataset)), min(subset_size, len(dataset)))
        eval_dataset = Subset(dataset, indices)

    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
            targets = targets.to(device)

            outputs = model(**inputs)
            logits = outputs['logits']
            loss = criterion(logits, targets)
            total_loss += loss.item()

            # Expecting multi-class (not binary)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(targets.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = {
        "loss": total_loss / len(loader),
        "f1_score": f1_score(all_labels, all_preds, average="macro"),
        "accuracy": accuracy_score(all_labels, all_preds)
    }

    if include_advanced_metrics:
        metrics.update(
            calculate_additional_metrics(
                model=model,
                dataset=eval_dataset,
                batch_size=batch_size,
                device=device,
                max_samples=advanced_metric_subset_size,
                update_state=update_advanced_metric_state,
            )
        )

    subset_info = f" on subset of {len(eval_dataset)} samples" if subset_size else ""
    metric_suffix = ""
    if include_advanced_metrics:
        metric_suffix = (
            f" | alpha={_format_metric_value(metrics['spectral_alpha'])}"
            f" nc1={_format_metric_value(metrics['nc1_ratio'])}"
            f" cka={_format_metric_value(metrics['cka'])}"
            f" dW={_format_metric_value(metrics['l2_weight_distance'])}"
        )
    logging.info(
        f"Model evaluation{subset_info} took {time.perf_counter() - start:.2f} seconds"
        f" | f1={metrics['f1_score']:.4f} acc={metrics['accuracy']:.4f} loss={metrics['loss']:.4f}"
        f"{metric_suffix}"
    )
    return metrics


def evaluate_model(
        model: torch.nn.Module,
        criterion,
        batch_size: int,
        dataset: Optional[torch.utils.data.Dataset] = None,
        device: str = "cuda",
        include_advanced_metrics: bool = False,
        advanced_metric_subset_size: int = 512,
        update_advanced_metric_state: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a PyTorch model on a given dataset.

    This function computes the average loss, macro F1-score, and accuracy
    of the model on the provided dataset.

    Args:
        model (torch.nn.Module):
            The trained (or partially trained) model to evaluate.
        criterion (torch.nn.Module or callable):
            Loss function used to compute evaluation loss (e.g., CrossEntropyLoss).
        batch_size (int):
            Batch size to use when evaluating.
        dataset (torch.utils.data.Dataset, optional):
            Dataset to evaluate the model on. Must return (inputs, targets) where
            inputs is a dictionary of tensors and targets is a tensor of labels.
        device: Device on which to perform evaluation.

    Returns:
        Dict[str, float]: A dictionary containing:
            - "loss": Average loss over the dataset.
            - "f1_score": Macro-averaged F1 score across all classes.
            - "accuracy": Overall classification accuracy.
    """
    if dataset is None:
        raise ValueError("Dataset must be provided for evaluation")

    return _evaluate_model_core(
        model,
        criterion,
        batch_size,
        dataset,
        device,
        include_advanced_metrics=include_advanced_metrics,
        advanced_metric_subset_size=advanced_metric_subset_size,
        update_advanced_metric_state=update_advanced_metric_state,
    )


def approximate_evaluate_model(
        model: torch.nn.Module,
        criterion,
        batch_size: int,
        dataset: torch.utils.data.Dataset,
        subset_size: int,
        device: str = "cuda",
        random_seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate a PyTorch model on a random subset of the dataset for faster evaluation.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion: Loss function.
        batch_size (int): Batch size for evaluation.
        dataset (torch.utils.data.Dataset): Full dataset to sample from.
        subset_size (int): Size of random subset to evaluate on.
        device (str): Device for evaluation.
        random_seed (Optional[int]): Seed for reproducible results.

    Returns:
        Dict[str, float]: Evaluation metrics on the subset.
    """
    return _evaluate_model_core(
        model, criterion, batch_size, dataset, device,
        subset_size=subset_size, random_seed=random_seed
    )


def approximate_evaluate_variance(
        model: torch.nn.Module,
        criterion,
        batch_size: int,
        dataset: torch.utils.data.Dataset,
        subset_size: int,
        num_evaluations: int = 5,
        device: str = "cuda",
        base_seed: Optional[int] = None
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Perform multiple approximate evaluations and compute variance in metrics.

    This function runs approximate evaluation multiple times with different random
    subsets and returns both individual results and variance statistics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion: Loss function.
        batch_size (int): Batch size for evaluation.
        dataset (torch.utils.data.Dataset): Full dataset to sample from.
        subset_size (int): Size of random subset for each evaluation.
        num_evaluations (int): Number of evaluations to perform.
        device (str): Device for evaluation.
        base_seed (Optional[int]): Base seed for reproducible results.

    Returns:
        Tuple[List[Dict[str, float]], Dict[str, float]]:
            - List of individual evaluation results
            - Dictionary with mean and std for each metric
    """
    results = []

    for i in range(num_evaluations):
        seed = base_seed + i if base_seed is not None else None
        result = approximate_evaluate_model(
            model, criterion, batch_size, dataset, subset_size, device, seed
        )
        results.append(result)

    # Compute variance statistics
    metrics_arrays = {}
    for metric in results[0].keys():
        metrics_arrays[metric] = np.array([result[metric] for result in results])

    variance_stats = {}
    for metric, values in metrics_arrays.items():
        variance_stats[f"{metric}_mean"] = float(np.mean(values))
        variance_stats[f"{metric}_std"] = float(np.std(values))
        variance_stats[f"{metric}_var"] = float(np.var(values))

    logging.info(f"Completed {num_evaluations} approximate evaluations with subset size {subset_size}")
    logging.info("Variance statistics:")
    for metric in ['loss', 'f1_score', 'accuracy']:
        mean_val = variance_stats[f"{metric}_mean"]
        std_val = variance_stats[f"{metric}_std"]
        logging.info(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")

    return results, variance_stats



def compute_confusion_matrix(
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        device: str = "cuda",
        normalize: bool = False,
        class_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Compute and optionally plot the confusion matrix for a model.

    Args:
        model (torch.nn.Module): Trained model.
        dataset (torch.utils.data.Dataset): Dataset to evaluate on.
        batch_size (int): Batch size for DataLoader.
        device (str): Device for evaluation.
        normalize (bool): Whether to normalize counts per class.
        class_names (Optional[List[str]]): List of class names for plotting.

    Returns:
        np.ndarray: Confusion matrix (optionally normalized).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
            targets = targets.to(device)

            outputs = model(**inputs)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(targets.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    cm = confusion_matrix(all_labels, all_preds, normalize="true" if normalize else None).tolist()

    return cm
