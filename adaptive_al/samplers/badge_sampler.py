"""
BADGE (Batch Active learning by Diverse Gradient Embeddings) sampler.

Implements the acquisition function from Ash et al. (2019) "Deep Batch Active Learning
by Diverse, Uncertain Gradient Lower Bounds". Selects a diverse batch using gradient
embeddings derived from the penultimate layer, then applies k-means++ initialization
to pick the most spread-out subset.
"""

import logging
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base_sampler import BaseSampler
from ..pool import DataPool


def _get_penultimate_module(model):
    """
    Return the penultimate linear layer for supported HuggingFace architectures.

    Works for DistilBERT, BERT, and RoBERTa. The penultimate module is the one
    whose output is the direct input to the final classification projection.

    Args:
        model: A HuggingFace AutoModelForSequenceClassification instance.

    Returns:
        nn.Module: The penultimate linear layer.

    Raises:
        NotImplementedError: If the architecture is not recognized.
    """
    if hasattr(model, 'pre_classifier'):
        # DistilBERT: pre_classifier → classifier
        return model.pre_classifier
    elif hasattr(model, 'classifier') and hasattr(model.classifier, 'dense'):
        # RoBERTa: classifier.dense → dropout → out_proj
        return model.classifier.dense
    elif hasattr(model, 'bert') and hasattr(model.bert, 'pooler'):
        # BERT: bert.pooler.dense → classifier
        return model.bert.pooler.dense
    raise NotImplementedError(
        f"BADGE penultimate module detection not supported for {type(model).__name__}. "
        "Supported: DistilBERT, BERT, RoBERTa."
    )


def _compute_gradient_embeddings(model, dataloader, device):
    """
    Compute gradient embeddings for all batches in the dataloader.

    For each sample, the gradient embedding is computed analytically as:
        g = h * (p_hat - 1)
    where h is the penultimate layer output and p_hat is the softmax probability
    of the predicted class. This is the gradient of the cross-entropy loss w.r.t.
    the penultimate representation under the hypothetical label (argmax prediction).

    Args:
        model: The classification model.
        dataloader: DataLoader over unlabeled samples.
        device (str): Device for computation.

    Returns:
        np.ndarray: Gradient embeddings of shape (N, hidden_dim).
    """
    model.eval()
    model.to(device)

    penultimate_module = _get_penultimate_module(model)
    all_embeddings = []

    for batch in dataloader:
        inputs, _ = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}

        captured = []

        def hook_fn(module, input, output):
            captured.append(output.detach())

        handle = penultimate_module.register_forward_hook(hook_fn)

        with torch.no_grad():
            outputs = model(**inputs)

        handle.remove()

        logits = outputs.logits                                    # (B, C)
        probs = F.softmax(logits, dim=-1)                          # (B, C)
        pred_classes = probs.argmax(dim=-1)                        # (B,)

        h = captured[0]                                            # (B, hidden)
        # Analytic gradient: scale = p(predicted_class) - 1
        scale = probs[range(len(pred_classes)), pred_classes] - 1.0  # (B,)
        g = h * scale.unsqueeze(-1)                                # (B, hidden)

        all_embeddings.append(g.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()                # (N, hidden)


def _kmeans_plusplus_init(embeddings, k, rng):
    """
    Select k diverse centers using k-means++ initialization.

    Greedily selects centers such that each subsequent center is chosen
    with probability proportional to its squared distance from the nearest
    already-chosen center.

    Args:
        embeddings (np.ndarray): Array of shape (N, D).
        k (int): Number of centers to select.
        rng (np.random.RandomState): Random state for reproducibility.

    Returns:
        List[int]: Indices of the k selected centers.
    """
    n = len(embeddings)
    first = rng.randint(0, n)
    centers = [first]

    for _ in range(k - 1):
        # Distance from each point to the nearest current center
        center_embeds = embeddings[centers]                        # (c, D)
        diff = embeddings[:, None, :] - center_embeds[None, :, :] # (N, c, D)
        sq_dists = (diff ** 2).sum(axis=-1)                        # (N, c)
        min_sq_dists = sq_dists.min(axis=1)                        # (N,)

        # Already-selected centers have distance 0
        min_sq_dists[centers] = 0.0

        total = min_sq_dists.sum()
        if total == 0.0:
            # All remaining points coincide with a center — fall back to random
            remaining = [i for i in range(n) if i not in centers]
            if not remaining:
                break
            centers.append(rng.choice(remaining))
        else:
            probs = min_sq_dists / total
            centers.append(int(rng.choice(n, p=probs)))

    return centers


class BADGESampler(BaseSampler):
    """
    BADGE acquisition function: selects a diverse batch using gradient embeddings.

    Computes a gradient embedding for each unlabeled sample by propagating the
    predicted label back through the penultimate layer (analytically). Applies
    k-means++ initialization on the embedding space to select a batch that is
    both uncertain (high gradient magnitude) and diverse (spread across the space).

    Attributes:
        random_subset_size (int): Pre-filter the unlabeled pool to this size before
            computing embeddings. Keeps BADGE tractable on large pools. Set to -1
            to use the entire unlabeled pool (may be slow for large datasets).
    """

    def __init__(self, random_subset_size: int = 1000, **kwargs):
        """
        Initialize the BADGE sampler.

        Args:
            random_subset_size (int): Size of the random pre-filter subset.
                                     Set to -1 to disable pre-filtering.
            **kwargs: Arguments passed to BaseSampler (model, batch_size, device, seed).
        """
        super().__init__(**kwargs)
        self.random_subset_size = random_subset_size
        self._rng = np.random.RandomState(self.seed)

    def select(self, pool: DataPool, num_samples: int,
               random_indices_fraction: float = 0) -> List[int]:
        """
        Select a diverse batch from the unlabeled pool using BADGE.

        Args:
            pool (DataPool): Data pool with labeled/unlabeled splits.
            num_samples (int): Number of samples to select.
            random_indices_fraction (float): Accepted for API compatibility with
                DeltaF1Strategy.pass_args_to_sampler() but ignored by BADGE
                (diversity is inherent in the gradient embedding space).

        Returns:
            List[int]: Indices of selected samples.
        """
        unlabeled_indices = pool.get_unlabeled_indices()

        if len(unlabeled_indices) <= num_samples:
            return unlabeled_indices

        # Optionally pre-filter to a random subset for efficiency on large pools
        if self.random_subset_size > 0 and len(unlabeled_indices) > self.random_subset_size:
            candidate_indices = random.sample(unlabeled_indices, self.random_subset_size)
        else:
            candidate_indices = unlabeled_indices

        unlabeled_subset = pool.get_subset(candidate_indices)
        dataloader = DataLoader(unlabeled_subset, batch_size=self.batch_size, shuffle=False)

        logging.info(f"BADGE: computing gradient embeddings for {len(candidate_indices)} candidates...")
        embeddings = _compute_gradient_embeddings(self.model, dataloader, self.device)

        selected_local = _kmeans_plusplus_init(embeddings, num_samples, self._rng)
        return [candidate_indices[i] for i in selected_local]
