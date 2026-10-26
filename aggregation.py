"""
aggregation.py – Ensemble prediction aggregation strategies.
Supports hard majority voting, soft voting (mean probs), and weighted soft voting.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional
from sklearn.metrics import accuracy_score, f1_score


def hard_majority_vote(
    predictions: List[np.ndarray],  # list of (N,) int arrays per model
) -> np.ndarray:
    """
    Hard majority voting: each model votes for its argmax class.
    Ties broken by lowest class index.

    Args:
        predictions: list of (N,) integer arrays of class predictions.

    Returns:
        (N,) array of final predicted labels.
    """
    stacked = np.stack(predictions, axis=1)  # (N, M)
    n_samples = stacked.shape[0]
    final = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        counts = np.bincount(stacked[i], minlength=4)
        final[i] = counts.argmax()
    return final


def soft_vote(
    probabilities: List[np.ndarray],  # list of (N, C) arrays
) -> np.ndarray:
    """
    Soft voting: average class probabilities across models.

    Args:
        probabilities: list of (N, num_classes) softmax probability arrays.

    Returns:
        (N,) array of predicted labels.
    """
    avg_probs = np.mean(np.stack(probabilities, axis=0), axis=0)  # (N, C)
    return avg_probs.argmax(axis=1)


def weighted_soft_vote(
    probabilities: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """
    Weighted soft voting: weighted average of class probabilities.

    Args:
        probabilities: list of (N, num_classes) arrays.
        weights: per-model weights (e.g., validation F1 scores).

    Returns:
        (N,) array of predicted labels.
    """
    weights_arr = np.array(weights, dtype=np.float32)
    weights_arr = weights_arr / weights_arr.sum()  # normalize
    stacked = np.stack(probabilities, axis=0)  # (M, N, C)
    weighted = np.einsum("m,mnc->nc", weights_arr, stacked)  # (N, C)
    return weighted.argmax(axis=1)


def compute_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> dict:
    """Compute accuracy and macro-F1."""
    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


def aggregate_all_methods(
    probabilities: List[np.ndarray],
    model_keys: List[str],
    true_labels: np.ndarray,
    model_weights: Optional[List[float]] = None,
) -> dict:
    """
    Run all three aggregation methods and return metrics for each.

    Args:
        probabilities: list of (N, C) arrays, one per model.
        model_keys: list of model names.
        true_labels: ground-truth labels (N,).
        model_weights: optional weights for weighted soft vote.

    Returns:
        dict mapping method name → metrics dict.
    """
    results = {}

    # Hard majority vote
    hard_preds = [p.argmax(axis=1) for p in probabilities]
    hard_ensemble = hard_majority_vote(hard_preds)
    results["hard_majority_vote"] = compute_metrics(true_labels, hard_ensemble)

    # Soft vote (mean)
    soft_preds = soft_vote(probabilities)
    results["soft_vote"] = compute_metrics(true_labels, soft_preds)

    # Weighted soft vote
    if model_weights is None:
        model_weights = [1.0] * len(probabilities)
    weighted_preds = weighted_soft_vote(probabilities, model_weights)
    results["weighted_soft_vote"] = compute_metrics(true_labels, weighted_preds)

    # Individual model metrics
    for i, key in enumerate(model_keys):
        preds = probabilities[i].argmax(axis=1)
        results[f"single_{key}"] = compute_metrics(true_labels, preds)

    return results
