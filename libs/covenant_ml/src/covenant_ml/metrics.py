"""Metrics computation for covenant breach prediction.

Pure functions for computing classification metrics without sklearn dependencies.
All functions are deterministic and type-safe.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .types import AMEXMetricResult, EvalMetrics


def compute_log_loss(
    y_true: NDArray[np.int64],
    y_prob: NDArray[np.float64],
    eps: float = 1e-15,
) -> float:
    """Compute binary cross-entropy (log loss).

    Args:
        y_true: True binary labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities for class 1, shape (n_samples,)
        eps: Small value to avoid log(0)

    Returns:
        Log loss (lower is better)
    """
    # Clip probabilities to avoid log(0)
    y_prob_clipped: NDArray[np.float64] = np.clip(y_prob, eps, 1 - eps)

    # Convert y_true to float64 for arithmetic
    y_true_float: NDArray[np.float64] = y_true.astype(np.float64)

    # Binary cross-entropy: -mean(y*log(p) + (1-y)*log(1-p))
    log_p: NDArray[np.float64] = np.log(y_prob_clipped)
    log_1_minus_p: NDArray[np.float64] = np.log(1 - y_prob_clipped)
    cross_entropy: NDArray[np.float64] = y_true_float * log_p + (1 - y_true_float) * log_1_minus_p
    # np.mean returns scalar - use sum/len for clean typing
    total = float(np.sum(cross_entropy))
    return -total / len(cross_entropy)


def compute_auc(
    y_true: NDArray[np.int64],
    y_prob: NDArray[np.float64],
) -> float:
    """Compute Area Under ROC Curve using trapezoidal rule.

    This is a pure numpy implementation without sklearn dependency.

    Args:
        y_true: True binary labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities for class 1, shape (n_samples,)

    Returns:
        AUC score (0.5 = random, 1.0 = perfect)
    """
    # Sort by predicted probability descending
    desc_score_indices: NDArray[np.intp] = np.argsort(y_prob)[::-1]
    y_true_sorted: NDArray[np.int64] = y_true[desc_score_indices]

    # Count positives and negatives
    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined, return random baseline

    # Compute TPR and FPR at each threshold
    tps: NDArray[np.int64] = np.cumsum(y_true_sorted)
    fps: NDArray[np.int64] = np.cumsum(1 - y_true_sorted)

    tpr: NDArray[np.float64] = tps.astype(np.float64) / n_pos
    fpr: NDArray[np.float64] = fps.astype(np.float64) / n_neg

    # Add origin point (0, 0) - use typed arrays for concatenation
    origin: NDArray[np.float64] = np.zeros(1, dtype=np.float64)
    tpr_with_origin: NDArray[np.float64] = np.concatenate([origin, tpr])
    fpr_with_origin: NDArray[np.float64] = np.concatenate([origin, fpr])

    # Trapezoidal integration for AUC
    # Manual trapezoid rule: sum((y[i] + y[i+1]) * (x[i+1] - x[i]) / 2)
    dx: NDArray[np.float64] = np.diff(fpr_with_origin)
    avg_y: NDArray[np.float64] = (tpr_with_origin[:-1] + tpr_with_origin[1:]) / 2.0
    return float(np.sum(dx * avg_y))


def compute_average_precision(
    y_true: NDArray[np.int64],
    y_prob: NDArray[np.float64],
) -> float:
    """Compute average precision (area under the precision-recall curve).

    Summarises the precision-recall curve as the recall-weighted mean of the
    precision at each threshold, which is the step-wise definition rather than
    a trapezoidal interpolation. Preferred over ROC-AUC on heavily imbalanced
    data, where a large true-negative count flatters ROC-AUC.

    This is a pure numpy implementation without sklearn dependency.

    Args:
        y_true: True binary labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities for class 1, shape (n_samples,)

    Returns:
        Average precision. Returns 0.0 when the labels contain no positive
        case, where the metric is otherwise undefined.
    """
    n_pos = int(np.sum(y_true))
    if n_pos == 0:
        return 0.0

    desc_score_indices: NDArray[np.intp] = np.argsort(y_prob)[::-1]
    y_true_sorted: NDArray[np.int64] = y_true[desc_score_indices]

    tps: NDArray[np.int64] = np.cumsum(y_true_sorted)
    ranks: NDArray[np.float64] = np.arange(1, len(y_true_sorted) + 1, dtype=np.float64)

    precision: NDArray[np.float64] = tps.astype(np.float64) / ranks
    recall: NDArray[np.float64] = tps.astype(np.float64) / n_pos

    # Recall gain at each position; the first position gains from zero.
    origin: NDArray[np.float64] = np.zeros(1, dtype=np.float64)
    recall_head: NDArray[np.float64] = recall[:-1]
    recall_prev: NDArray[np.float64] = np.concatenate([origin, recall_head])
    recall_gain: NDArray[np.float64] = recall - recall_prev

    return float(np.sum(recall_gain * precision))


def compute_brier_score(
    y_true: NDArray[np.int64],
    y_prob: NDArray[np.float64],
) -> float:
    """Compute the Brier score (mean squared error of the probabilities).

    Measures calibration as well as discrimination: a model that ranks cases
    perfectly but is systematically over-confident scores worse than one that
    ranks equally well and is calibrated. Lower is better.

    This is a pure numpy implementation without sklearn dependency.

    Args:
        y_true: True binary labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities for class 1, shape (n_samples,)

    Returns:
        Brier score in the range [0.0, 1.0], where 0.0 is a perfect forecast.
    """
    y_true_float: NDArray[np.float64] = y_true.astype(np.float64)
    squared_error: NDArray[np.float64] = (y_prob - y_true_float) ** 2
    total = float(np.sum(squared_error))
    return total / len(squared_error)


def compute_amex_metric(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.float64],
) -> AMEXMetricResult:
    """Compute AMEX competition metric: 0.5 * (normalized_gini + D@4%).

    The AMEX Default Prediction competition uses a custom metric that combines:
    1. Normalized Gini coefficient (measures ranking quality)
    2. Default rate captured at top 4% (measures precision at high confidence)

    Both components apply 20x weight to negative samples to account for
    the 5% subsampling of the negative class in the competition data.

    Args:
        y_true: Ground truth binary labels (0 or 1), shape (n_samples,).
        y_pred: Predicted probabilities for default (positive class), shape (n_samples,).

    Returns:
        AMEXMetricResult with score, normalized_gini, and default_rate_at_4_percent.

    Raises:
        ValueError: If arrays have different lengths or no positive samples exist.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Array length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}"
        )

    n_samples = len(y_true)
    if n_samples == 0:
        raise ValueError("Cannot compute metric on empty arrays")

    n_pos = int(np.sum(y_true))
    if n_pos == 0:
        raise ValueError("Cannot compute metric with no positive samples")

    # Convert to float64 for calculations
    y_true_float: NDArray[np.float64] = y_true.astype(np.float64)

    # Compute default rate at top 4% (D@4%)
    # Sort by prediction descending
    desc_indices: NDArray[np.intp] = np.argsort(y_pred)[::-1]
    y_true_sorted: NDArray[np.float64] = y_true_float[desc_indices]

    # Apply 20x weight to negatives (accounts for 5% subsampling)
    # Use explicit array construction to avoid np.where typing issues
    weights: NDArray[np.float64] = np.ones(len(y_true_sorted), dtype=np.float64)
    neg_mask: NDArray[np.bool_] = y_true_sorted == 0
    weights[neg_mask] = 20.0
    total_weight = float(np.sum(weights))
    weight_cutoff = int(0.04 * total_weight)

    # Find samples in top 4% by cumulative weight
    cum_weights: NDArray[np.float64] = np.cumsum(weights)
    top_4_mask: NDArray[np.bool_] = cum_weights <= weight_cutoff
    top_4_positives = float(np.sum(y_true_sorted[top_4_mask]))
    total_positives = float(np.sum(y_true_sorted))
    default_rate_at_4: float = top_4_positives / total_positives if total_positives > 0 else 0.0

    # Compute Gini coefficients
    # gini_pred: Gini when sorted by predictions (actual model performance)
    # gini_true: Gini when sorted by truth (perfect model - for normalization)
    gini_pred = _compute_weighted_gini(y_true_float, y_pred, sort_by_pred=True)
    gini_true = _compute_weighted_gini(y_true_float, y_pred, sort_by_pred=False)

    # Normalized Gini = gini_pred / gini_true
    normalized_gini: float = gini_pred / gini_true if gini_true != 0 else 0.0

    # Final score: average of normalized Gini and D@4%
    score: float = 0.5 * (normalized_gini + default_rate_at_4)

    return AMEXMetricResult(
        score=score,
        normalized_gini=normalized_gini,
        default_rate_at_4_percent=default_rate_at_4,
    )


def _compute_weighted_gini(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    sort_by_pred: bool,
) -> float:
    """Compute weighted Gini coefficient for AMEX metric.

    Args:
        y_true: Ground truth labels as float64.
        y_pred: Predicted probabilities.
        sort_by_pred: If True, sort by predictions; if False, sort by truth.

    Returns:
        Weighted Gini coefficient (unnormalized).
    """
    # Sort by predictions or truth
    if sort_by_pred:
        sort_indices: NDArray[np.intp] = np.argsort(y_pred)[::-1]
    else:
        sort_indices = np.argsort(y_true)[::-1]

    y_true_sorted: NDArray[np.float64] = y_true[sort_indices]

    # Apply 20x weight to negatives
    weights: NDArray[np.float64] = np.ones(len(y_true_sorted), dtype=np.float64)
    neg_mask: NDArray[np.bool_] = y_true_sorted == 0
    weights[neg_mask] = 20.0
    total_weight = float(np.sum(weights))

    # Cumulative weight (normalized to [0, 1])
    weight_random: NDArray[np.float64] = np.cumsum(weights) / total_weight

    # Weighted cumulative positives (Lorentz curve)
    weighted_positives: NDArray[np.float64] = y_true_sorted * weights
    total_weighted_pos = float(np.sum(weighted_positives))

    if total_weighted_pos == 0:
        return 0.0

    cum_pos: NDArray[np.float64] = np.cumsum(weighted_positives)
    lorentz: NDArray[np.float64] = cum_pos / total_weighted_pos

    # Gini = sum of (lorentz - random) weighted by sample weights
    gini_values: NDArray[np.float64] = (lorentz - weight_random) * weights
    return float(np.sum(gini_values))


def compute_accuracy(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
) -> float:
    """Compute classification accuracy.

    Args:
        y_true: True binary labels, shape (n_samples,)
        y_pred: Predicted binary labels, shape (n_samples,)

    Returns:
        Accuracy (0.0 to 1.0)
    """
    correct: NDArray[np.bool_] = y_true == y_pred
    correct_count = int(np.sum(correct))
    return float(correct_count) / float(len(y_true))


def compute_precision(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
) -> float:
    """Compute precision for positive class (breach).

    Precision = TP / (TP + FP)

    Args:
        y_true: True binary labels, shape (n_samples,)
        y_pred: Predicted binary labels, shape (n_samples,)

    Returns:
        Precision (0.0 to 1.0), 0.0 if no positive predictions
    """
    pred_positive: NDArray[np.bool_] = y_pred == 1
    true_positive: NDArray[np.bool_] = y_true == 1
    tp_mask: NDArray[np.bool_] = pred_positive & true_positive

    true_positives = int(np.sum(tp_mask))
    predicted_positives = int(np.sum(pred_positive))

    if predicted_positives == 0:
        return 0.0
    return float(true_positives) / float(predicted_positives)


def compute_recall(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
) -> float:
    """Compute recall for positive class (breach).

    Recall = TP / (TP + FN)

    Args:
        y_true: True binary labels, shape (n_samples,)
        y_pred: Predicted binary labels, shape (n_samples,)

    Returns:
        Recall (0.0 to 1.0), 0.0 if no actual positives
    """
    pred_positive: NDArray[np.bool_] = y_pred == 1
    true_positive: NDArray[np.bool_] = y_true == 1
    tp_mask: NDArray[np.bool_] = pred_positive & true_positive

    true_positives = int(np.sum(tp_mask))
    actual_positives = int(np.sum(true_positive))

    if actual_positives == 0:
        return 0.0
    return float(true_positives) / float(actual_positives)


def compute_f1_score(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall.

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        precision: Precision score
        recall: Recall score

    Returns:
        F1 score (0.0 to 1.0), 0.0 if both precision and recall are 0
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_all_metrics(
    y_true: NDArray[np.int64],
    y_prob: NDArray[np.float64],
    threshold: float = 0.5,
) -> EvalMetrics:
    """Compute all evaluation metrics for a dataset split.

    Args:
        y_true: True binary labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities for class 1, shape (n_samples,)
        threshold: Classification threshold for converting prob to label

    Returns:
        EvalMetrics with all computed metrics
    """
    # Convert probabilities to binary predictions
    y_pred: NDArray[np.int64] = (y_prob >= threshold).astype(np.int64)

    # Compute all metrics
    loss = compute_log_loss(y_true, y_prob)
    ppl = float(math.exp(loss))
    auc = compute_auc(y_true, y_prob)
    accuracy = compute_accuracy(y_true, y_pred)
    precision = compute_precision(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)
    f1 = compute_f1_score(precision, recall)

    return EvalMetrics(
        loss=loss,
        ppl=ppl,
        auc=auc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
    )


def format_metrics_str(metrics: EvalMetrics) -> str:
    """Format metrics as a human-readable string.

    Args:
        metrics: Computed evaluation metrics

    Returns:
        Formatted string like "loss=0.45 auc=0.82 acc=0.78 P=0.75 R=0.80 F1=0.77"
    """
    return (
        f"loss={metrics['loss']:.4f} "
        f"ppl={metrics['ppl']:.4f} "
        f"auc={metrics['auc']:.4f} "
        f"acc={metrics['accuracy']:.4f} "
        f"P={metrics['precision']:.4f} "
        f"R={metrics['recall']:.4f} "
        f"F1={metrics['f1_score']:.4f}"
    )


# =============================================================================
# Regression Metrics (continuous predictions vs continuous targets)
# =============================================================================


__all__ = [
    "compute_accuracy",
    "compute_all_metrics",
    "compute_amex_metric",
    "compute_auc",
    "compute_f1_score",
    "compute_log_loss",
    "compute_precision",
    "compute_recall",
    "format_metrics_str",
]
