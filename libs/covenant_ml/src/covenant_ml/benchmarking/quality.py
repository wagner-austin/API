"""Assembly of the benchmark's quality metrics.

Every statistic is delegated to :mod:`covenant_ml.metrics`, which holds the
library's pure-numpy implementations. This module only arranges them into a
:class:`~covenant_ml.benchmarking.types.QualityMetrics` record, so a metric
is never defined twice.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..metrics import (
    compute_auc,
    compute_average_precision,
    compute_brier_score,
    compute_log_loss,
)
from .types import ERR_LENGTH_MISMATCH, QualityMetrics


def compute_quality(
    y_true: NDArray[np.int64],
    positive_proba: NDArray[np.float64],
) -> QualityMetrics:
    """Score predictions on the held-out fold.

    Args:
        y_true: True binary labels (0 or 1), shape (n_samples,).
        positive_proba: Predicted positive-class probabilities, shape
            (n_samples,).

    Returns:
        The full quality record for one model at one seed.

    Raises:
        ValueError: If the label and probability arrays differ in length,
            which would silently misalign every metric.
    """
    n_true = len(y_true)
    n_proba = len(positive_proba)
    if n_true != n_proba:
        raise ValueError(
            f"[{ERR_LENGTH_MISMATCH}] y_true and positive_proba must have equal length, "
            f"got {n_true} and {n_proba}"
        )

    return {
        "auc_roc": compute_auc(y_true, positive_proba),
        "auc_pr": compute_average_precision(y_true, positive_proba),
        "log_loss": compute_log_loss(y_true, positive_proba),
        "brier": compute_brier_score(y_true, positive_proba),
        "mean_pred": float(np.sum(positive_proba)) / n_proba,
        "positive_rate": float(np.sum(y_true)) / n_true,
    }


__all__ = ["compute_quality"]
