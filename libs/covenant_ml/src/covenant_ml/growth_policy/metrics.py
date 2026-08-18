"""The three metrics every arm is scored on.

These are scikit-learn's implementations rather than
:mod:`covenant_ml.metrics`, and the reason is specific: the tables recorded in
``libs/cleargbm/docs/EXPERIMENT_2026-08-17_growth_policy_xgb_instrument.md``
were measured with them. This package's obligation is to reproduce those
tables, so scoring them through a second implementation -- even a correct one
-- would mean the document's numbers no longer come from the code that claims
to produce them.

The three callables are injected rather than imported here, so this class names
no vendor and :mod:`covenant_ml.growth_policy.factory` stays the only place
that does.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .vendors import AveragePrecisionProto, LogLossProto, RocAucProto

#: The label set passed to log loss, so a fold that happens to contain one
#: class still scores against both instead of raising.
_BINARY_LABELS = [0, 1]


class SklearnMetrics:
    """Scores held-out predictions on the experiment's three metrics."""

    def __init__(
        self,
        roc_auc: RocAucProto,
        average_precision: AveragePrecisionProto,
        log_loss: LogLossProto,
    ) -> None:
        """Bind the three metric callables.

        Args:
            roc_auc: Area under the ROC curve.
            average_precision: Area under the precision-recall curve.
            log_loss: Log loss.
        """
        self._roc_auc = roc_auc
        self._average_precision = average_precision
        self._log_loss = log_loss

    def auc_roc(self, y_true: NDArray[np.int64], positive_proba: NDArray[np.float64]) -> float:
        """Score area under the ROC curve.

        Args:
            y_true: True binary labels, shape (n_samples,).
            positive_proba: Positive-class probabilities, shape (n_samples,).

        Returns:
            The metric value.
        """
        return float(self._roc_auc(y_true, positive_proba))

    def auc_pr(self, y_true: NDArray[np.int64], positive_proba: NDArray[np.float64]) -> float:
        """Score area under the precision-recall curve.

        Args:
            y_true: True binary labels, shape (n_samples,).
            positive_proba: Positive-class probabilities, shape (n_samples,).

        Returns:
            The metric value.
        """
        return float(self._average_precision(y_true, positive_proba))

    def log_loss(self, y_true: NDArray[np.int64], positive_proba: NDArray[np.float64]) -> float:
        """Score log loss.

        Args:
            y_true: True binary labels, shape (n_samples,).
            positive_proba: Positive-class probabilities, shape (n_samples,).

        Returns:
            The metric value.
        """
        return float(self._log_loss(y_true, positive_proba, labels=_BINARY_LABELS))


__all__ = ["SklearnMetrics"]
