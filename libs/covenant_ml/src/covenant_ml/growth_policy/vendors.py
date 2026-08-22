"""Typed boundaries onto the three vendors this experiment reaches.

Every vendor is resolved through :func:`__import__` with the members named in
``fromlist``, and the result is assigned straight to a ``Protocol``-typed
variable, which is where the type comes from. That is required rather than
stylistic: ``lightgbm`` and ``scikit-learn`` ship no ``py.typed`` marker, so a
direct import would pull untyped modules into a package configured with
``disallow_any_unimported`` and ``disallow_any_expr``.

The LightGBM and XGBoost Protocols are imported from
:mod:`covenant_ml.benchmarking.adapters` rather than restated. Two declarations
of one vendor signature can drift into disagreeing about a surface only one of
them has checked, and the drift is silent because each type-checks alone.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..benchmarking.adapters import (
    LGBMClassifierProto,
    XgbBoosterProto,
    XgbClassifierCtor,
    XgbClassifierProto,
    XgbFittableProto,
)


class LgbClassifierCtor(Protocol):
    """Protocol for LightGBM's classifier constructor."""

    def __call__(
        self,
        *,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
        max_bin: int,
        min_child_samples: int,
        num_leaves: int,
        reg_alpha: float,
        reg_lambda: float,
        n_jobs: int,
        random_state: int,
        verbose: int,
    ) -> LGBMClassifierProto: ...


class RocAucProto(Protocol):
    """Protocol for scikit-learn's ROC-AUC metric."""

    def __call__(self, y_true: NDArray[np.int64], y_score: NDArray[np.float64]) -> float:
        """Score predictions.

        Args:
            y_true: True binary labels, shape (n_samples,).
            y_score: Positive-class scores, shape (n_samples,).

        Returns:
            Area under the ROC curve.
        """
        ...


class AveragePrecisionProto(Protocol):
    """Protocol for scikit-learn's average-precision metric."""

    def __call__(self, y_true: NDArray[np.int64], y_score: NDArray[np.float64]) -> float:
        """Score predictions.

        Args:
            y_true: True binary labels, shape (n_samples,).
            y_score: Positive-class scores, shape (n_samples,).

        Returns:
            Area under the precision-recall curve.
        """
        ...


class LogLossProto(Protocol):
    """Protocol for scikit-learn's log-loss metric."""

    def __call__(
        self,
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.float64],
        *,
        labels: list[int],
    ) -> float:
        """Score predictions.

        Args:
            y_true: True binary labels, shape (n_samples,).
            y_pred: Positive-class probabilities, shape (n_samples,).
            labels: The full label set, passed so a fold that happens to miss
                a class still scores against both.

        Returns:
            The log loss.
        """
        ...


class StratifiedSplitProto(Protocol):
    """Protocol for scikit-learn's stratified splitter."""

    def __call__(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
        *,
        test_size: float,
        random_state: int,
        stratify: NDArray[np.int64],
    ) -> list[NDArray[np.float64] | NDArray[np.int64]]:
        """Partition arrays into a stratified train/test split.

        The return type is a heterogeneous list because that is what
        ``train_test_split`` actually returns: one list holding the feature
        folds and the label folds together. Callers narrow each element to its
        real dtype rather than assuming the order carries the type.

        Args:
            x: Feature matrix, shape (n_samples, n_features).
            y: Labels, shape (n_samples,).
            test_size: Fraction of rows held out.
            random_state: Seed controlling the permutation.
            stratify: Array whose class proportions are preserved.

        Returns:
            ``[x_train, x_test, y_train, y_test]``.
        """
        ...


def load_xgb_ctor() -> XgbClassifierCtor:
    """Resolve XGBoost's classifier constructor.

    Returns:
        The ``XGBClassifier`` constructor.
    """
    module = __import__("xgboost", fromlist=["XGBClassifier"])
    constructor: XgbClassifierCtor = module.XGBClassifier
    return constructor


def load_lgb_ctor() -> LgbClassifierCtor:
    """Resolve LightGBM's classifier constructor.

    Returns:
        The ``LGBMClassifier`` constructor.
    """
    module = __import__("lightgbm", fromlist=["LGBMClassifier"])
    constructor: LgbClassifierCtor = module.LGBMClassifier
    return constructor


def load_roc_auc() -> RocAucProto:
    """Resolve scikit-learn's ROC-AUC metric.

    Returns:
        The ``roc_auc_score`` callable.
    """
    module = __import__("sklearn.metrics", fromlist=["roc_auc_score"])
    metric: RocAucProto = module.roc_auc_score
    return metric


def load_average_precision() -> AveragePrecisionProto:
    """Resolve scikit-learn's average-precision metric.

    Returns:
        The ``average_precision_score`` callable.
    """
    module = __import__("sklearn.metrics", fromlist=["average_precision_score"])
    metric: AveragePrecisionProto = module.average_precision_score
    return metric


def load_log_loss() -> LogLossProto:
    """Resolve scikit-learn's log-loss metric.

    Returns:
        The ``log_loss`` callable.
    """
    module = __import__("sklearn.metrics", fromlist=["log_loss"])
    metric: LogLossProto = module.log_loss
    return metric


def load_stratified_split() -> StratifiedSplitProto:
    """Resolve scikit-learn's stratified splitter.

    Returns:
        The ``train_test_split`` callable.
    """
    module = __import__("sklearn.model_selection", fromlist=["train_test_split"])
    splitter: StratifiedSplitProto = module.train_test_split
    return splitter


__all__ = [
    "AveragePrecisionProto",
    "LgbClassifierCtor",
    "LogLossProto",
    "RocAucProto",
    "StratifiedSplitProto",
    "XgbBoosterProto",
    "XgbClassifierCtor",
    "XgbClassifierProto",
    "XgbFittableProto",
    "load_average_precision",
    "load_lgb_ctor",
    "load_log_loss",
    "load_roc_auc",
    "load_stratified_split",
    "load_xgb_ctor",
]
