"""Native (Rust full-loop) training backend hooks for cleargbm.

Provides hooks for the Rust full training loop and model-level prediction.
These hooks have no Python default — they are only available when the Rust
backend is active via ``use_rust_backend()``.

Unlike per-operation hooks in the ``_hooks_*`` sub-modules, which accelerate individual
steps within the Python training loop, these hooks replace the entire training
loop with a single native call.

This module is private (underscore prefix) - not for external use.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cleargbm.types import GradientBoostingConfig

# =============================================================================
# Opaque model type
# =============================================================================


class NativeModel(Protocol):
    """Opaque handle to a natively-trained gradient boosting model.

    Satisfied by ``cleargbm_rs.PyGbmModel``. Has no Python-accessible
    methods — pass to ``predict_raw_native`` or ``predict_proba_native``.
    """

    ...


# =============================================================================
# Protocols
# =============================================================================


class TrainNativeBackend(Protocol):
    """Protocol for native full-loop training backend."""

    def __call__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        x_val: NDArray[np.float64] | None,
        y_val: NDArray[np.int64] | None,
        config: GradientBoostingConfig,
        feature_names: tuple[str, ...],
    ) -> NativeModel:
        """Train gradient boosting model in a single native call.

        Args:
            x_train: Training feature matrix (n_samples, n_features).
            y_train: Training labels (0 or 1).
            x_val: Optional validation feature matrix.
            y_val: Optional validation labels.
            config: Training configuration.
            feature_names: Names for each feature.

        Returns:
            Opaque native model handle.
        """
        ...


class PredictRawNativeBackend(Protocol):
    """Protocol for native model raw prediction backend."""

    def __call__(
        self,
        model: NativeModel,
        features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict raw log-odds using a native model.

        Args:
            model: Native model handle.
            features: Feature matrix (n_samples, n_features).

        Returns:
            1D array of raw predictions (log-odds).
        """
        ...


class PredictProbaNativeBackend(Protocol):
    """Protocol for native model probability prediction backend."""

    def __call__(
        self,
        model: NativeModel,
        features: NDArray[np.float64],
    ) -> tuple[tuple[float, float], ...]:
        """Predict class probabilities using a native model.

        Args:
            model: Native model handle.
            features: Feature matrix (n_samples, n_features).

        Returns:
            Tuple of (prob_class_0, prob_class_1) per sample.
        """
        ...


# =============================================================================
# Module-level hooks (None = Rust backend not active)
# =============================================================================

_train_native_backend: TrainNativeBackend | None = None
_predict_raw_native_backend: PredictRawNativeBackend | None = None
_predict_proba_native_backend: PredictProbaNativeBackend | None = None


# =============================================================================
# Public delegate functions
# =============================================================================


def train_native(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_val: NDArray[np.float64] | None,
    y_val: NDArray[np.int64] | None,
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
) -> NativeModel:
    """Train gradient boosting model using the Rust full training loop.

    Runs the entire training loop in a single native call with no
    per-iteration FFI overhead. Requires ``use_rust_backend()`` first.

    Args:
        x_train: Training feature matrix (n_samples, n_features).
        y_train: Training labels (0 or 1).
        x_val: Optional validation feature matrix.
        y_val: Optional validation labels.
        config: Training configuration.
        feature_names: Names for each feature.

    Returns:
        Opaque native model handle.

    Raises:
        RuntimeError: If Rust backend is not active.
    """
    if _train_native_backend is None:
        msg = "Native training requires Rust backend. Call use_rust_backend() first."
        raise RuntimeError(msg)
    return _train_native_backend(x_train, y_train, x_val, y_val, config, feature_names)


def predict_raw_native(
    model: NativeModel,
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predict raw log-odds using a native model.

    Delegates to the active native prediction backend.

    Args:
        model: Native model handle from ``train_native``.
        features: Feature matrix (n_samples, n_features).

    Returns:
        1D array of raw predictions (log-odds).

    Raises:
        RuntimeError: If Rust backend is not active.
    """
    if _predict_raw_native_backend is None:
        msg = "Native prediction requires Rust backend. Call use_rust_backend() first."
        raise RuntimeError(msg)
    return _predict_raw_native_backend(model, features)


def predict_proba_native(
    model: NativeModel,
    features: NDArray[np.float64],
) -> tuple[tuple[float, float], ...]:
    """Predict class probabilities using a native model.

    Delegates to the active native prediction backend.

    Args:
        model: Native model handle from ``train_native``.
        features: Feature matrix (n_samples, n_features).

    Returns:
        Tuple of (prob_class_0, prob_class_1) per sample.

    Raises:
        RuntimeError: If Rust backend is not active.
    """
    if _predict_proba_native_backend is None:
        msg = "Native prediction requires Rust backend. Call use_rust_backend() first."
        raise RuntimeError(msg)
    return _predict_proba_native_backend(model, features)


__all__ = [
    "NativeModel",
    "PredictProbaNativeBackend",
    "PredictRawNativeBackend",
    "TrainNativeBackend",
    "predict_proba_native",
    "predict_raw_native",
    "train_native",
]
