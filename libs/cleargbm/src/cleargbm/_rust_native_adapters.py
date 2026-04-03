"""Rust native training adapters for cleargbm.

Bridges between cleargbm's Python types and cleargbm_rs's full training
loop functions. Each adapter conforms to the corresponding Protocol in
``_hooks_native.py``.

Unlike per-operation adapters in ``_rust_adapters.py``, which accelerate
individual steps, these adapters replace the entire training loop with
a single native call and provide model-level prediction.

No try/except, no auto-detection, no fallback.

This module is private (underscore prefix) - not for external use.
"""

from __future__ import annotations

import types
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cleargbm.types import GradientBoostingConfig

# =============================================================================
# Protocols for native Rust functions (typed getattr assignments)
# =============================================================================


class _PyGbmModel(Protocol):
    """Protocol matching ``cleargbm_rs.PyGbmModel`` (opaque)."""

    ...


class _TrainGradientBoostingRs(Protocol):
    """Protocol matching ``cleargbm_rs.train_gradient_boosting_rs``."""

    def __call__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        x_val: NDArray[np.float64] | None,
        y_val: NDArray[np.int64] | None,
        config: dict[str, int | float | bool | list[int] | None],
        feature_names: list[str],
    ) -> _PyGbmModel:
        """Train gradient boosting model in a single native call.

        Args:
            x_train: 2D training features.
            y_train: 1D binary labels.
            x_val: Optional 2D validation features.
            y_val: Optional 1D validation labels.
            config: Training hyperparameters dict.
            feature_names: Feature name strings.

        Returns:
            Opaque PyGbmModel handle.
        """
        ...


class _PredictProbaModelRs(Protocol):
    """Protocol matching ``cleargbm_rs.predict_proba_model_rs``."""

    def __call__(
        self,
        model: _PyGbmModel,
        features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict class probabilities using a trained model.

        Args:
            model: Trained PyGbmModel.
            features: 2D feature matrix.

        Returns:
            2D array of shape (n_samples, 2).
        """
        ...


class _PredictRawModelRs(Protocol):
    """Protocol matching ``cleargbm_rs.predict_raw_model_rs``."""

    def __call__(
        self,
        model: _PyGbmModel,
        features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict raw log-odds using a trained model.

        Args:
            model: Trained PyGbmModel.
            features: 2D feature matrix.

        Returns:
            1D array of raw predictions.
        """
        ...


# =============================================================================
# Deferred native module loading (called by wire_native_hooks / tests)
# =============================================================================

# Typed module-level references — set by _load_native_functions().
_rs_train_gradient_boosting: _TrainGradientBoostingRs
_rs_predict_proba_model: _PredictProbaModelRs
_rs_predict_raw_model: _PredictRawModelRs


def _load_native_functions() -> None:
    """Load native Rust training functions from cleargbm_rs.

    Must be called before any native adapter function is used. Called
    automatically by ``wire_native_hooks()``. Tests that call adapter
    functions directly should call this at module level.

    Raises:
        ModuleNotFoundError: If cleargbm_rs native extension is not installed.
    """
    global _rs_train_gradient_boosting, _rs_predict_proba_model, _rs_predict_raw_model

    mod: types.ModuleType = __import__("cleargbm_rs.cleargbm_rs", fromlist=["cleargbm_rs"])

    # Typed intermediates — Protocol annotations override Any from ModuleType
    tgb: _TrainGradientBoostingRs = mod.train_gradient_boosting_rs
    ppm: _PredictProbaModelRs = mod.predict_proba_model_rs
    prm: _PredictRawModelRs = mod.predict_raw_model_rs

    _rs_train_gradient_boosting = tgb
    _rs_predict_proba_model = ppm
    _rs_predict_raw_model = prm


# =============================================================================
# Config conversion helper
# =============================================================================


def _config_to_rust_dict(
    config: GradientBoostingConfig,
) -> dict[str, int | float | bool | list[int] | None]:
    """Convert GradientBoostingConfig to dict for Rust training function.

    Extracts the 12 fields that the Rust training loop expects.
    Python-only fields (max_features, track_contributions, n_jobs)
    are excluded.

    Args:
        config: Python GradientBoostingConfig TypedDict.

    Returns:
        Dict with keys matching Rust ``extract_config`` expectations.
    """
    mc = config["monotonic_constraints"]
    mc_list: list[int] | None = list(mc) if mc is not None else None

    result: dict[str, int | float | bool | list[int] | None] = {
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "learning_rate": config["learning_rate"],
        "min_samples_split": config["min_samples_split"],
        "min_samples_leaf": config["min_samples_leaf"],
        "max_bins": config["max_bins"],
        "subsample": config["subsample"],
        "random_state": config["random_state"],
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
        "monotonic_constraints": mc_list,
        "early_stopping_rounds": config["early_stopping_rounds"],
    }
    return result


# =============================================================================
# Adapter functions (match _hooks_native Protocol signatures)
# =============================================================================


def _rust_train_gradient_boosting(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_val: NDArray[np.float64] | None,
    y_val: NDArray[np.int64] | None,
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
) -> _PyGbmModel:
    """Rust-backed full training loop.

    Converts Python config and feature names to Rust-compatible types,
    then calls ``cleargbm_rs.train_gradient_boosting_rs`` which runs
    the entire training loop in native code.

    Args:
        x_train: Training feature matrix (n_samples, n_features).
        y_train: Training labels (0 or 1).
        x_val: Optional validation feature matrix.
        y_val: Optional validation labels.
        config: Training configuration.
        feature_names: Names for each feature.

    Returns:
        Opaque PyGbmModel handle.
    """
    rust_config = _config_to_rust_dict(config)
    names_list: list[str] = list(feature_names)
    model: _PyGbmModel = _rs_train_gradient_boosting(
        x_train,
        y_train,
        x_val,
        y_val,
        rust_config,
        names_list,
    )
    return model


def _rust_predict_proba_model(
    model: _PyGbmModel,
    features: NDArray[np.float64],
) -> tuple[tuple[float, float], ...]:
    """Rust-backed model probability prediction.

    Calls ``cleargbm_rs.predict_proba_model_rs`` and converts the 2D
    numpy result to a tuple of (prob_class_0, prob_class_1) tuples.

    Args:
        model: Opaque PyGbmModel handle.
        features: Feature matrix (n_samples, n_features).

    Returns:
        Tuple of (prob_class_0, prob_class_1) per sample.
    """
    proba_2d: NDArray[np.float64] = _rs_predict_proba_model(model, features)
    n_samples: int = int(proba_2d.shape[0])
    result: list[tuple[float, float]] = []
    for i in range(n_samples):
        p0: float = proba_2d.item(i * 2)
        p1: float = proba_2d.item(i * 2 + 1)
        result.append((p0, p1))
    return tuple(result)


def _rust_predict_raw_model(
    model: _PyGbmModel,
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rust-backed model raw prediction.

    Calls ``cleargbm_rs.predict_raw_model_rs`` directly.

    Args:
        model: Opaque PyGbmModel handle.
        features: Feature matrix (n_samples, n_features).

    Returns:
        1D array of raw predictions (log-odds).
    """
    result: NDArray[np.float64] = _rs_predict_raw_model(model, features)
    return result


# =============================================================================
# Backend wiring (called from _rust_adapters.use_rust_backend)
# =============================================================================


def wire_native_hooks() -> None:
    """Set native training hooks to Rust implementations.

    Called by ``_rust_adapters.use_rust_backend()``.

    Raises:
        ModuleNotFoundError: If cleargbm_rs native extension is not installed.
    """
    _load_native_functions()

    from cleargbm import _hooks_native

    _hooks_native._train_native_backend = _rust_train_gradient_boosting
    _hooks_native._predict_raw_native_backend = _rust_predict_raw_model
    _hooks_native._predict_proba_native_backend = _rust_predict_proba_model


def unwire_native_hooks() -> None:
    """Reset native training hooks to None.

    Called by ``_rust_adapters.use_python_backend()``.
    """
    from cleargbm import _hooks_native

    _hooks_native._train_native_backend = None
    _hooks_native._predict_raw_native_backend = None
    _hooks_native._predict_proba_native_backend = None


__all__ = [
    "_load_native_functions",
    "unwire_native_hooks",
    "wire_native_hooks",
]
