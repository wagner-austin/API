"""Protocol-typed accessors onto the ``cleargbm_rs`` native extension.

``cleargbm_rs`` is the compiled Rust extension itself (a ``.pyd``), built by
maturin as a top-level module. This module imports it exactly once,
pins each callable to a ``Protocol`` type so mypy sees precise signatures
instead of ``Any`` leaking from the dynamic ``__import__``, and re-exports the
callables the rest of ``cleargbm`` uses.

Strict typing only: no ``Any``, no ``cast``, no ``type: ignore``. If
``cleargbm_rs`` is not installed, this module raises ``ImportError`` at import
time — there is no Python fallback.

This module is private (underscore prefix); consumers import through
``cleargbm.ensemble`` and ``cleargbm.types``.
"""

from __future__ import annotations

import types
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class PyGbmModelProto(Protocol):
    """Opaque native model handle produced by the Rust training loop."""

    ...


class _TrainProto(Protocol):
    """Signature of ``cleargbm_rs.train_gradient_boosting_rs``."""

    def __call__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        x_val: NDArray[np.float64] | None,
        y_val: NDArray[np.int64] | None,
        config: dict[str, int | float | bool | str | list[int] | None],
        feature_names: list[str],
    ) -> PyGbmModelProto:
        """Train a native binary-classification gradient boosting model.

        Args:
            x_train: 2D training feature matrix.
            y_train: 1D binary training labels.
            x_val: Optional 2D validation feature matrix.
            y_val: Optional 1D validation labels.
            config: Hyperparameter dict (Rust-side shape; produced by
                ``cleargbm.ensemble._config_to_rust_dict``); its
                ``objective`` must be ``"binary_log_loss"``.
            feature_names: Feature names list, length = ``x_train.shape[1]``.

        Returns:
            Opaque native model handle.
        """
        ...


class _TrainRegressionProto(Protocol):
    """Signature of ``cleargbm_rs.train_gradient_boosting_regression_rs``."""

    def __call__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        x_val: NDArray[np.float64] | None,
        y_val: NDArray[np.float64] | None,
        config: dict[str, int | float | bool | str | list[int] | None],
        feature_names: list[str],
    ) -> PyGbmModelProto:
        """Train a native squared-error regression model.

        Args:
            x_train: 2D training feature matrix.
            y_train: 1D continuous training targets.
            x_val: Optional 2D validation feature matrix.
            y_val: Optional 1D continuous validation targets.
            config: Hyperparameter dict (Rust-side shape); its ``objective``
                must be ``"squared_error"``.
            feature_names: Feature names list, length = ``x_train.shape[1]``.

        Returns:
            Opaque native model handle.
        """
        ...


class _PredictProbaProto(Protocol):
    """Signature of ``cleargbm_rs.predict_proba_model_rs``."""

    def __call__(
        self,
        model: PyGbmModelProto,
        features: NDArray[np.float64],
    ) -> tuple[tuple[float, float], ...]:
        """Predict class probabilities from a native model.

        Args:
            model: Trained native model handle.
            features: 2D feature matrix.

        Returns:
            Tuple of ``(prob_class_0, prob_class_1)`` per sample.
        """
        ...


class _PredictRawProto(Protocol):
    """Signature of ``cleargbm_rs.predict_raw_model_rs``."""

    def __call__(
        self,
        model: PyGbmModelProto,
        features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict raw log-odds from a native model.

        Args:
            model: Trained native model handle.
            features: 2D feature matrix.

        Returns:
            1D array of raw log-odds predictions.
        """
        ...


class _ToJsonProto(Protocol):
    """Signature of ``cleargbm_rs.py_gbm_model_to_json_rs``."""

    def __call__(self, model: PyGbmModelProto) -> str:
        """Serialize a native model to JSON.

        Args:
            model: Trained native model handle.

        Returns:
            JSON representation.
        """
        ...


class _FromJsonProto(Protocol):
    """Signature of ``cleargbm_rs.py_gbm_model_from_json_rs``."""

    def __call__(self, json_str: str) -> PyGbmModelProto:
        """Deserialize a native model from JSON.

        Args:
            json_str: JSON produced by the paired to-JSON function.

        Returns:
            Native model handle.
        """
        ...


class _FeatureImportancesProto(Protocol):
    """Signature of ``cleargbm_rs.py_gbm_model_feature_importances_rs``."""

    def __call__(self, model: PyGbmModelProto) -> list[tuple[str, float]]:
        """Return split-count feature importances.

        Args:
            model: Trained native model handle.

        Returns:
            List of ``(feature_name, importance)`` pairs in feature-index order,
            normalized to sum to 1.0 when at least one internal split exists.
        """
        ...


class _NTreesProto(Protocol):
    """Signature of ``cleargbm_rs.py_gbm_model_n_trees_rs``."""

    def __call__(self, model: PyGbmModelProto) -> int:
        """Return the trained tree count.

        Args:
            model: Trained native model handle.

        Returns:
            Number of trees kept in the ensemble.
        """
        ...


_native_mod: types.ModuleType = __import__("cleargbm_rs")

train_gradient_boosting_rs: _TrainProto = _native_mod.train_gradient_boosting_rs
train_gradient_boosting_regression_rs: _TrainRegressionProto = (
    _native_mod.train_gradient_boosting_regression_rs
)
predict_proba_model_rs: _PredictProbaProto = _native_mod.predict_proba_model_rs
predict_raw_model_rs: _PredictRawProto = _native_mod.predict_raw_model_rs
py_gbm_model_to_json_rs: _ToJsonProto = _native_mod.py_gbm_model_to_json_rs
py_gbm_model_from_json_rs: _FromJsonProto = _native_mod.py_gbm_model_from_json_rs
py_gbm_model_feature_importances_rs: _FeatureImportancesProto = (
    _native_mod.py_gbm_model_feature_importances_rs
)
py_gbm_model_n_trees_rs: _NTreesProto = _native_mod.py_gbm_model_n_trees_rs


__all__ = [
    "PyGbmModelProto",
    "predict_proba_model_rs",
    "predict_raw_model_rs",
    "py_gbm_model_feature_importances_rs",
    "py_gbm_model_from_json_rs",
    "py_gbm_model_n_trees_rs",
    "py_gbm_model_to_json_rs",
    "train_gradient_boosting_regression_rs",
    "train_gradient_boosting_rs",
]
