"""AMEX pipeline: dataset/test configs and the backend trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from covenant_ml.backends.protocol import ClassifierBackend
from covenant_ml.datasets import TimeSeriesDatasetConfig
from covenant_ml.types import (
    BackendName,
    LightGBMConfig,
    TrainConfig,
)
from covenant_ml.validation.runner import TrainedModel
from numpy.typing import NDArray


def build_dataset_config(
    data_dir: Path,
    aggregation: Literal["last", "first", "mean", "statistics"],
    include_rank_features: bool,
    include_diff_features: bool,
    include_window_features: bool,
    window_sizes: tuple[int, ...],
) -> TimeSeriesDatasetConfig:
    """Build a time-series dataset configuration for AMEX data.

    Args:
        data_dir: Directory containing data.csv and labels.csv.
        aggregation: Time-series aggregation strategy.
        include_rank_features: Whether to compute per-entity rank features.
        include_diff_features: Whether to compute row-to-row diff features.
        include_window_features: Whether to compute window features.
        window_sizes: Window sizes for window features.

    Returns:
        TimeSeriesDatasetConfig ready for loader.
    """
    return TimeSeriesDatasetConfig(
        name="amex_train",
        display_name="AMEX Training Data",
        folder=data_dir.name,
        file_name="train_data.csv",
        file_format="csv",
        encoding="utf-8",
        target={
            "column_name": "target",
            "label_type": "binary_int",
            "positive_values": (1,),
            "negative_values": (0,),
        },
        exclude_columns=(),
        n_samples_expected=0,  # Not validated for competition
        n_features_expected=0,  # Not validated for competition
        positive_class_ratio_expected=0.0,  # Not validated for competition
        time_series={
            "entity_column": "customer_ID",
            "time_column": "S_2",
            "aggregation": aggregation,
            "labels_file": "train_labels.csv",
            "labels_entity_column": "customer_ID",
            "include_rank_features": include_rank_features,
            "include_diff_features": include_diff_features,
            "include_window_features": include_window_features,
            "window_sizes": window_sizes,
        },
    )


def build_test_config(
    data_dir: Path,
    aggregation: Literal["last", "first", "mean", "statistics"],
    include_rank_features: bool,
    include_diff_features: bool,
    include_window_features: bool,
    window_sizes: tuple[int, ...],
) -> TimeSeriesDatasetConfig:
    """Build a time-series dataset configuration for AMEX test data.

    Args:
        data_dir: Directory containing test_data.csv.
        aggregation: Time-series aggregation strategy.
        include_rank_features: Whether to compute per-entity rank features.
        include_diff_features: Whether to compute row-to-row diff features.
        include_window_features: Whether to compute window features.
        window_sizes: Window sizes for window features.

    Returns:
        TimeSeriesDatasetConfig ready for loader.
    """
    return TimeSeriesDatasetConfig(
        name="amex_test",
        display_name="AMEX Test Data",
        folder=data_dir.name,
        file_name="test_data.csv",
        file_format="csv",
        encoding="utf-8",
        target={
            "column_name": "target",
            "label_type": "binary_int",
            "positive_values": (1,),
            "negative_values": (0,),
        },
        exclude_columns=(),
        n_samples_expected=0,
        n_features_expected=0,
        positive_class_ratio_expected=0.0,
        time_series={
            "entity_column": "customer_ID",
            "time_column": "S_2",
            "aggregation": aggregation,
            "labels_file": "",  # Test data has no labels
            "labels_entity_column": "customer_ID",
            "include_rank_features": include_rank_features,
            "include_diff_features": include_diff_features,
            "include_window_features": include_window_features,
            "window_sizes": window_sizes,
        },
    )


# =============================================================================
# Backend Config Builders
# =============================================================================


def _build_lightgbm_config(
    n_estimators: int,
    learning_rate: float,
    random_state: int,
) -> LightGBMConfig:
    """Build LightGBM training configuration.

    Args:
        n_estimators: Number of boosting rounds.
        learning_rate: Learning rate.
        random_state: Random seed.

    Returns:
        LightGBM-specific training configuration.
    """
    return LightGBMConfig(
        device="cpu",
        learning_rate=learning_rate,
        max_depth=-1,
        n_estimators=n_estimators,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=random_state,
        early_stopping_rounds=50,
    )


def _build_xgboost_config(
    n_estimators: int,
    learning_rate: float,
    random_state: int,
) -> TrainConfig:
    """Build XGBoost training configuration.

    Args:
        n_estimators: Number of boosting rounds.
        learning_rate: Learning rate.
        random_state: Random seed.

    Returns:
        XGBoost-specific training configuration.
    """
    return TrainConfig(
        device="cpu",
        learning_rate=learning_rate,
        max_depth=6,
        n_estimators=n_estimators,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=random_state,
        early_stopping_rounds=50,
    )


# =============================================================================
# Trainer Factory
# =============================================================================


class _BackendTrainer:
    """Trainer wrapper that uses a ClassifierBackend."""

    def __init__(
        self,
        backend: ClassifierBackend,
        backend_name: BackendName,
        n_estimators: int,
        learning_rate: float,
        random_state: int,
        feature_names: tuple[str, ...],
        output_dir: Path,
    ) -> None:
        """Initialize trainer.

        Args:
            backend: ML backend to use.
            backend_name: Name of the backend.
            n_estimators: Number of estimators.
            learning_rate: Learning rate.
            random_state: Random seed.
            feature_names: Feature column names.
            output_dir: Directory for model artifacts.
        """
        self._backend = backend
        self._backend_name = backend_name
        self._n_estimators = n_estimators
        self._learning_rate = learning_rate
        self._random_state = random_state
        self._feature_names = feature_names
        self._output_dir = output_dir

    def __call__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
        fold_number: int,
    ) -> TrainedModel:
        """Train model on one fold.

        Args:
            x_train: Training features.
            y_train: Training labels.
            x_val: Validation features.
            y_val: Validation labels.
            fold_number: Fold index.

        Returns:
            Trained model.
        """
        # Build config based on backend
        config: LightGBMConfig | TrainConfig
        if self._backend_name == "lightgbm":
            config = _build_lightgbm_config(
                self._n_estimators, self._learning_rate, self._random_state
            )
        else:
            config = _build_xgboost_config(
                self._n_estimators, self._learning_rate, self._random_state
            )

        # Create fold output directory
        fold_dir = self._output_dir / f"fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Train using backend
        outcome = self._backend.train(
            x_features=x_train,
            y_labels=y_train,
            feature_names=list(self._feature_names),
            config=config,
            output_dir=fold_dir,
            progress=None,
        )

        # Load trained model
        model_path = outcome["model_path"]
        return self._backend.load(path=str(model_path))


# =============================================================================
# Training Functions
# =============================================================================
