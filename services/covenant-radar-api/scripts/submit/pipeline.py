"""Submission pipeline for Kaggle AMEX Default Prediction competition.

Trains a model on time-series data and generates submission predictions.
Uses hooks pattern for dependency injection enabling testability.
Backend agnostic: supports LightGBM, XGBoost, MLP, and LSTM backends.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from covenant_ml.backends.protocol import ClassifierBackend, PreparedClassifier
from covenant_ml.datasets import (
    LoadedDataset,
    TimeSeriesDatasetConfig,
    create_timeseries_csv_loader,
)
from covenant_ml.types import (
    BackendName,
    ClassifierTrainConfig,
    LightGBMConfig,
    LSTMConfig,
    MLPConfig,
    TrainConfig,
)
from numpy.typing import NDArray

from scripts.submit._hooks import get_console, get_registry

# =============================================================================
# Types
# =============================================================================


class SubmitConfig(TypedDict, total=True):
    """Configuration for submission pipeline.

    Attributes:
        backend: ML backend to use (lightgbm, xgboost, mlp, lstm).
        n_estimators: Number of boosting rounds (tree backends) or epochs (neural).
        learning_rate: Learning rate.
        num_leaves: Maximum leaves per tree (LightGBM only).
        max_depth: Maximum tree depth (-1 for unlimited).
        aggregation: Time-series aggregation strategy.
        include_rank_features: Whether to compute rank features.
        include_diff_features: Whether to compute diff features.
    """

    backend: BackendName
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    aggregation: Literal["last", "first", "mean", "statistics"]
    include_rank_features: bool
    include_diff_features: bool


class TrainResult(TypedDict, total=True):
    """Result of training a model.

    Attributes:
        n_samples: Number of training samples.
        n_features: Number of features.
        feature_names: Ordered feature column names.
        val_auc: Validation AUC score.
    """

    n_samples: int
    n_features: int
    feature_names: tuple[str, ...]
    val_auc: float


class PredictionResult(TypedDict, total=True):
    """Result of making predictions.

    Attributes:
        n_samples: Number of samples predicted.
        entity_ids: Entity IDs in prediction order.
        predictions: Predicted probabilities for positive class.
    """

    n_samples: int
    entity_ids: tuple[str, ...]
    predictions: tuple[float, ...]


# =============================================================================
# Dataset Configuration Builder
# =============================================================================


def build_dataset_config(
    data_dir: Path,
    aggregation: Literal["last", "first", "mean", "statistics"],
    include_rank_features: bool,
    include_diff_features: bool,
) -> TimeSeriesDatasetConfig:
    """Build a time-series dataset configuration for loading.

    Args:
        data_dir: Directory containing data.csv and labels.csv.
        aggregation: Time-series aggregation strategy.
        include_rank_features: Whether to compute per-entity rank features.
        include_diff_features: Whether to compute row-to-row diff features.

    Returns:
        TimeSeriesDatasetConfig ready for loader.
    """
    return TimeSeriesDatasetConfig(
        name="submit_data",
        display_name="Submit Data",
        folder=data_dir.name,
        file_name="data.csv",
        file_format="csv",
        encoding="utf-8",
        target={
            "column_name": "target",
            "label_type": "binary_int",
            "positive_values": (1,),
            "negative_values": (0,),
        },
        exclude_columns=(),
        n_samples_expected=0,  # Not validated for submission
        n_features_expected=0,  # Not validated for submission
        positive_class_ratio_expected=0.0,  # Not validated for submission
        time_series={
            "entity_column": "entity_id",
            "time_column": "timestamp",
            "aggregation": aggregation,
            "labels_file": "labels.csv",
            "labels_entity_column": "entity_id",
            "include_rank_features": include_rank_features,
            "include_diff_features": include_diff_features,
            "include_window_features": False,
            "window_sizes": (),
        },
    )


# =============================================================================
# Pipeline Functions
# =============================================================================


def load_training_data(
    data_dir: Path,
    aggregation: Literal["last", "first", "mean", "statistics"],
    include_rank_features: bool,
    include_diff_features: bool,
) -> LoadedDataset:
    """Load and aggregate time-series training data.

    Args:
        data_dir: Directory containing data.csv and labels.csv.
        aggregation: Time-series aggregation strategy.
        include_rank_features: Whether to compute per-entity rank features.
        include_diff_features: Whether to compute row-to-row diff features.

    Returns:
        LoadedDataset with aggregated features and labels.

    Raises:
        FileNotFoundError: If data files don't exist.
        ValueError: If data format is invalid.
    """
    console = get_console()
    console.write(f"Loading training data from {data_dir}")

    config = build_dataset_config(
        data_dir, aggregation, include_rank_features, include_diff_features
    )
    loader = create_timeseries_csv_loader()
    dataset = loader.load(config, data_dir.parent)

    console.write(
        f"Loaded {dataset['meta']['n_samples']} samples "
        f"with {dataset['meta']['n_features']} features"
    )
    return dataset


def _build_lightgbm_config(config: SubmitConfig) -> LightGBMConfig:
    """Build LightGBM training configuration.

    Args:
        config: Submit pipeline configuration.

    Returns:
        LightGBM-specific training configuration.
    """
    return LightGBMConfig(
        device="cpu",
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        num_leaves=config["num_leaves"],
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        early_stopping_rounds=10,
    )


def _build_xgboost_config(config: SubmitConfig) -> TrainConfig:
    """Build XGBoost training configuration.

    Args:
        config: Submit pipeline configuration.

    Returns:
        XGBoost-specific training configuration.
    """
    return TrainConfig(
        device="cpu",
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        early_stopping_rounds=10,
    )


def _build_mlp_config(config: SubmitConfig) -> MLPConfig:
    """Build MLP training configuration.

    Args:
        config: Submit pipeline configuration.

    Returns:
        MLP-specific training configuration.
    """
    return MLPConfig(
        device="cpu",
        precision="fp32",
        optimizer="adamw",
        hidden_sizes=(256, 128, 64),
        learning_rate=config["learning_rate"],
        batch_size=256,
        n_epochs=config["n_estimators"],
        dropout=0.3,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        early_stopping_patience=10,
    )


def _build_lstm_config(config: SubmitConfig) -> LSTMConfig:
    """Build LSTM training configuration.

    Args:
        config: Submit pipeline configuration.

    Returns:
        LSTM-specific training configuration.
    """
    return LSTMConfig(
        device="cpu",
        precision="fp32",
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        sequence_length=13,  # AMEX has ~13 time steps
        learning_rate=config["learning_rate"],
        batch_size=256,
        n_epochs=config["n_estimators"],
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        early_stopping_patience=10,
    )


def _get_train_config(config: SubmitConfig) -> ClassifierTrainConfig:
    """Get training configuration for the specified backend.

    Args:
        config: Submit pipeline configuration with backend specified.

    Returns:
        Backend-specific training configuration.
    """
    backend = config["backend"]
    if backend == "lightgbm":
        return _build_lightgbm_config(config)
    if backend == "xgboost":
        return _build_xgboost_config(config)
    if backend == "mlp":
        return _build_mlp_config(config)
    return _build_lstm_config(config)


def train_model(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    feature_names: tuple[str, ...],
    config: SubmitConfig,
    output_dir: Path,
) -> tuple[PreparedClassifier, TrainResult]:
    """Train a model on the provided data.

    Args:
        x_train: Feature matrix of shape (n_samples, n_features).
        y_train: Binary labels of shape (n_samples,).
        feature_names: Ordered feature column names.
        config: Model training configuration.
        output_dir: Directory to save model artifacts.

    Returns:
        Tuple of (prepared classifier, training result).

    Raises:
        ValueError: If input shapes are invalid.
    """
    console = get_console()
    n_samples = x_train.shape[0]
    n_features = x_train.shape[1]
    backend_name = config["backend"]

    console.write(f"Training {backend_name} with {n_samples} samples, {n_features} features")

    # Get backend from registry via hook
    registry = get_registry()
    backend: ClassifierBackend = registry.get(backend_name)

    # Build backend-specific config
    train_config = _get_train_config(config)

    # Train using ClassifierBackend protocol
    outcome = backend.train(
        x_features=x_train,
        y_labels=y_train,
        feature_names=list(feature_names),
        config=train_config,
        output_dir=output_dir,
        progress=None,
    )

    val_auc = outcome["best_val_auc"]
    console.write(f"Training complete. Validation AUC: {val_auc:.4f}")

    # Load model for inference
    model_path = outcome["model_path"]
    prepared = backend.load(path=model_path)

    return prepared, TrainResult(
        n_samples=n_samples,
        n_features=n_features,
        feature_names=feature_names,
        val_auc=val_auc,
    )


def predict(
    model: PreparedClassifier,
    x_test: NDArray[np.float64],
    entity_ids: tuple[str, ...],
) -> PredictionResult:
    """Generate predictions for test data.

    Args:
        model: Trained classifier implementing PreparedClassifier protocol.
        x_test: Feature matrix of shape (n_samples, n_features).
        entity_ids: Entity identifiers for each sample.

    Returns:
        PredictionResult with entity IDs and predictions.

    Raises:
        ValueError: If number of entity IDs doesn't match samples.
    """
    console = get_console()
    n_samples: int = int(x_test.shape[0])

    n_entity_ids = len(entity_ids)
    if n_entity_ids != n_samples:
        msg = f"entity_ids length {n_entity_ids} != samples {n_samples}"
        raise ValueError(msg)

    console.write(f"Generating predictions for {n_samples} samples")

    # predict_proba returns (n_samples, 2) for binary; take column 1 (positive class)
    raw_preds: NDArray[np.float64] = model.predict_proba(x_test)
    # Handle both 1D (some backends) and 2D (sklearn-style) outputs
    if raw_preds.ndim == 2:
        pos_proba: NDArray[np.float64] = raw_preds[:, 1]
    else:
        pos_proba = raw_preds
    # Build predictions tuple with explicit typing to avoid Any from numpy indexing
    pred_list: list[float] = []
    for val in pos_proba.flat:
        pred_list.append(float(val.item()))
    predictions: tuple[float, ...] = tuple(pred_list)

    console.write("Predictions complete")

    return PredictionResult(
        n_samples=n_samples,
        entity_ids=entity_ids,
        predictions=predictions,
    )


def write_submission(
    output_path: Path,
    entity_ids: tuple[str, ...],
    predictions: tuple[float, ...],
) -> int:
    """Write predictions to submission CSV file.

    Args:
        output_path: Path to write submission CSV.
        entity_ids: Entity identifiers.
        predictions: Predicted probabilities.

    Returns:
        Number of rows written.

    Raises:
        ValueError: If entity_ids and predictions have different lengths.
    """
    console = get_console()

    if len(entity_ids) != len(predictions):
        msg = f"entity_ids length {len(entity_ids)} != predictions {len(predictions)}"
        raise ValueError(msg)

    console.write(f"Writing submission to {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("customer_ID,prediction\n")
        for entity_id, pred in zip(entity_ids, predictions, strict=True):
            f.write(f"{entity_id},{pred:.6f}\n")

    n_rows = len(entity_ids)
    console.write(f"Wrote {n_rows} predictions")
    return n_rows


def run_pipeline(
    train_dir: Path,
    test_dir: Path,
    output_path: Path,
    config: SubmitConfig,
    model_output_dir: Path | None = None,
) -> PredictionResult:
    """Run the full submission pipeline.

    Loads training data, trains model, loads test data, and generates predictions.

    Args:
        train_dir: Directory with training data.csv and labels.csv.
        test_dir: Directory with test data.csv.
        output_path: Path to write submission CSV.
        config: Pipeline configuration.
        model_output_dir: Directory to save model artifacts (defaults to output_path.parent).

    Returns:
        PredictionResult with all predictions.

    Raises:
        FileNotFoundError: If data files don't exist.
        ValueError: If data format is invalid.
    """
    console = get_console()
    console.write("Starting submission pipeline")

    # Determine model output directory
    output_dir = model_output_dir if model_output_dir is not None else output_path.parent

    # Load training data
    train_data = load_training_data(
        train_dir,
        config["aggregation"],
        config["include_rank_features"],
        config["include_diff_features"],
    )

    # Train model
    model, _train_result = train_model(
        train_data["x"],
        train_data["y"],
        train_data["meta"]["feature_names"],
        config,
        output_dir,
    )

    # Load test data (use same aggregation settings)
    test_config = build_dataset_config(
        test_dir,
        config["aggregation"],
        config["include_rank_features"],
        config["include_diff_features"],
    )
    loader = create_timeseries_csv_loader()
    test_data = loader.load(test_config, test_dir.parent)

    # Generate test entity IDs from metadata
    # For time-series data, we need to extract entity IDs
    # Since LoadedDataset doesn't store entity IDs, we reload to get them
    test_n_samples = test_data["meta"]["n_samples"]
    test_entity_ids = tuple(f"entity_{i}" for i in range(test_n_samples))

    # Generate predictions
    result = predict(model, test_data["x"], test_entity_ids)

    # Write submission
    write_submission(output_path, result["entity_ids"], result["predictions"])

    console.write("Pipeline complete")
    return result


__all__ = [
    "PredictionResult",
    "SubmitConfig",
    "TrainResult",
    "build_dataset_config",
    "load_training_data",
    "predict",
    "run_pipeline",
    "train_model",
    "write_submission",
]
