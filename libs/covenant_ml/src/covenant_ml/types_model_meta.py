"""Persisted model metadata shapes, one per classifier backend."""

from __future__ import annotations

from typing import Literal, TypedDict

from covenant_ml.types import (
    LogRegPenalty,
    LogRegSolver,
)


class MLPModelMeta(TypedDict, total=True):
    """Metadata required to reconstruct an MLP model for inference.

    Stored as JSON alongside the .pt state dict file. Contains only the
    architecture parameters needed to call _build_model() before loading
    the state dict.

    Args:
        backend: Literal discriminator for union type narrowing.
        n_features: Number of input features the model was trained on.
        hidden_sizes: List of hidden layer sizes (JSON doesn't support tuples).
        dropout: Dropout rate used in the model architecture.
    """

    backend: Literal["mlp"]
    n_features: int
    hidden_sizes: list[int]
    dropout: float


class LSTMModelMeta(TypedDict, total=True):
    """Metadata required to reconstruct an LSTM model for inference.

    Stored as JSON alongside the .pt state dict file. Contains the full
    architecture specification needed to rebuild the LSTM network.

    Args:
        backend: Literal discriminator for union type narrowing.
        n_features: Number of input features per time step.
        sequence_length: Number of time steps in each sequence.
        hidden_size: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        bidirectional: Whether LSTM processes sequences in both directions.
        dropout: Dropout rate between LSTM layers.
    """

    backend: Literal["lstm"]
    n_features: int
    sequence_length: int
    hidden_size: int
    num_layers: int
    bidirectional: bool
    dropout: float


class LightGBMModelMeta(TypedDict, total=True):
    """Metadata for LightGBM model.

    LightGBM's .txt format is self-describing, so minimal metadata is needed.
    The backend field enables consistent discriminated union handling.

    Args:
        backend: Literal discriminator for union type narrowing.
    """

    backend: Literal["lightgbm"]


class LogRegModelMeta(TypedDict, total=True):
    """Metadata for Logistic Regression model.

    Stores architecture info for model reconstruction. Logistic regression
    models are saved as joblib files with the full sklearn estimator.

    Args:
        backend: Literal discriminator for union type narrowing.
        n_features: Number of input features the model was trained on.
        penalty: Regularization type used during training.
        solver: Optimization algorithm used.
    """

    backend: Literal["logreg"]
    n_features: int
    penalty: LogRegPenalty
    solver: LogRegSolver


class RandomForestModelMeta(TypedDict, total=True):
    """Metadata for Random Forest model.

    Stores architecture info for model reconstruction. Random forest
    models are saved as joblib files with the full sklearn estimator.

    Args:
        backend: Literal discriminator for union type narrowing.
        n_features: Number of input features the model was trained on.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth (None if unlimited).
    """

    backend: Literal["random_forest"]
    n_features: int
    n_estimators: int
    max_depth: int | None


ModelMeta = (
    MLPModelMeta | LSTMModelMeta | LightGBMModelMeta | LogRegModelMeta | RandomForestModelMeta
)


__all__ = [
    "LSTMModelMeta",
    "LightGBMModelMeta",
    "LogRegModelMeta",
    "MLPModelMeta",
    "ModelMeta",
    "RandomForestModelMeta",
]
