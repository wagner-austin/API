"""Shared fixtures and helpers for test_explain_script splits."""

from __future__ import annotations

from typing import Literal

import numpy as np
from covenant_ml.datasets import DatasetConfig, DatasetMeta, DatasetRegistry, LoadedDataset
from covenant_ml.explainers.registry import ExplainerRegistration, ExplainerRegistry
from covenant_ml.explainers.types import ExplainResult, SupportedExplainer
from covenant_ml.types import BackendName
from numpy.typing import NDArray
from platform_ml.explainers.protocol import FeatureExplainer, PredictorProtocol
from platform_ml.explainers.types import (
    ComputationalCost,
    ExplainerCapabilities,
    ExplainerName,
    FeatureImportanceScore,
)
from scripts.explain.runner import (
    ExplainRunResult,
)

DatasetNameLiteral = Literal["taiwan", "us", "polish"]


def _make_fake_feature_importances(n_features: int = 10) -> list[FeatureImportanceScore]:
    """Create fake feature importance scores for testing."""
    return [
        {"name": f"feature_{i}", "importance": 1.0 - (i * 0.1), "rank": i + 1}
        for i in range(n_features)
    ]


def _make_fake_explain_result(
    backend: BackendName = "xgboost",
    explainer: SupportedExplainer = "permutation",
    n_samples: int = 100,
    n_features: int = 10,
) -> ExplainResult:
    """Create a fake ExplainResult for testing."""
    return {
        "status": "complete",
        "backend": backend,
        "explainer": explainer,
        "n_samples_used": n_samples,
        "n_features": n_features,
        "target_class": 1,
        "feature_importances": _make_fake_feature_importances(n_features),
        "duration_seconds": 1.5,
    }


def _make_fake_run_result(
    backend: BackendName = "xgboost",
    dataset: DatasetNameLiteral = "taiwan",
    explainer: SupportedExplainer = "permutation",
) -> ExplainRunResult:
    """Create a fake ExplainRunResult for testing."""
    return {
        "backend": backend,
        "dataset": dataset,
        "explainer": explainer,
        "result": _make_fake_explain_result(backend, explainer),
        "elapsed": 1.5,
        "model_path": "/path/to/model.ubj",
    }


class FakePredictor:
    """Fake predictor for testing."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return fake probabilities."""
        n_samples = int(x.shape[0])
        proba: NDArray[np.float64] = np.column_stack(
            [np.zeros(n_samples), np.ones(n_samples)]
        ).astype(np.float64)
        return proba


class FakeExplainer:
    """Fake explainer for testing."""

    def __init__(self, name: SupportedExplainer = "permutation") -> None:
        """Initialize with explainer name."""
        self._name = name

    def explainer_name(self) -> ExplainerName:
        """Return the explainer name."""
        return "permutation"

    def capabilities(self) -> ExplainerCapabilities:
        """Return capabilities."""
        cost: ComputationalCost = "medium"
        return {
            "requires_gradients": False,
            "requires_background_data": False,
            "computational_cost": cost,
        }

    def compute_importance(
        self,
        *,
        model: PredictorProtocol,
        x_data: NDArray[np.float64],
        feature_names: list[str],
        target_class: int,
    ) -> list[FeatureImportanceScore]:
        """Return fake importances."""
        return [
            {"name": name, "importance": 1.0 / (i + 1), "rank": i + 1}
            for i, name in enumerate(feature_names)
        ]


def _make_fake_dataset() -> LoadedDataset:
    """Create fake dataset for testing."""
    rng = np.random.default_rng(42)
    x: NDArray[np.float64] = rng.random((200, 10)).astype(np.float64)
    y: NDArray[np.int64] = rng.integers(0, 2, size=200).astype(np.int64)
    n_positive = int(np.sum(y))
    meta: DatasetMeta = {
        "name": "fake",
        "n_samples": 200,
        "n_features": 10,
        "n_positive": n_positive,
        "n_negative": 200 - n_positive,
        "positive_ratio": float(n_positive) / 200.0,
        "feature_names": tuple(f"feature_{i}" for i in range(10)),
        "categorical_encodings": (),
    }
    return {"x": x, "y": y, "meta": meta, "groups": None}


def _make_fake_dataset_config(name: str) -> DatasetConfig:
    """Create fake dataset config for testing."""
    return {
        "name": name,
        "display_name": f"Fake {name}",
        "folder": f"{name}_data",
        "file_name": "data.csv",
        "file_format": "csv",
        "encoding": "utf-8",
        "target": {
            "column_name": "target",
            "label_type": "binary_int",
            "positive_values": (1,),
            "negative_values": (0,),
        },
        "exclude_columns": (),
        "n_samples_expected": 200,
        "n_features_expected": 10,
        "positive_class_ratio_expected": 0.5,
    }


def _make_fake_dataset_registry() -> DatasetRegistry:
    """Create fake dataset registry for testing."""
    configs = (
        _make_fake_dataset_config("taiwan"),
        _make_fake_dataset_config("us"),
        _make_fake_dataset_config("polish"),
    )
    return DatasetRegistry(configs)


def _make_fake_explainer_registry() -> ExplainerRegistry:
    """Create fake explainer registry for testing."""
    registry = ExplainerRegistry()

    def make_fake() -> FeatureExplainer:
        return FakeExplainer()

    # Register with proper ExplainerRegistration
    backends: frozenset[BackendName] = frozenset(["xgboost", "lightgbm", "mlp", "lstm"])
    registration = ExplainerRegistration(
        factory=make_fake,
        compatible_backends=backends,
        requires_gradients=False,
    )
    registry.register("permutation", registration)
    registry.register("shap_tree", registration)
    registry.register("gradient", registration)
    registry.register("integrated_gradients", registration)
    return registry
