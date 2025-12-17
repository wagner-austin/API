"""Core explanation runner with model loading and explainer execution.

Loads trained models and runs feature importance explanations using the
pluggable explainer registry.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict

import numpy as np
from covenant_ml.explainers.types import ExplainResult, SupportedExplainer
from covenant_ml.features import (
    FeaturePreset,
    engineer_features,
    get_feature_config_for_preset,
)
from covenant_ml.types import BackendName
from numpy.typing import NDArray

import scripts._test_hooks as _hooks
from scripts.explain.cli import DatasetName

# Model extensions for each backend
MODEL_EXTENSIONS: dict[BackendName, str] = {
    "xgboost": "ubj",
    "mlp": "pt",
    "lightgbm": "txt",
    "lstm": "pt",
}


class ExplainRunResult(TypedDict):
    """Result of an explanation run with timing info.

    Bundles the explanation result with timing and configuration
    data for display purposes.
    """

    backend: BackendName
    dataset: DatasetName
    explainer: SupportedExplainer
    result: ExplainResult
    elapsed: float
    model_path: str


def get_project_root() -> Path:
    """Get project root directory (covenant-radar-api service root).

    Returns:
        The absolute path to the service root directory.
    """
    return Path(__file__).parent.parent.parent


def _get_default_model_path(backend: BackendName, dataset: DatasetName) -> Path:
    """Get default model path for backend and dataset.

    Args:
        backend: ML backend type.
        dataset: Dataset name.

    Returns:
        Path to the best saved model file.
    """
    project_root = get_project_root()
    ext = MODEL_EXTENSIONS[backend]
    return project_root / "models" / backend / f"{dataset}_{backend}_best.{ext}"


def _load_dataset_with_features(
    dataset: DatasetName,
    feature_preset: FeaturePreset,
    external_dir: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    """Load dataset and apply feature engineering.

    Args:
        dataset: Dataset name (taiwan, us, polish).
        feature_preset: Feature engineering preset.
        external_dir: Directory containing external datasets.

    Returns:
        Tuple of (features, labels, feature_names).
    """
    registry = _hooks.dataset_registry_factory()
    config = registry.get(dataset)
    loaded = _hooks.dataset_loader(config, external_dir)

    # Apply feature engineering using the preset
    fe_config = get_feature_config_for_preset(feature_preset)
    engineered = engineer_features(
        loaded["x"],
        list(loaded["meta"]["feature_names"]),
        fe_config,
    )

    return engineered["x"], loaded["y"], engineered["feature_names"]


def _sample_data(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    n_samples: int,
    random_state: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Sample a subset of data for explanation.

    Args:
        x: Feature matrix.
        y: Label vector.
        n_samples: Number of samples to select.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (sampled_x, sampled_y).
    """
    n_total = int(x.shape[0])
    if n_samples >= n_total:
        return x, y

    rng = np.random.default_rng(random_state)
    indices = rng.choice(n_total, size=n_samples, replace=False)
    indices_arr: NDArray[np.int64] = np.asarray(indices, dtype=np.int64)
    return x[indices_arr], y[indices_arr]


def run_explanation(
    backend: BackendName,
    dataset: DatasetName,
    explainer: SupportedExplainer,
    model_path: str | None,
    n_samples: int,
    target_class: int,
) -> ExplainRunResult:
    """Run feature importance explanation.

    Args:
        backend: ML backend type (xgboost, mlp, lightgbm, lstm).
        dataset: Dataset name (taiwan, us, polish).
        explainer: Explainer method to use.
        model_path: Path to model file, or None to use default.
        n_samples: Number of samples to use for explanation.
        target_class: Target class for importance computation.

    Returns:
        ExplainRunResult with explanation results and timing.

    Raises:
        FileNotFoundError: If model file doesn't exist.
        ValueError: If explainer is incompatible with backend.
    """
    start_time = time.perf_counter()

    project_root = get_project_root()
    external_dir = project_root / "data" / "external"

    # Resolve model path
    if model_path is not None:
        resolved_path = Path(model_path)
    else:
        resolved_path = _get_default_model_path(backend, dataset)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Model file not found: {resolved_path}")

    # Load dataset with full feature engineering
    # Using "full" preset to match training
    x_features, y_labels, feature_names = _load_dataset_with_features(dataset, "full", external_dir)

    # Sample data (y is unused but needed for consistent sampling)
    x_sampled, _ = _sample_data(x_features, y_labels, n_samples, random_state=42)
    n_samples_used = int(x_sampled.shape[0])
    n_features = int(x_sampled.shape[1])

    # Get explainer registry and explainer instance
    registry = _hooks.explainer_registry_factory()

    # Check compatibility
    if not registry.is_compatible(explainer, backend):
        raise ValueError(f"Explainer '{explainer}' is not compatible with backend '{backend}'")

    # Load model using the model loader hook
    from covenant_radar_api.worker._explain_loaders import load_model_for_backend

    model = load_model_for_backend(
        backend=backend,
        model_path=str(resolved_path),
    )

    # Get explainer instance
    explainer_instance = registry.get(explainer)

    # Compute feature importances
    importances = explainer_instance.compute_importance(
        model=model,
        x_data=x_sampled,
        feature_names=feature_names,
        target_class=target_class,
    )

    elapsed = time.perf_counter() - start_time

    # Build result
    result: ExplainResult = {
        "status": "complete",
        "backend": backend,
        "explainer": explainer,
        "n_samples_used": n_samples_used,
        "n_features": n_features,
        "target_class": target_class,
        "feature_importances": importances,
        "duration_seconds": elapsed,
    }

    return {
        "backend": backend,
        "dataset": dataset,
        "explainer": explainer,
        "result": result,
        "elapsed": elapsed,
        "model_path": str(resolved_path),
    }


__all__ = [
    "MODEL_EXTENSIONS",
    "ExplainRunResult",
    "get_project_root",
    "run_explanation",
]
