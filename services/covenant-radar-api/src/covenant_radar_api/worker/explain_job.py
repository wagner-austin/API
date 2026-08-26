"""Background job for feature importance explanation.

Computes feature importances using pluggable explainers on trained models.
Supports XGBoost and LightGBM backends with permutation and SHAP explainers,
plus MLP/LSTM with gradient-based explainers.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, Protocol, TypedDict

import numpy as np
from covenant_ml.explainers.registry import ExplainerRegistry, default_explainer_registry
from covenant_ml.explainers.types import ExplainResult, SupportedExplainer
from covenant_ml.types import BackendName
from numpy.typing import NDArray
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    load_json_str,
    require_str,
)
from platform_core.logging import get_logger
from platform_ml.explainers.types import FeatureImportanceScore

from covenant_radar_api.core.model_paths import resolve_model_path
from covenant_radar_api.worker._explain_loaders import (
    LSTMModelConfig,
    MLPModelConfig,
    load_model_for_backend,
)
from covenant_radar_api.worker._optimize_common import (
    load_any_dataset,
    optional_int,
    parse_backend_name,
    parse_dataset_name,
)

_log = get_logger(__name__)

# Progress status type for explainer job
ExplainJobStatus = Literal["started", "loading_model", "loading_data", "computing", "complete"]


# ---------------------------------------------------------------------------
# Config Parsing
# ---------------------------------------------------------------------------


class ExplainParseResult(TypedDict, total=True):
    """Parsed explanation request configuration."""

    dataset: str
    backend: BackendName
    model_path: str
    explainer: SupportedExplainer
    target_class: int
    n_samples: int
    random_state: int
    mlp_config: MLPModelConfig | None
    lstm_config: LSTMModelConfig | None


def _require_int_field(data: JSONObject, key: str) -> int:
    """Extract required int from JSON object.

    Args:
        data: JSON object.
        key: Key to extract.

    Returns:
        Integer value.

    Raises:
        JSONTypeError: If key is missing or not an integer.
    """
    raw = data.get(key)
    if raw is None:
        raise JSONTypeError(f"Field '{key}' is required")
    if not isinstance(raw, int):
        raise JSONTypeError(f"Field '{key}' must be an integer")
    return raw


def _require_float_field(data: JSONObject, key: str) -> float:
    """Extract required float from JSON object.

    Args:
        data: JSON object.
        key: Key to extract.

    Returns:
        Float value.

    Raises:
        JSONTypeError: If key is missing or not a number.
    """
    raw = data.get(key)
    if raw is None:
        raise JSONTypeError(f"Field '{key}' is required")
    if isinstance(raw, int):
        return float(raw)
    if isinstance(raw, float):
        return raw
    raise JSONTypeError(f"Field '{key}' must be a number")


def _require_bool_field(data: JSONObject, key: str) -> bool:
    """Extract required bool from JSON object.

    Args:
        data: JSON object.
        key: Key to extract.

    Returns:
        Boolean value.

    Raises:
        JSONTypeError: If key is missing or not a boolean.
    """
    raw = data.get(key)
    if raw is None:
        raise JSONTypeError(f"Field '{key}' is required")
    if not isinstance(raw, bool):
        raise JSONTypeError(f"Field '{key}' must be a boolean")
    return raw


def _parse_int_tuple(raw: JSONValue, field_name: str) -> tuple[int, ...]:
    """Parse JSON array to tuple of ints.

    Args:
        raw: JSON array value.
        field_name: Name of field for error messages.

    Returns:
        Tuple of integers.

    Raises:
        JSONTypeError: If not a valid array of integers.
    """
    if not isinstance(raw, list):
        raise JSONTypeError(f"Field '{field_name}' must be an array of integers")
    result: list[int] = []
    for item in raw:
        if not isinstance(item, int):
            raise JSONTypeError(f"Field '{field_name}' must contain only integers")
        result.append(item)
    return tuple(result)


def _parse_mlp_config(raw: JSONObject) -> MLPModelConfig:
    """Parse MLP model configuration from JSON object.

    Args:
        raw: JSON object with mlp_config fields.

    Returns:
        MLPModelConfig TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    n_features = _require_int_field(raw, "n_features")
    dropout = _require_float_field(raw, "dropout")

    hidden_raw = raw.get("hidden_sizes")
    if hidden_raw is None:
        raise JSONTypeError("Field 'hidden_sizes' is required in mlp_config")
    hidden_sizes = _parse_int_tuple(hidden_raw, "hidden_sizes")

    return MLPModelConfig(
        n_features=n_features,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    )


def _parse_lstm_config(raw: JSONObject) -> LSTMModelConfig:
    """Parse LSTM model configuration from JSON object.

    Args:
        raw: JSON object with lstm_config fields.

    Returns:
        LSTMModelConfig TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    n_features = _require_int_field(raw, "n_features")
    hidden_size = _require_int_field(raw, "hidden_size")
    num_layers = _require_int_field(raw, "num_layers")
    dropout = _require_float_field(raw, "dropout")
    bidirectional = _require_bool_field(raw, "bidirectional")
    sequence_length = _require_int_field(raw, "sequence_length")

    return LSTMModelConfig(
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        sequence_length=sequence_length,
    )


def _parse_explainer(raw: JSONValue) -> SupportedExplainer:
    """Parse and validate explainer name.

    Args:
        raw: Raw JSON value.

    Returns:
        Validated SupportedExplainer literal.

    Raises:
        JSONTypeError: If value is not a valid explainer name.
    """
    if not isinstance(raw, str):
        raise JSONTypeError("explainer must be a string")
    if raw == "permutation":
        return "permutation"
    if raw == "gradient":
        return "gradient"
    if raw == "integrated_gradients":
        return "integrated_gradients"
    if raw == "shap_tree":
        return "shap_tree"
    raise JSONTypeError(
        "explainer must be one of: permutation, gradient, integrated_gradients, shap_tree"
    )


def _parse_explain_config(config_json: str) -> ExplainParseResult:
    """Parse explanation config from JSON string.

    Args:
        config_json: JSON configuration string.

    Returns:
        ExplainParseResult with all explanation parameters.

    Raises:
        JSONTypeError: If config is invalid.
        ValueError: If dataset/backend/explainer names are invalid.
    """
    raw = load_json_str(config_json)
    if not isinstance(raw, dict):
        raise JSONTypeError("config must be a JSON object")

    # Cast to JSONObject for type safety with require_* functions
    raw_obj: JSONObject = raw

    # Required fields
    dataset_str = require_str(raw_obj, "dataset")
    dataset = parse_dataset_name(dataset_str)

    backend_raw = raw_obj.get("backend")
    backend = parse_backend_name(backend_raw)

    model_path = require_str(raw_obj, "model_path")

    explainer_raw = raw_obj.get("explainer")
    if explainer_raw is None:
        raise JSONTypeError("explainer is required")
    explainer = _parse_explainer(explainer_raw)

    target_class = optional_int(raw_obj, "target_class", 1)
    n_samples = optional_int(raw_obj, "n_samples", 1000)
    random_state = optional_int(raw_obj, "random_state", 42)

    # Backend-specific configs
    mlp_config: MLPModelConfig | None = None
    lstm_config: LSTMModelConfig | None = None

    if backend == "mlp":
        mlp_config_raw = raw_obj.get("mlp_config")
        if mlp_config_raw is None:
            raise JSONTypeError("mlp_config is required when backend is 'mlp'")
        if not isinstance(mlp_config_raw, dict):
            raise JSONTypeError("mlp_config must be an object")
        mlp_config = _parse_mlp_config(mlp_config_raw)

    if backend == "lstm":
        lstm_config_raw = raw_obj.get("lstm_config")
        if lstm_config_raw is None:
            raise JSONTypeError("lstm_config is required when backend is 'lstm'")
        if not isinstance(lstm_config_raw, dict):
            raise JSONTypeError("lstm_config must be an object")
        lstm_config = _parse_lstm_config(lstm_config_raw)

    return ExplainParseResult(
        dataset=dataset,
        backend=backend,
        model_path=model_path,
        explainer=explainer,
        target_class=target_class,
        n_samples=n_samples,
        random_state=random_state,
        mlp_config=mlp_config,
        lstm_config=lstm_config,
    )


# ---------------------------------------------------------------------------
# Explanation Execution
# ---------------------------------------------------------------------------


def _sample_data(
    x: NDArray[np.float64],
    n_samples: int,
    random_state: int,
) -> NDArray[np.float64]:
    """Sample data for explanation.

    Args:
        x: Full feature matrix with shape (n_total, n_features).
        n_samples: Number of samples to select.
        random_state: Random seed for reproducibility.

    Returns:
        Sampled feature matrix with shape (min(n_samples, n_total), n_features).
    """
    n_total = int(x.shape[0])
    if n_samples >= n_total:
        return x

    rng = np.random.default_rng(random_state)
    indices = rng.choice(n_total, size=n_samples, replace=False)
    indices_sorted: NDArray[np.int64] = np.sort(indices)
    return x[indices_sorted]


class ExplainProgressInfo(TypedDict):
    """Progress information for explanation computation."""

    status: Literal["started", "loading_model", "loading_data", "computing", "complete"]
    elapsed_seconds: float


class ExplainProgressCallbackProtocol(Protocol):
    """Protocol for explanation progress callback."""

    def __call__(self, info: ExplainProgressInfo) -> None:
        """Called with progress updates."""
        ...


def run_explanation(
    config_json: str,
    external_dir: Path,
    models_root: Path,
    registry: ExplainerRegistry | None = None,
    progress_callback: ExplainProgressCallbackProtocol | None = None,
) -> ExplainResult:
    """Run feature importance explanation.

    Args:
        config_json: JSON config with dataset, backend, model_path, explainer settings.
        external_dir: Path to data/external directory with datasets.
        models_root: Directory the caller-supplied model_path must resolve under.
        registry: Optional explainer registry (uses default if None).
        progress_callback: Optional callback for progress updates.

    Returns:
        ExplainResult with feature importances.

    Raises:
        ValueError: If explainer incompatible with backend, or if model_path
            resolves outside models_root.
        FileNotFoundError: If model or dataset not found.
    """
    start_time = time.monotonic()

    def _report_progress(status: ExplainJobStatus) -> None:
        if progress_callback is not None:
            elapsed = time.monotonic() - start_time
            info: ExplainProgressInfo = {
                "status": status,
                "elapsed_seconds": elapsed,
            }
            progress_callback(info)

    _report_progress("started")

    # Parse config
    parse_result = _parse_explain_config(config_json)
    dataset_name = parse_result["dataset"]
    backend = parse_result["backend"]
    # Confine the caller-supplied path before it reaches any loader.
    model_path = str(resolve_model_path(parse_result["model_path"], models_root))
    explainer_name = parse_result["explainer"]
    target_class = parse_result["target_class"]
    n_samples = parse_result["n_samples"]
    random_state = parse_result["random_state"]
    mlp_config = parse_result["mlp_config"]
    lstm_config = parse_result["lstm_config"]

    # Get registry
    reg = registry if registry is not None else default_explainer_registry()

    # Validate compatibility
    if not reg.is_compatible(explainer_name, backend):
        compatible = reg.list_compatible_explainers(backend)
        raise ValueError(
            f"Explainer '{explainer_name}' is not compatible with backend '{backend}'. "
            f"Compatible explainers: {compatible}"
        )

    _log.info(
        "Starting feature importance explanation",
        extra={
            "dataset": dataset_name,
            "backend": backend,
            "explainer": explainer_name,
            "target_class": target_class,
            "n_samples": n_samples,
        },
    )

    # Load model
    _report_progress("loading_model")
    model = load_model_for_backend(backend, model_path, mlp_config, lstm_config)

    # Load and sample data
    _report_progress("loading_data")
    dataset = load_any_dataset(dataset_name, external_dir)
    x_full: NDArray[np.float64] = dataset["x"]
    x_sampled = _sample_data(x_full, n_samples, random_state)
    n_samples_used = int(x_sampled.shape[0])
    n_features = int(x_sampled.shape[1])

    # Get feature names
    feature_names: list[str] = list(dataset["meta"]["feature_names"])

    _log.info(
        "Data loaded and sampled",
        extra={
            "n_samples_total": int(x_full.shape[0]),
            "n_samples_used": n_samples_used,
            "n_features": n_features,
        },
    )

    # Run explainer
    _report_progress("computing")
    explainer = reg.get(explainer_name)
    importances: list[FeatureImportanceScore] = explainer.compute_importance(
        model=model,
        x_data=x_sampled,
        feature_names=feature_names,
        target_class=target_class,
    )

    elapsed = time.monotonic() - start_time
    _report_progress("complete")

    _log.info(
        "Explanation complete",
        extra={
            "dataset": dataset_name,
            "explainer": explainer_name,
            "n_samples_used": n_samples_used,
            "duration_seconds": f"{elapsed:.2f}",
        },
    )

    return ExplainResult(
        status="complete",
        backend=backend,
        explainer=explainer_name,
        n_samples_used=n_samples_used,
        n_features=n_features,
        target_class=target_class,
        feature_importances=importances,
        duration_seconds=elapsed,
    )


def process_explain_job(config_json: str) -> dict[str, JSONValue]:
    """RQ job entry point for feature importance explanation.

    Args:
        config_json: JSON config with explanation parameters.

    Returns:
        ExplainResult as JSON-serializable dict.
    """
    from covenant_radar_api.core.config import load_settings

    settings = load_settings()

    # Get data directory from settings
    data_root = Path(settings["app"]["data_root"])
    external_dir = data_root / "external"
    models_root = Path(settings["app"]["models_root"])

    result = run_explanation(config_json, external_dir, models_root)

    # Convert FeatureImportanceScore list to JSON-serializable format
    importances_json: list[JSONValue] = []
    for score in result["feature_importances"]:
        score_dict: dict[str, JSONValue] = {
            "name": score["name"],
            "importance": score["importance"],
            "rank": score["rank"],
        }
        importances_json.append(score_dict)

    return {
        "status": result["status"],
        "backend": result["backend"],
        "explainer": result["explainer"],
        "n_samples_used": result["n_samples_used"],
        "n_features": result["n_features"],
        "target_class": result["target_class"],
        "feature_importances": importances_json,
        "duration_seconds": result["duration_seconds"],
    }


__all__ = [
    "ExplainParseResult",
    "ExplainProgressCallbackProtocol",
    "ExplainProgressInfo",
    "ExplainResult",
    "LSTMModelConfig",
    "MLPModelConfig",
    "load_model_for_backend",
    "process_explain_job",
    "run_explanation",
]
