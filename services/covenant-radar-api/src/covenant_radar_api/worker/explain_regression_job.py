"""Background job for regression feature importance explanation.

Computes feature importances using pluggable explainers on trained regressor
models. Supports tree-based regressors (XGBoost, LightGBM) with permutation
and SHAP explainers, plus neural regressors (MLP, LSTM) with gradient-based
explainers.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, Protocol, TypedDict

import numpy as np
from covenant_ml.explainers.regression_registry import RegressionExplainerRegistry
from covenant_ml.explainers.types import RegressionExplainResult, SupportedExplainer
from covenant_ml.types import RegressorBackendName
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
from covenant_radar_api.worker._regression_hooks import (
    regression_dataset_loader,
    regression_explainer_registry_factory,
    regression_registry_factory,
    regressor_registry_factory,
)

_log = get_logger(__name__)

# Progress status type for regression explainer job
RegressionExplainJobStatus = Literal[
    "started",
    "loading_model",
    "loading_data",
    "computing",
    "complete",
]


# ---------------------------------------------------------------------------
# Config Parsing
# ---------------------------------------------------------------------------


class RegressionExplainParseResult(TypedDict, total=True):
    """Parsed regression explanation request configuration.

    Args:
        dataset: Regression dataset name.
        backend: Regressor backend name.
        model_path: Path to saved regressor model file.
        explainer: Which explainer to use.
        n_samples: Number of samples for explanation.
        random_state: Random seed for reproducibility.
    """

    dataset: str
    backend: RegressorBackendName
    model_path: str
    explainer: SupportedExplainer
    n_samples: int
    random_state: int


def _optional_int(data: JSONObject, key: str, default: int) -> int:
    """Extract optional int from JSON, raising on wrong type.

    Args:
        data: JSON object.
        key: Key to extract.
        default: Default value if key not present.

    Returns:
        Integer value.

    Raises:
        JSONTypeError: If value exists but is not an integer.
    """
    raw = data.get(key)
    if raw is None:
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    raise JSONTypeError(f"Field '{key}' must be a number")


def _parse_regressor_backend(raw: JSONValue) -> RegressorBackendName:
    """Parse and validate regressor backend name.

    Args:
        raw: Raw JSON value.

    Returns:
        Validated RegressorBackendName literal.

    Raises:
        JSONTypeError: If value is not a valid regressor backend name.
    """
    if not isinstance(raw, str):
        raise JSONTypeError("backend must be a string")
    if raw == "xgboost_reg":
        return "xgboost_reg"
    if raw == "lightgbm_reg":
        return "lightgbm_reg"
    if raw == "mlp_reg":
        return "mlp_reg"
    if raw == "lstm_reg":
        return "lstm_reg"
    raise JSONTypeError("backend must be one of: xgboost_reg, lightgbm_reg, mlp_reg, lstm_reg")


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


def _parse_regression_explain_config(
    config_json: str,
) -> RegressionExplainParseResult:
    """Parse regression explanation config from JSON string.

    Args:
        config_json: JSON configuration string.

    Returns:
        RegressionExplainParseResult with all explanation parameters.

    Raises:
        JSONTypeError: If config is invalid.
    """
    raw = load_json_str(config_json)
    if not isinstance(raw, dict):
        raise JSONTypeError("config must be a JSON object")

    raw_obj: JSONObject = raw

    dataset = require_str(raw_obj, "dataset")

    backend_raw = raw_obj.get("backend")
    if backend_raw is None:
        raise JSONTypeError("backend is required")
    backend = _parse_regressor_backend(backend_raw)

    model_path = require_str(raw_obj, "model_path")

    explainer_raw = raw_obj.get("explainer")
    if explainer_raw is None:
        raise JSONTypeError("explainer is required")
    explainer = _parse_explainer(explainer_raw)

    n_samples = _optional_int(raw_obj, "n_samples", 1000)
    random_state = _optional_int(raw_obj, "random_state", 42)

    return RegressionExplainParseResult(
        dataset=dataset,
        backend=backend,
        model_path=model_path,
        explainer=explainer,
        n_samples=n_samples,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Data Sampling
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


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


class RegressionExplainProgressInfo(TypedDict):
    """Progress information for regression explanation computation.

    Args:
        status: Current job status.
        elapsed_seconds: Seconds since job started.
    """

    status: RegressionExplainJobStatus
    elapsed_seconds: float


class RegressionExplainProgressCallbackProtocol(Protocol):
    """Protocol for regression explanation progress callback."""

    def __call__(self, info: RegressionExplainProgressInfo) -> None:
        """Called with progress updates.

        Args:
            info: Progress information.
        """
        ...


# ---------------------------------------------------------------------------
# Explanation Execution
# ---------------------------------------------------------------------------


def run_regression_explanation(
    config_json: str,
    external_dir: Path,
    models_root: Path,
    registry: RegressionExplainerRegistry | None = None,
    progress_callback: RegressionExplainProgressCallbackProtocol | None = None,
) -> RegressionExplainResult:
    """Run regression feature importance explanation.

    Args:
        config_json: JSON config with dataset, backend, model_path, explainer.
        external_dir: Path to data/external directory with datasets.
        models_root: Directory the caller-supplied model_path must resolve under.
        registry: Optional explainer registry (uses hook factory if None).
        progress_callback: Optional callback for progress updates.

    Returns:
        RegressionExplainResult with feature importances.

    Raises:
        ValueError: If explainer incompatible with backend, or if model_path
            resolves outside models_root.
        FileNotFoundError: If model or dataset not found.
    """
    start_time = time.monotonic()

    def _report_progress(status: RegressionExplainJobStatus) -> None:
        if progress_callback is not None:
            elapsed = time.monotonic() - start_time
            info: RegressionExplainProgressInfo = {
                "status": status,
                "elapsed_seconds": elapsed,
            }
            progress_callback(info)

    _report_progress("started")

    # Parse config
    parse_result = _parse_regression_explain_config(config_json)
    dataset_name = parse_result["dataset"]
    backend = parse_result["backend"]
    # Confine the caller-supplied path before it reaches any loader.
    model_path = str(resolve_model_path(parse_result["model_path"], models_root))
    explainer_name = parse_result["explainer"]
    n_samples = parse_result["n_samples"]
    random_state = parse_result["random_state"]

    # Get explainer registry
    reg = registry if registry is not None else regression_explainer_registry_factory()

    # Validate compatibility
    if not reg.is_compatible(explainer_name, backend):
        compatible = reg.list_compatible_explainers(backend)
        raise ValueError(
            f"Explainer '{explainer_name}' is not compatible with "
            f"backend '{backend}'. Compatible explainers: {compatible}"
        )

    _log.info(
        "Starting regression feature importance explanation",
        extra={
            "dataset": dataset_name,
            "backend": backend,
            "explainer": explainer_name,
            "n_samples": n_samples,
        },
    )

    # Load model via regressor backend registry hook
    _report_progress("loading_model")
    backend_reg = regressor_registry_factory()
    backend_impl = backend_reg.get(backend)
    model = backend_impl.load(path=model_path)

    # Load regression dataset via dataset registry hook
    _report_progress("loading_data")
    ds_registry = regression_registry_factory()
    ds_config = ds_registry.get(dataset_name)
    loaded = regression_dataset_loader(ds_config, external_dir)
    x_full: NDArray[np.float64] = loaded["x"]
    x_sampled = _sample_data(x_full, n_samples, random_state)
    n_samples_used = int(x_sampled.shape[0])
    n_features = int(x_sampled.shape[1])

    feature_names: list[str] = list(loaded["meta"]["feature_names"])

    _log.info(
        "Regression data loaded and sampled",
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
    )

    elapsed = time.monotonic() - start_time
    _report_progress("complete")

    _log.info(
        "Regression explanation complete",
        extra={
            "dataset": dataset_name,
            "explainer": explainer_name,
            "n_samples_used": n_samples_used,
            "duration_seconds": f"{elapsed:.2f}",
        },
    )

    return RegressionExplainResult(
        status="complete",
        backend=backend,
        explainer=explainer_name,
        n_samples_used=n_samples_used,
        n_features=n_features,
        feature_importances=importances,
        duration_seconds=elapsed,
    )


def process_regression_explain_job(
    config_json: str,
) -> dict[str, JSONValue]:
    """RQ job entry point for regression feature importance explanation.

    Args:
        config_json: JSON config with explanation parameters.

    Returns:
        RegressionExplainResult as JSON-serializable dict.
    """
    from covenant_radar_api.core.config import settings_from_env

    settings = settings_from_env()

    data_root = Path(settings["app"]["data_root"])
    external_dir = data_root / "external"
    models_root = Path(settings["app"]["models_root"])

    result = run_regression_explanation(config_json, external_dir, models_root)

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
        "feature_importances": importances_json,
        "duration_seconds": result["duration_seconds"],
    }


__all__ = [
    "RegressionExplainParseResult",
    "RegressionExplainProgressCallbackProtocol",
    "RegressionExplainProgressInfo",
    "process_regression_explain_job",
    "run_regression_explanation",
]
