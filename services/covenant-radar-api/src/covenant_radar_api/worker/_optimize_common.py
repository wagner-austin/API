"""Shared utilities for hyperparameter optimization jobs.

Contains common parsing, dataset loading, and config building functions
used by backend-specific optimization jobs.

Supports both standard datasets (CSV/ARFF) and time-series datasets.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from covenant_ml.datasets import LoadedDataset
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.features import FeaturePreset
from covenant_ml.optimizer import OptimizationConfig, make_default_optimization_config
from covenant_ml.types import BackendName
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

# Dataset type discriminator
DatasetType = Literal["standard", "timeseries"]


def parse_precision(raw: JSONValue | None) -> Literal["fp32", "fp16", "bf16", "auto"]:
    """Parse precision setting, defaulting to 'fp32'.

    Args:
        raw: Raw JSON value.

    Returns:
        Precision literal.

    Raises:
        JSONTypeError: If value is not a valid precision.
    """
    if raw is None:
        return "fp32"
    if not isinstance(raw, str):
        raise JSONTypeError("precision must be a string")
    if raw == "fp32":
        return "fp32"
    if raw == "fp16":
        return "fp16"
    if raw == "bf16":
        return "bf16"
    if raw == "auto":
        return "auto"
    raise JSONTypeError("precision must be one of: fp32, fp16, bf16, auto")


def parse_nn_optimizer(raw: JSONValue | None) -> Literal["adamw", "adam", "sgd"]:
    """Parse neural network optimizer, defaulting to 'adamw'.

    Args:
        raw: Raw JSON value.

    Returns:
        NN optimizer literal.

    Raises:
        JSONTypeError: If value is not a valid optimizer.
    """
    if raw is None:
        return "adamw"
    if not isinstance(raw, str):
        raise JSONTypeError("optimizer must be a string")
    if raw == "adamw":
        return "adamw"
    if raw == "adam":
        return "adam"
    if raw == "sgd":
        return "sgd"
    raise JSONTypeError("optimizer must be one of: adamw, adam, sgd")


def parse_bidirectional(raw: JSONValue | None) -> bool:
    """Parse bidirectional flag, defaulting to False.

    Args:
        raw: Raw JSON value.

    Returns:
        Boolean value.

    Raises:
        JSONTypeError: If value is not a boolean.
    """
    if raw is None:
        return False
    if not isinstance(raw, bool):
        raise JSONTypeError("bidirectional must be a boolean")
    return raw


def optional_int(data: JSONObject, key: str, default: int) -> int:
    """Extract optional int from dict.

    Args:
        data: JSON object to extract from.
        key: Key to look up.
        default: Default value if key is missing.

    Returns:
        Integer value or default.

    Raises:
        JSONTypeError: If value is present but not a number.
    """
    raw = data.get(key)
    if raw is None:
        return default
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    raise JSONTypeError(f"Field '{key}' must be a number")


def parse_device(raw: JSONValue | None) -> Literal["cpu", "cuda", "auto"]:
    """Parse device setting, defaulting to 'auto'.

    Args:
        raw: Raw JSON value.

    Returns:
        Device literal.

    Raises:
        JSONTypeError: If value is not a string.
        ValueError: If value is not a valid device.
    """
    if raw is None:
        return "auto"
    if not isinstance(raw, str):
        raise JSONTypeError("device must be a string")
    if raw == "cpu":
        return "cpu"
    if raw == "cuda":
        return "cuda"
    if raw == "auto":
        return "auto"
    raise ValueError("device must be one of: cpu, cuda, auto")


def parse_feature_preset(raw: JSONValue | None) -> FeaturePreset:
    """Parse feature preset, defaulting to 'none'.

    Args:
        raw: Raw JSON value.

    Returns:
        FeaturePreset literal.

    Raises:
        JSONTypeError: If value is invalid.
    """
    if raw is None:
        return "none"
    if not isinstance(raw, str):
        raise JSONTypeError("feature_preset must be a string")
    if raw == "none":
        return "none"
    if raw == "log_only":
        return "log_only"
    if raw == "ratios_only":
        return "ratios_only"
    if raw == "full":
        return "full"
    raise JSONTypeError("feature_preset must be one of: none, log_only, ratios_only, full")


def get_dataset_type(dataset: str) -> DatasetType:
    """Determine whether a dataset is standard or time-series.

    Args:
        dataset: Dataset name string.

    Returns:
        DatasetType indicating "standard" or "timeseries".

    Raises:
        ValueError: If dataset name is not in any registry.
    """
    from covenant_radar_api.worker import _test_hooks as hooks

    # Check standard registry first
    standard_registry = hooks.dataset_registry_factory()
    if dataset in standard_registry:
        return "standard"

    # Check time-series registry
    timeseries_registry = hooks.timeseries_registry_factory()
    if dataset in timeseries_registry:
        return "timeseries"

    # Not found in either registry
    standard_names = standard_registry.list_names()
    timeseries_names = timeseries_registry.list_names()
    all_names = sorted(set(standard_names) | set(timeseries_names))
    raise ValueError(f"dataset must be one of: {', '.join(all_names)} (got {dataset})")


def parse_dataset_name(dataset: str) -> str:
    """Parse and validate dataset name against both registries.

    Checks standard registry first, then time-series registry.

    Args:
        dataset: Dataset name string.

    Returns:
        Validated dataset name from registry.

    Raises:
        ValueError: If dataset name is not in any registry.
    """
    from covenant_radar_api.worker import _test_hooks as hooks

    # Check standard registry
    standard_registry = hooks.dataset_registry_factory()
    if dataset in standard_registry:
        return dataset

    # Check time-series registry
    timeseries_registry = hooks.timeseries_registry_factory()
    if dataset in timeseries_registry:
        return dataset

    # Not found - build error message with all available datasets
    standard_names = standard_registry.list_names()
    timeseries_names = timeseries_registry.list_names()
    all_names = sorted(set(standard_names) | set(timeseries_names))
    raise ValueError(f"dataset must be one of: {', '.join(all_names)} (got {dataset})")


def parse_backend_name(raw: JSONValue | None) -> BackendName:
    """Parse backend name, defaulting to 'xgboost'.

    Args:
        raw: Raw JSON value.

    Returns:
        BackendName literal.

    Raises:
        JSONTypeError: If value is not a string.
        ValueError: If value is not a valid backend.
    """
    if raw is None:
        return "xgboost"
    if not isinstance(raw, str):
        raise JSONTypeError("backend must be a string")
    if raw == "xgboost":
        return "xgboost"
    if raw == "mlp":
        return "mlp"
    if raw == "lstm":
        return "lstm"
    if raw == "lightgbm":
        return "lightgbm"
    if raw == "cleargbm":
        return "cleargbm"
    if raw == "logreg":
        return "logreg"
    if raw == "random_forest":
        return "random_forest"
    raise ValueError(
        "backend must be one of: xgboost, mlp, lstm, lightgbm, cleargbm, logreg, random_forest"
    )


def load_dataset(
    dataset_name: str,
    external_dir: Path,
    progress_callback: ProgressCallbackProtocol | None = None,
) -> LoadedDataset:
    """Load the specified standard dataset using registry and pluggable loader.

    Args:
        dataset_name: Name of dataset in standard registry.
        external_dir: Path to data/external directory.
        progress_callback: Optional callback for loading progress updates.

    Returns:
        LoadedDataset with feature matrix, labels, and metadata.

    Raises:
        KeyError: If dataset not in registry.
        FileNotFoundError: If dataset file doesn't exist.
        ValueError: If data doesn't match expected format.
    """
    from covenant_radar_api.worker import _test_hooks as hooks

    registry = hooks.dataset_registry_factory()
    config = registry.get(dataset_name)
    return hooks.dataset_loader(config, external_dir, progress_callback)


def load_timeseries_dataset(
    dataset_name: str,
    external_dir: Path,
    progress_callback: ProgressCallbackProtocol | None = None,
) -> LoadedDataset:
    """Load the specified time-series dataset using registry and pluggable loader.

    Time-series datasets contain multiple observations per entity over time.
    The loader aggregates them into single feature vectors for ML.

    Args:
        dataset_name: Name of dataset in time-series registry.
        external_dir: Path to data/external directory.
        progress_callback: Optional callback for loading progress updates.

    Returns:
        LoadedDataset with aggregated feature matrix, labels, and metadata.

    Raises:
        KeyError: If dataset not in time-series registry.
        FileNotFoundError: If dataset file doesn't exist.
        ValueError: If data doesn't match expected format.
    """
    from covenant_radar_api.worker import _test_hooks as hooks

    registry = hooks.timeseries_registry_factory()
    config = registry.get(dataset_name)
    return hooks.timeseries_loader(config, external_dir, progress_callback)


def load_any_dataset(
    dataset_name: str,
    external_dir: Path,
    progress_callback: ProgressCallbackProtocol | None = None,
) -> LoadedDataset:
    """Load a dataset from either standard or time-series registry.

    Automatically detects whether the dataset is standard or time-series
    and uses the appropriate loader.

    Args:
        dataset_name: Name of dataset in any registry.
        external_dir: Path to data/external directory.
        progress_callback: Optional callback for loading progress updates.

    Returns:
        LoadedDataset with feature matrix, labels, and metadata.

    Raises:
        ValueError: If dataset not in any registry.
        FileNotFoundError: If dataset file doesn't exist.
    """
    dataset_type = get_dataset_type(dataset_name)

    if dataset_type == "standard":
        return load_dataset(dataset_name, external_dir, progress_callback)

    return load_timeseries_dataset(dataset_name, external_dir, progress_callback)


def build_optimization_config(
    n_trials: int,
    timeout_seconds: int | None,
    random_state: int,
) -> OptimizationConfig:
    """Build optimization config with standard train/val/test splits.

    Args:
        n_trials: Number of optimization trials.
        timeout_seconds: Optional timeout in seconds.
        random_state: Random seed for reproducibility.

    Returns:
        OptimizationConfig with standard settings.
    """
    return make_default_optimization_config(
        n_trials=n_trials,
        timeout_seconds=timeout_seconds,
        random_state=random_state,
    )


def save_optimization_results(
    output_dir: Path,
    dataset_name: str,
    backend_name: str,
    result_dict: dict[str, JSONValue],
    config_dict: dict[str, JSONValue],
) -> tuple[Path, Path]:
    """Save optimization results and config to JSON files.

    Args:
        output_dir: Directory to save files to.
        dataset_name: Name of the dataset.
        backend_name: Name of the backend (lstm, mlp, etc).
        result_dict: Dictionary with optimization results.
        config_dict: Dictionary with recommended config.

    Returns:
        Tuple of (result_path, config_path).
    """
    from platform_core.json_utils import dump_json_str

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{dataset_name}_{backend_name}_optuna_result.json"
    config_path = output_dir / f"{dataset_name}_{backend_name}_optimal_config.json"

    with open(result_path, "w") as f:
        f.write(dump_json_str(result_dict))

    with open(config_path, "w") as f:
        f.write(dump_json_str(config_dict))

    return result_path, config_path


def load_dataset_with_progress(
    dataset_name: str,
    external_dir: Path,
    progress_callback: ProgressCallbackProtocol | None,
) -> LoadedDataset:
    """Load dataset with optional progress reporting.

    This helper wraps load_any_dataset and only passes the callback
    when it's not None, reducing complexity in calling functions.

    Args:
        dataset_name: Name of dataset in registry.
        external_dir: Path to data/external directory.
        progress_callback: Optional callback for loading progress.

    Returns:
        LoadedDataset with feature matrix, labels, and metadata.
    """
    return load_any_dataset(
        dataset_name,
        external_dir,
        progress_callback,
    )


__all__ = [
    "DatasetType",
    "LoadedDataset",
    "ProgressCallbackProtocol",
    "build_optimization_config",
    "get_dataset_type",
    "load_any_dataset",
    "load_dataset",
    "load_timeseries_dataset",
    "optional_int",
    "parse_backend_name",
    "parse_bidirectional",
    "parse_dataset_name",
    "parse_device",
    "parse_feature_preset",
    "parse_nn_optimizer",
    "parse_precision",
    "save_optimization_results",
]
