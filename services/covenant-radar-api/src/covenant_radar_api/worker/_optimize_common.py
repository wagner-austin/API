"""Shared utilities for hyperparameter optimization jobs.

Contains common parsing, dataset loading, and config building functions
used by backend-specific optimization jobs.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from covenant_ml.datasets import LoadedDataset
from covenant_ml.features import FeaturePreset
from covenant_ml.optimizer import OptimizationConfig, make_default_optimization_config
from covenant_ml.types import BackendName
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue


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


def parse_dataset_name(dataset: str) -> str:
    """Parse and validate dataset name against registry.

    Args:
        dataset: Dataset name string.

    Returns:
        Validated dataset name from registry.

    Raises:
        ValueError: If dataset name is not in registry.
    """
    from covenant_radar_api.worker import _test_hooks as hooks

    registry = hooks.dataset_registry_factory()
    if dataset not in registry:
        available = ", ".join(registry.list_names())
        raise ValueError(f"dataset must be one of: {available} (got {dataset})")
    return dataset


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
    raise ValueError("backend must be one of: xgboost, mlp, lstm, lightgbm")


def load_dataset(dataset_name: str, external_dir: Path) -> LoadedDataset:
    """Load the specified dataset using registry and pluggable loader.

    Args:
        dataset_name: Name of dataset in registry.
        external_dir: Path to data/external directory.

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
    return hooks.dataset_loader(config, external_dir)


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


__all__ = [
    "LoadedDataset",
    "build_optimization_config",
    "load_dataset",
    "optional_int",
    "parse_backend_name",
    "parse_dataset_name",
    "parse_device",
    "parse_feature_preset",
]
