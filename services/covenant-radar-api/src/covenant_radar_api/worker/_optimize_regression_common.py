"""Shared utilities for regression hyperparameter optimization jobs.

Contains regression-specific dataset loading and backend parsing.
Reuses shared parsers from _optimize_common for device, feature_preset, etc.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

from covenant_ml.datasets import RegressionLoadedDataset
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.types import RegressorBackendName
from platform_core.json_utils import JSONTypeError, JSONValue


def parse_regressor_backend_name(raw: JSONValue | None) -> RegressorBackendName:
    """Parse regressor backend name, defaulting to 'xgboost_reg'.

    Args:
        raw: Raw JSON value.

    Returns:
        RegressorBackendName literal.

    Raises:
        JSONTypeError: If value is not a string.
        ValueError: If value is not a valid regressor backend.
    """
    if raw is None:
        return "xgboost_reg"
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
    raise ValueError("backend must be one of: xgboost_reg, lightgbm_reg, mlp_reg, lstm_reg")


def parse_regression_dataset_name(dataset: str) -> str:
    """Parse and validate regression dataset name against the regression registry.

    Args:
        dataset: Dataset name string.

    Returns:
        Validated dataset name from regression registry.

    Raises:
        ValueError: If dataset name is not in regression registry.
    """
    from covenant_radar_api.worker import _regression_hooks as hooks

    registry = hooks.regression_registry_factory()
    if dataset in registry:
        return dataset

    all_names = sorted(registry.list_names())
    raise ValueError(f"dataset must be one of: {', '.join(all_names)} (got {dataset})")


def load_regression_dataset(
    dataset_name: str,
    external_dir: Path,
    progress_callback: ProgressCallbackProtocol | None = None,
) -> RegressionLoadedDataset:
    """Load the specified regression dataset using registry and pluggable loader.

    Args:
        dataset_name: Name of dataset in regression registry.
        external_dir: Path to data/external directory.
        progress_callback: Optional callback for loading progress updates.

    Returns:
        RegressionLoadedDataset with features, continuous targets, and metadata.

    Raises:
        KeyError: If dataset not in registry.
        FileNotFoundError: If dataset file doesn't exist.
        ValueError: If data doesn't match expected format.
    """
    from covenant_radar_api.worker import _regression_hooks as hooks

    registry = hooks.regression_registry_factory()
    config = registry.get(dataset_name)
    return hooks.regression_dataset_loader(config, external_dir, progress_callback)


__all__ = [
    "load_regression_dataset",
    "parse_regression_dataset_name",
    "parse_regressor_backend_name",
]
