"""Shared utilities for regression hyperparameter optimization jobs.

Contains regression-specific dataset loading and backend parsing.
Reuses shared parsers from _optimize_common for device, feature_preset, etc.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

from covenant_ml.datasets import RegressionLoadedDataset
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.types_regression import RegressorBackendName
from platform_core.json_utils import JSONTypeError, JSONValue
from platform_core.members import find_member

#: The regressor backends this service accepts, in the order its refusals name
#: them. cleargbm_reg is registered in covenant_ml but is not served here.
SERVED_REGRESSOR_BACKENDS: tuple[RegressorBackendName, ...] = (
    RegressorBackendName.XGBOOST_REG,
    RegressorBackendName.LIGHTGBM_REG,
    RegressorBackendName.MLP_REG,
    RegressorBackendName.LSTM_REG,
)


def find_served_regressor_backend(raw: str) -> RegressorBackendName | None:
    """Return the served regressor backend named by ``raw``, if any.

    Args:
        raw: The candidate backend word.

    Returns:
        The member when it names a served backend, else None.
    """
    backend = find_member(raw, RegressorBackendName)
    return backend if backend in SERVED_REGRESSOR_BACKENDS else None


def parse_regressor_backend_name(raw: JSONValue | None) -> RegressorBackendName:
    """Parse regressor backend name, defaulting to 'xgboost_reg'.

    Args:
        raw: Raw JSON value.

    Returns:
        The RegressorBackendName member.

    Raises:
        JSONTypeError: If value is not a string.
        ValueError: If value is not a served regressor backend.
    """
    if raw is None:
        return RegressorBackendName.XGBOOST_REG
    if not isinstance(raw, str):
        raise JSONTypeError("backend must be a string")
    backend = find_served_regressor_backend(raw)
    if backend is None:
        raise ValueError(f"backend must be one of: {', '.join(SERVED_REGRESSOR_BACKENDS)}")
    return backend


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
    "SERVED_REGRESSOR_BACKENDS",
    "find_served_regressor_backend",
    "load_regression_dataset",
    "parse_regression_dataset_name",
    "parse_regressor_backend_name",
]
