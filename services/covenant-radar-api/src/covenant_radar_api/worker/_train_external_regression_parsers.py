"""Config parsers and ParseResult types for regression external training.

Parses raw JSON into strictly-typed per-backend regressor config TypedDicts.
Each backend has its own ParseResult type with a discriminated union
tag for type-safe narrowing.

Reuses config field parsers from _train_external_parsers.py since
hyperparameter configs (TrainConfig, LightGBMConfig) are backend-agnostic.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from covenant_ml.types import LightGBMConfig, TrainConfig
from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    load_json_str,
    require_str,
)

from covenant_radar_api.worker._optimize_regression_common import (
    parse_regression_dataset_name,
)
from covenant_radar_api.worker._train_external_parsers import (
    _parse_device,
    _parse_xgboost_config,
)
from covenant_radar_api.worker._train_external_parsers_tree import (
    _optional_float,
    _parse_lightgbm_config,
)

# =============================================================================
# ParseResult TypedDicts
# =============================================================================


class XGBoostRegParseResult(TypedDict, total=True):
    """Result of parsing XGBoost regression config.

    Args:
        backend: Discriminant tag for type narrowing.
        config: XGBoost hyperparameter config (same as classifier).
        dataset: Regression dataset name.
    """

    backend: Literal["xgboost_reg"]
    config: TrainConfig
    dataset: str


class LightGBMRegParseResult(TypedDict, total=True):
    """Result of parsing LightGBM regression config.

    Args:
        backend: Discriminant tag for type narrowing.
        config: LightGBM hyperparameter config (same as classifier).
        dataset: Regression dataset name.
    """

    backend: Literal["lightgbm_reg"]
    config: LightGBMConfig
    dataset: str


RegressionParseResult = XGBoostRegParseResult | LightGBMRegParseResult


# =============================================================================
# Backend name validation
# =============================================================================


def _parse_regression_train_backend(
    raw: JSONValue | None,
) -> Literal["xgboost_reg", "lightgbm_reg"]:
    """Parse regression train-external backend name.

    Only xgboost_reg and lightgbm_reg are supported for training.

    Args:
        raw: Raw JSON value for backend field.

    Returns:
        Validated regressor backend name.

    Raises:
        JSONTypeError: If value is not a string.
        ValueError: If value is not a supported training backend.
    """
    if raw is None:
        return "xgboost_reg"
    if not isinstance(raw, str):
        raise JSONTypeError("backend must be a string")
    if raw == "xgboost_reg":
        return "xgboost_reg"
    if raw == "lightgbm_reg":
        return "lightgbm_reg"
    raise ValueError("backend must be one of: xgboost_reg, lightgbm_reg")


# =============================================================================
# Top-level parse dispatch
# =============================================================================


def parse_external_regression_train_config(
    config_json: str,
) -> RegressionParseResult:
    """Parse regression training config JSON into a backend-specific ParseResult.

    Validates the dataset against the regression registry, split ratios,
    device setting, and backend-specific fields.

    Args:
        config_json: Raw JSON string with training config.

    Returns:
        Discriminated RegressionParseResult with backend tag, typed config,
        and dataset name.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
        ValueError: If dataset is unknown or ratios don't sum to 1.0.
    """
    raw = load_json_str(config_json)
    if not isinstance(raw, dict):
        raise JSONTypeError("config must be a JSON object")

    # Dataset selection - validate against regression registry
    dataset = require_str(raw, "dataset")
    dataset_name = parse_regression_dataset_name(dataset)

    # Common split defaults
    train_ratio = _optional_float(raw, "train_ratio", 0.7)
    val_ratio = _optional_float(raw, "val_ratio", 0.15)
    test_ratio = _optional_float(raw, "test_ratio", 0.15)

    # Validate ratios sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.3f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    device = _parse_device(raw.get("device"))

    # Backend selection
    backend_name = _parse_regression_train_backend(raw.get("backend"))

    if backend_name == "lightgbm_reg":
        lgbm_result: LightGBMRegParseResult = {
            "backend": "lightgbm_reg",
            "config": _parse_lightgbm_config(raw, device, train_ratio, val_ratio, test_ratio),
            "dataset": dataset_name,
        }
        return lgbm_result

    # xgboost_reg (default)
    xgb_result: XGBoostRegParseResult = {
        "backend": "xgboost_reg",
        "config": _parse_xgboost_config(raw, device, train_ratio, val_ratio, test_ratio),
        "dataset": dataset_name,
    }
    return xgb_result


__all__ = [
    "LightGBMRegParseResult",
    "RegressionParseResult",
    "XGBoostRegParseResult",
    "parse_external_regression_train_config",
]
