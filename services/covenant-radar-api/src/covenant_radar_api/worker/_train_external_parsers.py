"""Config parsers and ParseResult types for external training.

Parses raw JSON into strictly-typed per-backend config TypedDicts.
Each backend has its own ParseResult type with a discriminated union
tag for type-safe narrowing.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from covenant_ml.types import (
    ClearGBMConfig,
    LightGBMConfig,
    LogRegConfig,
    LSTMConfig,
    MLPConfig,
    RandomForestConfig,
    TrainConfig,
)
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    load_json_str,
    require_float,
    require_int,
    require_str,
)

from covenant_radar_api.worker._train_external_parsers_tree import (
    _optional_float,
    _optional_int,
    _parse_cleargbm_config,
    _parse_lightgbm_config,
    _parse_logreg_config,
    _parse_random_forest_config,
)

# =============================================================================
# Shared helpers
# =============================================================================


def _parse_device(raw: JSONValue | None) -> Literal["cpu", "cuda", "auto"]:
    """Parse device setting, defaulting to 'auto'.

    Args:
        raw: Raw JSON value for the device field.

    Returns:
        Device literal.

    Raises:
        JSONTypeError: If value is not a string.
        ValueError: If value is not cpu, cuda, or auto.
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


def _parse_xgboost_config(
    raw: JSONObject,
    device: Literal["cpu", "cuda", "auto"],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> TrainConfig:
    """Parse XGBoost backend config from JSON object.

    Args:
        raw: JSON object with XGBoost parameters.
        device: Compute device.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.

    Returns:
        TrainConfig TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    early_stopping_rounds = _optional_int(raw, "early_stopping_rounds", 10)
    reg_alpha = _optional_float(raw, "reg_alpha", 0.0)
    reg_lambda = _optional_float(raw, "reg_lambda", 1.0)
    xgb_cfg: TrainConfig = {
        "device": device,
        "learning_rate": require_float(raw, "learning_rate"),
        "max_depth": require_int(raw, "max_depth"),
        "n_estimators": require_int(raw, "n_estimators"),
        "subsample": require_float(raw, "subsample"),
        "colsample_bytree": require_float(raw, "colsample_bytree"),
        "random_state": require_int(raw, "random_state"),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "early_stopping_rounds": early_stopping_rounds,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
    }
    scale_pos_weight_raw = raw.get("scale_pos_weight")
    if isinstance(scale_pos_weight_raw, (int, float)):
        xgb_cfg["scale_pos_weight"] = float(scale_pos_weight_raw)
    elif scale_pos_weight_raw is not None:
        raise JSONTypeError("scale_pos_weight must be a number")
    return xgb_cfg


# =============================================================================
# MLP parser
# =============================================================================


def _parse_mlp_precision(
    raw: JSONObject,
) -> Literal["fp32", "fp16", "bf16", "auto"]:
    """Parse and validate MLP precision field.

    Args:
        raw: JSON object containing precision field.

    Returns:
        Precision literal.

    Raises:
        JSONTypeError: If value is not a valid precision.
    """
    precision_val = raw.get("precision")
    if precision_val == "fp32":
        return "fp32"
    if precision_val == "fp16":
        return "fp16"
    if precision_val == "bf16":
        return "bf16"
    if precision_val == "auto":
        return "auto"
    raise JSONTypeError("precision must be fp32, fp16, bf16, or auto")


def _parse_mlp_optimizer(
    raw: JSONObject,
) -> Literal["adamw", "adam", "sgd"]:
    """Parse and validate MLP optimizer field.

    Args:
        raw: JSON object containing optimizer field.

    Returns:
        Optimizer literal.

    Raises:
        JSONTypeError: If value is not a valid optimizer.
    """
    optimizer_val = raw.get("optimizer")
    if optimizer_val == "adamw":
        return "adamw"
    if optimizer_val == "adam":
        return "adam"
    if optimizer_val == "sgd":
        return "sgd"
    raise JSONTypeError("optimizer must be adamw, adam, or sgd")


def _parse_mlp_hidden_sizes(raw: JSONObject) -> tuple[int, ...]:
    """Parse and validate hidden_sizes as tuple of ints.

    Args:
        raw: JSON object containing hidden_sizes field.

    Returns:
        Tuple of hidden layer sizes.

    Raises:
        JSONTypeError: If value is not a list of ints.
    """
    hidden_sizes_val = raw.get("hidden_sizes")
    if not isinstance(hidden_sizes_val, list):
        raise JSONTypeError("hidden_sizes must be list of ints for mlp")
    result: list[int] = []
    for item in hidden_sizes_val:
        if not isinstance(item, int):
            raise JSONTypeError("hidden_sizes must be list of ints for mlp")
        result.append(item)
    return tuple(result)


def _parse_mlp_config(
    raw: JSONObject,
    device: Literal["cpu", "cuda", "auto"],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> MLPConfig:
    """Parse MLP backend config from JSON object.

    Args:
        raw: JSON object with MLP parameters.
        device: Compute device.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.

    Returns:
        MLPConfig TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    return {
        "device": device,
        "precision": _parse_mlp_precision(raw),
        "optimizer": _parse_mlp_optimizer(raw),
        "hidden_sizes": _parse_mlp_hidden_sizes(raw),
        "learning_rate": require_float(raw, "learning_rate"),
        "batch_size": require_int(raw, "batch_size"),
        "n_epochs": require_int(raw, "n_epochs"),
        "dropout": require_float(raw, "dropout"),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_state": require_int(raw, "random_state"),
        "early_stopping_patience": require_int(raw, "early_stopping_patience"),
    }


# =============================================================================
# LSTM parser
# =============================================================================


def _parse_lstm_precision(
    raw: JSONObject,
) -> Literal["fp32", "fp16", "bf16", "auto"]:
    """Parse and validate LSTM precision field.

    Args:
        raw: JSON object containing precision field.

    Returns:
        Precision literal.

    Raises:
        JSONTypeError: If value is not a valid precision.
    """
    precision_val = raw.get("precision")
    if precision_val == "fp32":
        return "fp32"
    if precision_val == "fp16":
        return "fp16"
    if precision_val == "bf16":
        return "bf16"
    if precision_val == "auto":
        return "auto"
    raise JSONTypeError("precision must be fp32, fp16, bf16, or auto")


def _parse_lstm_config(
    raw: JSONObject,
    device: Literal["cpu", "cuda", "auto"],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> LSTMConfig:
    """Parse LSTM backend config from JSON object.

    Args:
        raw: JSON object with LSTM parameters.
        device: Compute device.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.

    Returns:
        LSTMConfig TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    bidirectional_val = raw.get("bidirectional")
    if not isinstance(bidirectional_val, bool):
        raise JSONTypeError("bidirectional must be a boolean")
    return {
        "device": device,
        "precision": _parse_lstm_precision(raw),
        "hidden_size": require_int(raw, "hidden_size"),
        "num_layers": require_int(raw, "num_layers"),
        "dropout": require_float(raw, "dropout"),
        "bidirectional": bidirectional_val,
        "sequence_length": require_int(raw, "sequence_length"),
        "learning_rate": require_float(raw, "learning_rate"),
        "batch_size": require_int(raw, "batch_size"),
        "n_epochs": require_int(raw, "n_epochs"),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_state": require_int(raw, "random_state"),
        "early_stopping_patience": require_int(raw, "early_stopping_patience"),
    }


# =============================================================================
# LightGBM parser
# =============================================================================


class XGBoostParseResult(TypedDict, total=True):
    """Result of parsing XGBoost config."""

    backend: Literal["xgboost"]
    config: TrainConfig
    dataset: str


class MLPParseResult(TypedDict, total=True):
    """Result of parsing MLP config."""

    backend: Literal["mlp"]
    config: MLPConfig
    dataset: str


class LSTMParseResult(TypedDict, total=True):
    """Result of parsing LSTM config."""

    backend: Literal["lstm"]
    config: LSTMConfig
    dataset: str


class LightGBMParseResult(TypedDict, total=True):
    """Result of parsing LightGBM config."""

    backend: Literal["lightgbm"]
    config: LightGBMConfig
    dataset: str


class ClearGBMParseResult(TypedDict, total=True):
    """Result of parsing ClearGBM config."""

    backend: Literal["cleargbm"]
    config: ClearGBMConfig
    dataset: str


class LogRegParseResult(TypedDict, total=True):
    """Result of parsing LogReg config."""

    backend: Literal["logreg"]
    config: LogRegConfig
    dataset: str


class RandomForestParseResult(TypedDict, total=True):
    """Result of parsing RandomForest config."""

    backend: Literal["random_forest"]
    config: RandomForestConfig
    dataset: str


ParseResult = (
    XGBoostParseResult
    | MLPParseResult
    | LSTMParseResult
    | LightGBMParseResult
    | ClearGBMParseResult
    | LogRegParseResult
    | RandomForestParseResult
)


# =============================================================================
# Top-level parse dispatch
# =============================================================================


def parse_external_train_config(config_json: str) -> ParseResult:
    """Parse training config JSON into a backend-specific ParseResult.

    Validates the dataset against the dataset registry, split ratios,
    device setting, and backend-specific fields.

    Args:
        config_json: Raw JSON string with training config.

    Returns:
        Discriminated ParseResult with backend tag, typed config,
        and dataset name.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
        ValueError: If dataset is unknown or ratios don't sum to 1.0.
    """
    raw = load_json_str(config_json)
    if not isinstance(raw, dict):
        raise JSONTypeError("config must be a JSON object")

    # Dataset selection (required) - validate against registry
    from covenant_radar_api.worker import _test_hooks as hooks

    dataset = require_str(raw, "dataset")
    registry = hooks.dataset_registry_factory()
    if dataset not in registry:
        available = ", ".join(registry.list_names())
        raise ValueError(f"dataset must be one of: {available} (got {dataset})")
    dataset_name = dataset

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

    # Backend selection (optional; default xgboost)
    backend_val = raw.get("backend")
    if backend_val == "mlp":
        mlp_result: MLPParseResult = {
            "backend": "mlp",
            "config": _parse_mlp_config(raw, device, train_ratio, val_ratio, test_ratio),
            "dataset": dataset_name,
        }
        return mlp_result
    if backend_val == "lstm":
        lstm_result: LSTMParseResult = {
            "backend": "lstm",
            "config": _parse_lstm_config(raw, device, train_ratio, val_ratio, test_ratio),
            "dataset": dataset_name,
        }
        return lstm_result
    if backend_val == "lightgbm":
        lgbm_result: LightGBMParseResult = {
            "backend": "lightgbm",
            "config": _parse_lightgbm_config(raw, device, train_ratio, val_ratio, test_ratio),
            "dataset": dataset_name,
        }
        return lgbm_result
    if backend_val == "cleargbm":
        cgbm_result: ClearGBMParseResult = {
            "backend": "cleargbm",
            "config": _parse_cleargbm_config(raw, train_ratio, val_ratio, test_ratio),
            "dataset": dataset_name,
        }
        return cgbm_result
    if backend_val == "logreg":
        lr_result: LogRegParseResult = {
            "backend": "logreg",
            "config": _parse_logreg_config(raw, train_ratio, val_ratio, test_ratio),
            "dataset": dataset_name,
        }
        return lr_result
    if backend_val == "random_forest":
        rf_result: RandomForestParseResult = {
            "backend": "random_forest",
            "config": _parse_random_forest_config(raw, train_ratio, val_ratio, test_ratio),
            "dataset": dataset_name,
        }
        return rf_result
    xgb_result: XGBoostParseResult = {
        "backend": "xgboost",
        "config": _parse_xgboost_config(raw, device, train_ratio, val_ratio, test_ratio),
        "dataset": dataset_name,
    }
    return xgb_result
