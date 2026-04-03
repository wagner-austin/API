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
    LogRegPenalty,
    LogRegSolver,
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


def _optional_float(data: JSONObject, key: str, default: float) -> float:
    """Extract optional float from dict.

    Args:
        data: JSON object.
        key: Field name.
        default: Default value if absent.

    Returns:
        Float value.

    Raises:
        JSONTypeError: If value is present but not a number.
    """
    raw = data.get(key)
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    raise JSONTypeError(f"Field '{key}' must be a number")


def _optional_int(data: JSONObject, key: str, default: int) -> int:
    """Extract optional int from dict.

    Args:
        data: JSON object.
        key: Field name.
        default: Default value if absent.

    Returns:
        Int value.

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


def _parse_optional_bool(
    raw: JSONObject,
    key: str,
    default: bool,
) -> bool:
    """Parse optional boolean field with a default.

    Args:
        raw: JSON object containing the field.
        key: Field name.
        default: Default value if field is absent.

    Returns:
        Boolean value.

    Raises:
        JSONTypeError: If value is present but not a boolean.
    """
    val = raw.get(key)
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    raise JSONTypeError(f"Field '{key}' must be a boolean")


# =============================================================================
# XGBoost parser
# =============================================================================


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


def _parse_lightgbm_config(
    raw: JSONObject,
    device: Literal["cpu", "cuda", "auto"],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> LightGBMConfig:
    """Parse LightGBM backend config from JSON object.

    Args:
        raw: JSON object with LightGBM parameters.
        device: Compute device.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.

    Returns:
        LightGBMConfig TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    early_stopping_rounds = _optional_int(raw, "early_stopping_rounds", 10)
    reg_alpha = _optional_float(raw, "reg_alpha", 0.0)
    reg_lambda = _optional_float(raw, "reg_lambda", 1.0)
    return {
        "device": device,
        "learning_rate": require_float(raw, "learning_rate"),
        "max_depth": require_int(raw, "max_depth"),
        "n_estimators": require_int(raw, "n_estimators"),
        "num_leaves": require_int(raw, "num_leaves"),
        "min_child_samples": require_int(raw, "min_child_samples"),
        "subsample": require_float(raw, "subsample"),
        "colsample_bytree": require_float(raw, "colsample_bytree"),
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_state": require_int(raw, "random_state"),
        "early_stopping_rounds": early_stopping_rounds,
    }


# =============================================================================
# ClearGBM parser
# =============================================================================


def _parse_max_features_nullable(
    raw: JSONObject,
) -> int | float | None:
    """Parse max_features field accepting int, float, or null.

    Args:
        raw: JSON object containing the field.

    Returns:
        int, float, or None for max_features.

    Raises:
        JSONTypeError: If value is not int, float, or null.
    """
    val = raw.get("max_features")
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return val
    raise JSONTypeError("max_features must be an int, float, or null")


def _parse_monotonic_constraints(
    raw: JSONObject,
) -> dict[str, int] | None:
    """Parse monotonic_constraints field accepting dict[str, int] or null.

    Args:
        raw: JSON object containing the field.

    Returns:
        Dict mapping feature names to +1/-1 constraints, or None.

    Raises:
        JSONTypeError: If value is not a dict of str->int or null.
    """
    val = raw.get("monotonic_constraints")
    if val is None:
        return None
    if not isinstance(val, dict):
        raise JSONTypeError("monotonic_constraints must be a dict or null")
    result: dict[str, int] = {}
    for k, v in val.items():
        if not isinstance(v, int):
            raise JSONTypeError("monotonic_constraints values must be ints")
        result[k] = v
    return result


def _parse_cleargbm_config(
    raw: JSONObject,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> ClearGBMConfig:
    """Parse ClearGBM backend config from JSON object.

    Args:
        raw: JSON object with ClearGBM parameters.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.

    Returns:
        ClearGBMConfig TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    track_val = raw.get("track_contributions")
    if not isinstance(track_val, bool):
        raise JSONTypeError("track_contributions must be a boolean")
    return {
        "n_estimators": require_int(raw, "n_estimators"),
        "max_depth": require_int(raw, "max_depth"),
        "learning_rate": require_float(raw, "learning_rate"),
        "min_samples_split": require_int(raw, "min_samples_split"),
        "min_samples_leaf": require_int(raw, "min_samples_leaf"),
        "max_features": _parse_max_features_nullable(raw),
        "max_bins": _optional_int(raw, "max_bins", 64),
        "subsample": require_float(raw, "subsample"),
        "random_state": require_int(raw, "random_state"),
        "track_contributions": track_val,
        "monotonic_constraints": _parse_monotonic_constraints(raw),
        "reg_alpha": _optional_float(raw, "reg_alpha", 0.0),
        "reg_lambda": _optional_float(raw, "reg_lambda", 1.0),
        "n_jobs": _optional_int(raw, "n_jobs", -1),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "early_stopping_rounds": _optional_int(raw, "early_stopping_rounds", 10),
    }


# =============================================================================
# LogReg parser
# =============================================================================


def _parse_logreg_solver(raw: JSONObject) -> LogRegSolver:
    """Parse and validate LogReg solver field.

    Args:
        raw: JSON object containing the solver field.

    Returns:
        Validated LogRegSolver literal.

    Raises:
        JSONTypeError: If value is not a valid solver.
    """
    val = raw.get("solver")
    if val == "lbfgs":
        return "lbfgs"
    if val == "liblinear":
        return "liblinear"
    if val == "newton-cg":
        return "newton-cg"
    if val == "newton-cholesky":
        return "newton-cholesky"
    if val == "sag":
        return "sag"
    if val == "saga":
        return "saga"
    raise JSONTypeError(
        "solver must be one of: lbfgs, liblinear, newton-cg, newton-cholesky, sag, saga"
    )


def _parse_logreg_penalty(raw: JSONObject) -> LogRegPenalty:
    """Parse and validate LogReg penalty field.

    Args:
        raw: JSON object containing the penalty field.

    Returns:
        Validated LogRegPenalty literal.

    Raises:
        JSONTypeError: If value is not a valid penalty.
    """
    val = raw.get("penalty")
    if val == "l1":
        return "l1"
    if val == "l2":
        return "l2"
    if val == "elasticnet":
        return "elasticnet"
    if val == "none":
        return "none"
    raise JSONTypeError("penalty must be one of: l1, l2, elasticnet, none")


def _parse_logreg_config(
    raw: JSONObject,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> LogRegConfig:
    """Parse LogReg backend config from JSON object.

    Args:
        raw: JSON object with LogReg parameters.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.

    Returns:
        LogRegConfig TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    class_weight_val = raw.get("class_weight_balanced")
    if not isinstance(class_weight_val, bool):
        raise JSONTypeError("class_weight_balanced must be a boolean")
    return {
        "solver": _parse_logreg_solver(raw),
        "penalty": _parse_logreg_penalty(raw),
        "C": require_float(raw, "C"),
        "max_iter": require_int(raw, "max_iter"),
        "tol": require_float(raw, "tol"),
        "class_weight_balanced": class_weight_val,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_state": require_int(raw, "random_state"),
        "l1_ratio": _optional_float(raw, "l1_ratio", 0.0),
    }


# =============================================================================
# Random Forest parser
# =============================================================================


def _parse_max_depth_nullable(raw: JSONObject) -> int | None:
    """Parse max_depth field accepting int or null.

    Args:
        raw: JSON object containing the max_depth field.

    Returns:
        int or None.

    Raises:
        JSONTypeError: If value is not int or null.
    """
    val = raw.get("max_depth")
    if val is None:
        return None
    if isinstance(val, int):
        return val
    raise JSONTypeError("max_depth must be an int or null")


def _parse_rf_max_features(
    raw: JSONObject,
) -> Literal["sqrt", "log2"] | float | int | None:
    """Parse Random Forest max_features field.

    Accepts "sqrt", "log2", float, int, or null.

    Args:
        raw: JSON object containing the max_features field.

    Returns:
        Validated max_features value.

    Raises:
        JSONTypeError: If value is not a valid max_features option.
    """
    val = raw.get("max_features")
    if val is None:
        return None
    if val == "sqrt":
        return "sqrt"
    if val == "log2":
        return "log2"
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return val
    raise JSONTypeError("max_features must be 'sqrt', 'log2', int, float, or null")


def _parse_random_forest_config(
    raw: JSONObject,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> RandomForestConfig:
    """Parse Random Forest backend config from JSON object.

    Args:
        raw: JSON object with Random Forest parameters.
        train_ratio: Training split ratio.
        val_ratio: Validation split ratio.
        test_ratio: Test split ratio.

    Returns:
        RandomForestConfig TypedDict.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    bootstrap_val = raw.get("bootstrap")
    if not isinstance(bootstrap_val, bool):
        raise JSONTypeError("bootstrap must be a boolean")
    class_weight_val = raw.get("class_weight_balanced")
    if not isinstance(class_weight_val, bool):
        raise JSONTypeError("class_weight_balanced must be a boolean")
    return {
        "n_estimators": require_int(raw, "n_estimators"),
        "max_depth": _parse_max_depth_nullable(raw),
        "min_samples_split": require_int(raw, "min_samples_split"),
        "min_samples_leaf": require_int(raw, "min_samples_leaf"),
        "max_features": _parse_rf_max_features(raw),
        "bootstrap": bootstrap_val,
        "class_weight_balanced": class_weight_val,
        "n_jobs": _optional_int(raw, "n_jobs", -1),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "random_state": require_int(raw, "random_state"),
        "oob_score": _parse_optional_bool(raw, "oob_score", False),
    }


# =============================================================================
# ParseResult TypedDicts and union
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
