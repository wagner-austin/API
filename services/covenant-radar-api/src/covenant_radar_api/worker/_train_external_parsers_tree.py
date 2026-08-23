"""Per-backend config parsers: LightGBM, ClearGBM, LogReg, RandomForest."""

from __future__ import annotations

from typing import Literal

from covenant_ml.types import (
    ClearGBMConfig,
    ClearGBMGrowthStrategy,
    LightGBMConfig,
    LogRegConfig,
    LogRegPenalty,
    LogRegSolver,
    RandomForestConfig,
)
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_float,
    require_int,
)


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


def _parse_colsample_bytree_nullable(
    raw: JSONObject,
) -> float | None:
    """Parse colsample_bytree field accepting float or null.

    Args:
        raw: JSON object containing the field.

    Returns:
        The per-tree feature fraction, or None for all features. Range
        validation ((0, 1) exclusive) is owned by the cleargbm boundary.

    Raises:
        JSONTypeError: If value is not a float or null.
    """
    val = raw.get("colsample_bytree")
    if val is None:
        return None
    if isinstance(val, float):
        return val
    raise JSONTypeError("colsample_bytree must be a float or null")


def _parse_categorical_features_nullable(
    raw: JSONObject,
) -> list[str] | None:
    """Parse categorical_features field accepting a list of names or null.

    Args:
        raw: JSON object containing the field.

    Returns:
        The categorical column names, or None for all-numeric. Name
        resolution against the dataset's features is owned by the
        covenant_ml backend.

    Raises:
        JSONTypeError: If value is not a list of strings or null.
    """
    val = raw.get("categorical_features")
    if val is None:
        return None
    if not isinstance(val, list):
        raise JSONTypeError("categorical_features must be a list of strings or null")
    names: list[str] = []
    for item in val:
        if not isinstance(item, str):
            raise JSONTypeError("categorical_features entries must be strings")
        names.append(item)
    return names


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
    growth_strategy = _parse_cleargbm_growth_strategy(raw)
    num_leaves = _parse_cleargbm_num_leaves(raw)
    if growth_strategy == "leaf_wise" and num_leaves is None:
        raise JSONTypeError("leaf_wise growth requires num_leaves")
    if growth_strategy == "depth_wise" and num_leaves is not None:
        raise JSONTypeError("depth_wise growth takes no num_leaves budget")
    return {
        "n_estimators": require_int(raw, "n_estimators"),
        "max_depth": require_int(raw, "max_depth"),
        "learning_rate": require_float(raw, "learning_rate"),
        "min_samples_split": require_int(raw, "min_samples_split"),
        "min_samples_leaf": require_int(raw, "min_samples_leaf"),
        "max_features": _parse_max_features_nullable(raw),
        "colsample_bytree": _parse_colsample_bytree_nullable(raw),
        "categorical_features": _parse_categorical_features_nullable(raw),
        "max_bins": _optional_int(raw, "max_bins", 64),
        "subsample": require_float(raw, "subsample"),
        "random_state": require_int(raw, "random_state"),
        "monotonic_constraints": _parse_monotonic_constraints(raw),
        "reg_alpha": _optional_float(raw, "reg_alpha", 0.0),
        "reg_lambda": _optional_float(raw, "reg_lambda", 1.0),
        "n_jobs": _optional_int(raw, "n_jobs", -1),
        "growth_strategy": growth_strategy,
        "num_leaves": num_leaves,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "early_stopping_rounds": _optional_int(raw, "early_stopping_rounds", 10),
    }


def _parse_cleargbm_growth_strategy(raw: JSONObject) -> ClearGBMGrowthStrategy:
    """Parse and validate the ClearGBM growth_strategy field.

    Args:
        raw: JSON object that may contain a growth_strategy field.

    Returns:
        Validated growth strategy; absent defaults to "depth_wise", the
        historical behavior.

    Raises:
        JSONTypeError: If a present value is not a valid strategy.
    """
    val = raw.get("growth_strategy")
    if val is None:
        return "depth_wise"
    if val == "depth_wise":
        return "depth_wise"
    if val == "leaf_wise":
        return "leaf_wise"
    raise JSONTypeError("growth_strategy must be one of: depth_wise, leaf_wise")


def _parse_cleargbm_num_leaves(raw: JSONObject) -> int | None:
    """Parse the ClearGBM num_leaves field.

    Args:
        raw: JSON object that may contain a num_leaves field.

    Returns:
        The leaf budget, or None when absent or explicitly null.

    Raises:
        JSONTypeError: If a present value is not an integer.
    """
    val = raw.get("num_leaves")
    if val is None:
        return None
    if isinstance(val, bool) or not isinstance(val, int):
        raise JSONTypeError("num_leaves must be an integer or null")
    return val


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
