"""Search space narrowing utilities for fine-tuning.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Provides functions to narrow search spaces around best known parameters.
"""

from __future__ import annotations

from ..optimizer.type_guards import (
    is_lightgbm_search_space,
    is_lstm_search_space,
    is_mlp_search_space,
    is_xgboost_search_space,
)
from ..optimizer.types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    CategoricalStringSpec,
    FloatRangeSpec,
    IntRangeSpec,
    LightGBMSearchSpace,
    LSTMSearchSpace,
    MLPSearchSpace,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    SearchSpace,
    XGBoostSearchSpace,
)


def _narrow_int_range(
    spec: IntRangeSpec | CategoricalIntSpec,
    best_value: int,
    radius: float,
) -> IntRangeSpec | CategoricalIntSpec:
    """Narrow an integer range around the best value.

    Args:
        spec: Original parameter specification.
        best_value: Best value found in previous optimization.
        radius: Fraction of original range to keep (0.5 = half width).

    Returns:
        Narrowed parameter specification.
    """
    if spec["param_type"] == "categorical_int":
        # Categorical specs stay the same
        return spec

    int_spec: IntRangeSpec = spec
    original_range = int_spec["high"] - int_spec["low"]
    new_half_range = max(1, int(original_range * radius / 2))

    new_low = max(int_spec["low"], best_value - new_half_range)
    new_high = min(int_spec["high"], best_value + new_half_range)

    # Ensure at least a range of 1
    if new_low >= new_high:
        new_low = max(int_spec["low"], best_value - 1)
        new_high = min(int_spec["high"], best_value + 1)

    return IntRangeSpec(
        param_type="int",
        low=new_low,
        high=new_high,
        log_scale=int_spec["log_scale"],
    )


def _narrow_float_range(
    spec: FloatRangeSpec | CategoricalFloatSpec,
    best_value: float,
    radius: float,
) -> FloatRangeSpec | CategoricalFloatSpec:
    """Narrow a float range around the best value.

    Args:
        spec: Original parameter specification.
        best_value: Best value found in previous optimization.
        radius: Fraction of original range to keep (0.5 = half width).

    Returns:
        Narrowed parameter specification.
    """
    if spec["param_type"] == "categorical_float":
        # Categorical specs stay the same
        return spec

    float_spec: FloatRangeSpec = spec
    original_range = float_spec["high"] - float_spec["low"]
    new_half_range = original_range * radius / 2

    new_low = max(float_spec["low"], best_value - new_half_range)
    new_high = min(float_spec["high"], best_value + new_half_range)

    # Ensure some range remains
    if new_low >= new_high:
        epsilon = original_range * 0.01
        new_low = max(float_spec["low"], best_value - epsilon)
        new_high = min(float_spec["high"], best_value + epsilon)

    return FloatRangeSpec(
        param_type="float",
        low=new_low,
        high=new_high,
        log_scale=float_spec["log_scale"],
    )


def narrow_xgboost_space(
    space: XGBoostSearchSpace,
    best_int: SampledIntParams,
    best_float: SampledFloatParams,
    best_string: SampledStringParams,
    radius: float,
) -> XGBoostSearchSpace:
    """Narrow XGBoost search space around best parameters.

    Args:
        space: Original search space.
        best_int: Best integer parameters.
        best_float: Best float parameters.
        best_string: Best string parameters.
        radius: Fraction of original range to keep.

    Returns:
        Narrowed XGBoostSearchSpace.
    """
    result: XGBoostSearchSpace = {
        "max_depth": _narrow_int_range(
            space["max_depth"],
            best_int.get("max_depth", 6),
            radius,
        ),
        "n_estimators": _narrow_int_range(
            space["n_estimators"],
            best_int.get("n_estimators", 100),
            radius,
        ),
        "learning_rate": _narrow_float_range(
            space["learning_rate"],
            best_float.get("learning_rate", 0.1),
            radius,
        ),
        "reg_alpha": _narrow_float_range(
            space["reg_alpha"],
            best_float.get("reg_alpha", 0.0),
            radius,
        ),
        "reg_lambda": _narrow_float_range(
            space["reg_lambda"],
            best_float.get("reg_lambda", 1.0),
            radius,
        ),
        "subsample": _narrow_float_range(
            space["subsample"],
            best_float.get("subsample", 0.8),
            radius,
        ),
        "colsample_bytree": _narrow_float_range(
            space["colsample_bytree"],
            best_float.get("colsample_bytree", 0.8),
            radius,
        ),
    }

    # Keep booster fixed if string params specify it
    if "booster" in space and "booster" in best_string:
        result["booster"] = CategoricalStringSpec(
            param_type="categorical_str",
            choices=(best_string["booster"],),
        )
        # Keep DART params if present
        if best_string["booster"] == "dart":
            if "rate_drop" in space and "rate_drop" in best_float:
                result["rate_drop"] = _narrow_float_range(
                    space["rate_drop"],
                    best_float["rate_drop"],
                    radius,
                )
            if "skip_drop" in space and "skip_drop" in best_float:
                result["skip_drop"] = _narrow_float_range(
                    space["skip_drop"],
                    best_float["skip_drop"],
                    radius,
                )

    return result


def narrow_mlp_space(
    space: MLPSearchSpace,
    best_int: SampledIntParams,
    best_float: SampledFloatParams,
    radius: float,
) -> MLPSearchSpace:
    """Narrow MLP search space around best parameters.

    Args:
        space: Original search space.
        best_int: Best integer parameters.
        best_float: Best float parameters.
        radius: Fraction of original range to keep.

    Returns:
        Narrowed MLPSearchSpace.
    """
    return MLPSearchSpace(
        n_layers=_narrow_int_range(
            space["n_layers"],
            best_int.get("n_layers", 2),
            radius,
        ),
        hidden_size=_narrow_int_range(
            space["hidden_size"],
            best_int.get("hidden_size", 64),
            radius,
        ),
        batch_size=_narrow_int_range(
            space["batch_size"],
            best_int.get("batch_size", 32),
            radius,
        ),
        learning_rate=_narrow_float_range(
            space["learning_rate"],
            best_float.get("learning_rate", 0.001),
            radius,
        ),
        dropout=_narrow_float_range(
            space["dropout"],
            best_float.get("dropout", 0.1),
            radius,
        ),
    )


def narrow_lstm_space(
    space: LSTMSearchSpace,
    best_int: SampledIntParams,
    best_float: SampledFloatParams,
    radius: float,
) -> LSTMSearchSpace:
    """Narrow LSTM search space around best parameters.

    Args:
        space: Original search space.
        best_int: Best integer parameters.
        best_float: Best float parameters.
        radius: Fraction of original range to keep.

    Returns:
        Narrowed LSTMSearchSpace.
    """
    return LSTMSearchSpace(
        hidden_size=_narrow_int_range(
            space["hidden_size"],
            best_int.get("hidden_size", 64),
            radius,
        ),
        num_layers=_narrow_int_range(
            space["num_layers"],
            best_int.get("num_layers", 2),
            radius,
        ),
        batch_size=_narrow_int_range(
            space["batch_size"],
            best_int.get("batch_size", 32),
            radius,
        ),
        learning_rate=_narrow_float_range(
            space["learning_rate"],
            best_float.get("learning_rate", 0.001),
            radius,
        ),
        dropout=_narrow_float_range(
            space["dropout"],
            best_float.get("dropout", 0.1),
            radius,
        ),
    )


def narrow_lightgbm_space(
    space: LightGBMSearchSpace,
    best_int: SampledIntParams,
    best_float: SampledFloatParams,
    best_string: SampledStringParams,
    radius: float,
) -> LightGBMSearchSpace:
    """Narrow LightGBM search space around best parameters.

    Args:
        space: Original search space.
        best_int: Best integer parameters.
        best_float: Best float parameters.
        best_string: Best string parameters.
        radius: Fraction of original range to keep.

    Returns:
        Narrowed LightGBMSearchSpace.
    """
    result: LightGBMSearchSpace = {
        "n_estimators": _narrow_int_range(
            space["n_estimators"],
            best_int.get("n_estimators", 100),
            radius,
        ),
        "num_leaves": _narrow_int_range(
            space["num_leaves"],
            best_int.get("num_leaves", 31),
            radius,
        ),
        "learning_rate": _narrow_float_range(
            space["learning_rate"],
            best_float.get("learning_rate", 0.1),
            radius,
        ),
        "subsample": _narrow_float_range(
            space["subsample"],
            best_float.get("subsample", 0.8),
            radius,
        ),
        "colsample_bytree": _narrow_float_range(
            space["colsample_bytree"],
            best_float.get("colsample_bytree", 0.8),
            radius,
        ),
        "reg_alpha": _narrow_float_range(
            space["reg_alpha"],
            best_float.get("reg_alpha", 0.0),
            radius,
        ),
        "reg_lambda": _narrow_float_range(
            space["reg_lambda"],
            best_float.get("reg_lambda", 1.0),
            radius,
        ),
    }

    # Keep boosting_type fixed if string params specify it
    if "boosting_type" in space and "boosting_type" in best_string:
        result["boosting_type"] = CategoricalStringSpec(
            param_type="categorical_str",
            choices=(best_string["boosting_type"],),
        )
        # Keep DART params if present
        if best_string["boosting_type"] == "dart":
            if "drop_rate" in space and "drop_rate" in best_float:
                result["drop_rate"] = _narrow_float_range(
                    space["drop_rate"],
                    best_float["drop_rate"],
                    radius,
                )
            if "skip_drop" in space and "skip_drop" in best_float:
                result["skip_drop"] = _narrow_float_range(
                    space["skip_drop"],
                    best_float["skip_drop"],
                    radius,
                )
            if "feature_fraction" in space and "feature_fraction" in best_float:
                result["feature_fraction"] = _narrow_float_range(
                    space["feature_fraction"],
                    best_float["feature_fraction"],
                    radius,
                )

    return result


def narrow_search_space(
    space: SearchSpace,
    best_int: SampledIntParams,
    best_float: SampledFloatParams,
    best_string: SampledStringParams,
    radius: float,
) -> SearchSpace:
    """Narrow any search space around best parameters.

    Args:
        space: Original search space (any backend type).
        best_int: Best integer parameters.
        best_float: Best float parameters.
        best_string: Best string parameters.
        radius: Fraction of original range to keep (0.5 = half width).

    Returns:
        Narrowed search space of the same type.

    Raises:
        ValueError: If search space type cannot be determined.
    """
    if is_xgboost_search_space(space):
        return narrow_xgboost_space(
            space,
            best_int,
            best_float,
            best_string,
            radius,
        )
    if is_mlp_search_space(space):
        return narrow_mlp_space(
            space,
            best_int,
            best_float,
            radius,
        )
    if is_lstm_search_space(space):
        return narrow_lstm_space(
            space,
            best_int,
            best_float,
            radius,
        )
    # LightGBM is the remaining type after other guards
    # Type assertion to satisfy mypy
    assert is_lightgbm_search_space(space)
    return narrow_lightgbm_space(
        space,
        best_int,
        best_float,
        best_string,
        radius,
    )


__all__ = [
    "narrow_lightgbm_space",
    "narrow_lstm_space",
    "narrow_mlp_space",
    "narrow_search_space",
    "narrow_xgboost_space",
]
