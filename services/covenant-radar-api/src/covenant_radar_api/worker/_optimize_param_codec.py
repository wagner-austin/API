"""Sampled-parameter encoding for optimization results."""

from __future__ import annotations

from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from platform_core.json_utils import JSONObject


def _encode_int_tree_params(params: SampledIntParams, result: JSONObject) -> None:
    """Encode tree/ensemble integer params into result dict.

    Args:
        params: Sampled integer parameters.
        result: Mutable output dict to populate.
    """
    if "max_depth" in params:
        result["max_depth"] = params["max_depth"]
    if "n_estimators" in params:
        result["n_estimators"] = params["n_estimators"]
    if "num_leaves" in params:
        result["num_leaves"] = params["num_leaves"]
    if "min_child_samples" in params:
        result["min_child_samples"] = params["min_child_samples"]
    if "min_samples_split" in params:
        result["min_samples_split"] = params["min_samples_split"]
    if "min_samples_leaf" in params:
        result["min_samples_leaf"] = params["min_samples_leaf"]


def _encode_int_nn_params(params: SampledIntParams, result: JSONObject) -> None:
    """Encode neural-net and other integer params into result dict.

    Args:
        params: Sampled integer parameters.
        result: Mutable output dict to populate.
    """
    if "n_layers" in params:
        result["n_layers"] = params["n_layers"]
    if "hidden_size" in params:
        result["hidden_size"] = params["hidden_size"]
    if "num_layers" in params:
        result["num_layers"] = params["num_layers"]
    if "batch_size" in params:
        result["batch_size"] = params["batch_size"]
    if "max_bins" in params:
        result["max_bins"] = params["max_bins"]
    if "max_iter" in params:
        result["max_iter"] = params["max_iter"]


def encode_sampled_int_params(params: SampledIntParams) -> JSONObject:
    """Encode SampledIntParams to a JSON-serializable dict.

    Shared by the classifier and regressor optimize paths. Both consume the
    same SampledIntParams from covenant_ml, so a single implementation keeps
    neural-net keys (hidden_size, num_layers, batch_size, ...) from being
    dropped on one path only.

    Args:
        params: Sampled integer parameters.

    Returns:
        JSON-serializable dict with only present keys.
    """
    result: JSONObject = {}
    _encode_int_tree_params(params, result)
    _encode_int_nn_params(params, result)
    return result


def _encode_float_core_params(params: SampledFloatParams, result: JSONObject) -> None:
    """Encode core float params into result dict.

    Args:
        params: Sampled float parameters.
        result: Mutable output dict to populate.
    """
    if "learning_rate" in params:
        result["learning_rate"] = params["learning_rate"]
    if "reg_alpha" in params:
        result["reg_alpha"] = params["reg_alpha"]
    if "reg_lambda" in params:
        result["reg_lambda"] = params["reg_lambda"]
    if "subsample" in params:
        result["subsample"] = params["subsample"]
    if "colsample_bytree" in params:
        result["colsample_bytree"] = params["colsample_bytree"]
    if "dropout" in params:
        result["dropout"] = params["dropout"]
    if "drop_rate" in params:
        result["drop_rate"] = params["drop_rate"]


def _encode_float_extra_params(params: SampledFloatParams, result: JSONObject) -> None:
    """Encode extra float params into result dict.

    Args:
        params: Sampled float parameters.
        result: Mutable output dict to populate.
    """
    if "skip_drop" in params:
        result["skip_drop"] = params["skip_drop"]
    if "rate_drop" in params:
        result["rate_drop"] = params["rate_drop"]
    if "feature_fraction" in params:
        result["feature_fraction"] = params["feature_fraction"]
    if "C" in params:
        result["C"] = params["C"]
    if "tol" in params:
        result["tol"] = params["tol"]
    if "l1_ratio" in params:
        result["l1_ratio"] = params["l1_ratio"]
    if "max_features_float" in params:
        result["max_features_float"] = params["max_features_float"]


def encode_sampled_float_params(params: SampledFloatParams) -> JSONObject:
    """Encode SampledFloatParams to a JSON-serializable dict.

    Args:
        params: Sampled float parameters.

    Returns:
        JSON-serializable dict with only present keys.
    """
    result: JSONObject = {}
    _encode_float_core_params(params, result)
    _encode_float_extra_params(params, result)
    return result


def encode_sampled_string_params(params: SampledStringParams) -> JSONObject:
    """Encode SampledStringParams to a JSON-serializable dict.

    Args:
        params: Sampled string parameters.

    Returns:
        JSON-serializable dict with only present keys.
    """
    result: JSONObject = {}
    if "boosting_type" in params:
        result["boosting_type"] = params["boosting_type"]
    if "booster" in params:
        result["booster"] = params["booster"]
    if "penalty" in params:
        result["penalty"] = params["penalty"]
    if "solver" in params:
        result["solver"] = params["solver"]
    if "max_features" in params:
        result["max_features"] = params["max_features"]
    return result


__all__ = [
    "encode_sampled_float_params",
    "encode_sampled_int_params",
    "encode_sampled_string_params",
]
