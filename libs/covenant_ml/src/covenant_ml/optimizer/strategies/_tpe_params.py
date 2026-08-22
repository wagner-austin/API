"""Per-backend Optuna trial samplers and best-params extractors."""

from __future__ import annotations

from ..type_guards import (
    is_cleargbm_search_space,
    is_lightgbm_search_space,
    is_logreg_search_space,
    is_lstm_search_space,
    is_mlp_search_space,
    is_random_forest_search_space,
    is_xgboost_search_space,
)
from ..types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    CategoricalStringSpec,
    ClearGBMSearchSpace,
    FloatRangeSpec,
    IntRangeSpec,
    LightGBMSearchSpace,
    LogRegSearchSpace,
    LSTMSearchSpace,
    MLPSearchSpace,
    RandomForestSearchSpace,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    SearchSpace,
    XGBoostSearchSpace,
)
from ._hooks import (
    OptunaTrialProtocol,
)


def _sample_int(
    trial: OptunaTrialProtocol,
    name: str,
    spec: IntRangeSpec | CategoricalIntSpec,
) -> int:
    """Sample integer parameter from trial."""
    if spec["param_type"] == "int":
        int_spec: IntRangeSpec = spec
        return trial.suggest_int(
            name,
            int_spec["low"],
            int_spec["high"],
            log=int_spec["log_scale"],
        )
    cat_spec: CategoricalIntSpec = spec
    result = trial.suggest_categorical(name, cat_spec["choices"])
    return int(result)


def _sample_float(
    trial: OptunaTrialProtocol,
    name: str,
    spec: FloatRangeSpec | CategoricalFloatSpec,
) -> float:
    """Sample float parameter from trial."""
    if spec["param_type"] == "float":
        float_spec: FloatRangeSpec = spec
        return trial.suggest_float(
            name,
            float_spec["low"],
            float_spec["high"],
            log=float_spec["log_scale"],
        )
    cat_spec: CategoricalFloatSpec = spec
    result = trial.suggest_categorical(name, cat_spec["choices"])
    return float(result)


def _sample_string(
    trial: OptunaTrialProtocol,
    name: str,
    spec: CategoricalStringSpec,
) -> str:
    """Sample string parameter from trial."""
    result = trial.suggest_categorical(name, spec["choices"])
    return str(result)


def _sample_xgboost_params(
    trial: OptunaTrialProtocol,
    space: XGBoostSearchSpace,
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Sample XGBoost hyperparameters from search space."""
    int_params: SampledIntParams = {
        "max_depth": _sample_int(trial, "max_depth", space["max_depth"]),
        "n_estimators": _sample_int(trial, "n_estimators", space["n_estimators"]),
    }

    float_params: SampledFloatParams = {
        "learning_rate": _sample_float(trial, "learning_rate", space["learning_rate"]),
        "reg_alpha": _sample_float(trial, "reg_alpha", space["reg_alpha"]),
        "reg_lambda": _sample_float(trial, "reg_lambda", space["reg_lambda"]),
        "subsample": _sample_float(trial, "subsample", space["subsample"]),
        "colsample_bytree": _sample_float(trial, "colsample_bytree", space["colsample_bytree"]),
    }

    string_params: SampledStringParams = {}

    # Optional DART params
    if "booster" in space:
        booster = _sample_string(trial, "booster", space["booster"])
        string_params["booster"] = booster
        if booster == "dart":
            if "rate_drop" in space:
                float_params["rate_drop"] = _sample_float(trial, "rate_drop", space["rate_drop"])
            if "skip_drop" in space:
                float_params["skip_drop"] = _sample_float(trial, "skip_drop", space["skip_drop"])

    return int_params, float_params, string_params


def _sample_mlp_params(
    trial: OptunaTrialProtocol,
    space: MLPSearchSpace,
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Sample MLP hyperparameters from search space."""
    int_params: SampledIntParams = {
        "n_layers": _sample_int(trial, "n_layers", space["n_layers"]),
        "hidden_size": _sample_int(trial, "hidden_size", space["hidden_size"]),
        "batch_size": _sample_int(trial, "batch_size", space["batch_size"]),
    }

    float_params: SampledFloatParams = {
        "learning_rate": _sample_float(trial, "learning_rate", space["learning_rate"]),
        "dropout": _sample_float(trial, "dropout", space["dropout"]),
    }

    return int_params, float_params, {}


def _sample_lstm_params(
    trial: OptunaTrialProtocol,
    space: LSTMSearchSpace,
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Sample LSTM hyperparameters from search space."""
    int_params: SampledIntParams = {
        "hidden_size": _sample_int(trial, "hidden_size", space["hidden_size"]),
        "num_layers": _sample_int(trial, "num_layers", space["num_layers"]),
        "batch_size": _sample_int(trial, "batch_size", space["batch_size"]),
    }

    float_params: SampledFloatParams = {
        "learning_rate": _sample_float(trial, "learning_rate", space["learning_rate"]),
        "dropout": _sample_float(trial, "dropout", space["dropout"]),
    }

    return int_params, float_params, {}


def _sample_lightgbm_params(
    trial: OptunaTrialProtocol,
    space: LightGBMSearchSpace,
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Sample LightGBM hyperparameters from search space."""
    int_params: SampledIntParams = {
        "n_estimators": _sample_int(trial, "n_estimators", space["n_estimators"]),
        "num_leaves": _sample_int(trial, "num_leaves", space["num_leaves"]),
    }

    float_params: SampledFloatParams = {
        "learning_rate": _sample_float(trial, "learning_rate", space["learning_rate"]),
        "subsample": _sample_float(trial, "subsample", space["subsample"]),
        "colsample_bytree": _sample_float(trial, "colsample_bytree", space["colsample_bytree"]),
        "reg_alpha": _sample_float(trial, "reg_alpha", space["reg_alpha"]),
        "reg_lambda": _sample_float(trial, "reg_lambda", space["reg_lambda"]),
    }

    string_params: SampledStringParams = {}

    # Optional DART params
    if "boosting_type" in space:
        boosting_type = _sample_string(trial, "boosting_type", space["boosting_type"])
        string_params["boosting_type"] = boosting_type
        if boosting_type == "dart":
            if "drop_rate" in space:
                float_params["drop_rate"] = _sample_float(trial, "drop_rate", space["drop_rate"])
            if "skip_drop" in space:
                float_params["skip_drop"] = _sample_float(trial, "skip_drop", space["skip_drop"])
            if "feature_fraction" in space:
                float_params["feature_fraction"] = _sample_float(
                    trial, "feature_fraction", space["feature_fraction"]
                )

    return int_params, float_params, string_params


def _sample_cleargbm_params(
    trial: OptunaTrialProtocol,
    space: ClearGBMSearchSpace,
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Sample ClearGBM hyperparameters from search space."""
    int_params: SampledIntParams = {
        "n_estimators": _sample_int(trial, "n_estimators", space["n_estimators"]),
        "max_depth": _sample_int(trial, "max_depth", space["max_depth"]),
        "min_samples_split": _sample_int(trial, "min_samples_split", space["min_samples_split"]),
        "min_samples_leaf": _sample_int(trial, "min_samples_leaf", space["min_samples_leaf"]),
        "max_bins": _sample_int(trial, "max_bins", space["max_bins"]),
    }
    float_params: SampledFloatParams = {
        "learning_rate": _sample_float(trial, "learning_rate", space["learning_rate"]),
        "subsample": _sample_float(trial, "subsample", space["subsample"]),
    }
    string_params: SampledStringParams = {}
    return int_params, float_params, string_params


def _sample_random_forest_params(
    trial: OptunaTrialProtocol,
    space: RandomForestSearchSpace,
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Sample RandomForest hyperparameters from search space.

    RandomForest samples no learning rate: it is not a boosted method. Sending
    its space to the XGBoost sampler, which every RandomForest run did while
    is_xgboost_search_space matched on max_depth alone, failed on exactly that
    missing key.
    """
    int_params: SampledIntParams = {
        "n_estimators": _sample_int(trial, "n_estimators", space["n_estimators"]),
        "max_depth": _sample_int(trial, "max_depth", space["max_depth"]),
        "min_samples_split": _sample_int(trial, "min_samples_split", space["min_samples_split"]),
        "min_samples_leaf": _sample_int(trial, "min_samples_leaf", space["min_samples_leaf"]),
    }
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {
        "max_features": _sample_string(trial, "max_features", space["max_features"]),
    }
    return int_params, float_params, string_params


def _sample_logreg_params(
    trial: OptunaTrialProtocol,
    space: LogRegSearchSpace,
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Sample LogReg hyperparameters from search space."""
    int_params: SampledIntParams = {
        "max_iter": _sample_int(trial, "max_iter", space["max_iter"]),
    }
    float_params: SampledFloatParams = {
        "C": _sample_float(trial, "C", space["C"]),
        "tol": _sample_float(trial, "tol", space["tol"]),
    }
    string_params: SampledStringParams = {}
    if "penalty" in space:
        string_params["penalty"] = _sample_string(trial, "penalty", space["penalty"])
    if "solver" in space:
        string_params["solver"] = _sample_string(trial, "solver", space["solver"])
    if "l1_ratio" in space:
        float_params["l1_ratio"] = _sample_float(trial, "l1_ratio", space["l1_ratio"])
    return int_params, float_params, string_params


def _sample_params(
    trial: OptunaTrialProtocol,
    search_space: SearchSpace,
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Sample parameters based on search space type."""
    if is_xgboost_search_space(search_space):
        return _sample_xgboost_params(trial, search_space)
    if is_mlp_search_space(search_space):
        return _sample_mlp_params(trial, search_space)
    if is_lstm_search_space(search_space):
        return _sample_lstm_params(trial, search_space)
    if is_cleargbm_search_space(search_space):
        return _sample_cleargbm_params(trial, search_space)
    if is_random_forest_search_space(search_space):
        return _sample_random_forest_params(trial, search_space)
    if is_logreg_search_space(search_space):
        return _sample_logreg_params(trial, search_space)
    # LightGBM is the only remaining space type. Every guard above keys off a
    # field unique to its own space, so the order of these branches does not
    # matter -- which is the property that was missing when is_xgboost matched
    # on max_depth alone and swallowed RandomForest and ClearGBM.
    assert is_lightgbm_search_space(search_space)
    return _sample_lightgbm_params(trial, search_space)


def _extract_xgboost_best_params(
    best_params: dict[str, float | int | str],
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Extract best parameters for XGBoost."""
    int_params: SampledIntParams = {
        "max_depth": int(best_params["max_depth"]),
        "n_estimators": int(best_params["n_estimators"]),
    }
    float_params: SampledFloatParams = {
        "learning_rate": float(best_params["learning_rate"]),
        "reg_alpha": float(best_params["reg_alpha"]),
        "reg_lambda": float(best_params["reg_lambda"]),
        "subsample": float(best_params["subsample"]),
        "colsample_bytree": float(best_params["colsample_bytree"]),
    }
    string_params: SampledStringParams = {}

    if "booster" in best_params:
        string_params["booster"] = str(best_params["booster"])
        if best_params.get("booster") == "dart":
            if "rate_drop" in best_params:
                float_params["rate_drop"] = float(best_params["rate_drop"])
            if "skip_drop" in best_params:
                float_params["skip_drop"] = float(best_params["skip_drop"])

    return int_params, float_params, string_params


def _extract_mlp_best_params(
    best_params: dict[str, float | int | str],
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Extract best parameters for MLP."""
    int_params: SampledIntParams = {
        "n_layers": int(best_params["n_layers"]),
        "hidden_size": int(best_params["hidden_size"]),
        "batch_size": int(best_params["batch_size"]),
    }
    float_params: SampledFloatParams = {
        "learning_rate": float(best_params["learning_rate"]),
        "dropout": float(best_params["dropout"]),
    }
    return int_params, float_params, {}


def _extract_lstm_best_params(
    best_params: dict[str, float | int | str],
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Extract best parameters for LSTM."""
    int_params: SampledIntParams = {
        "hidden_size": int(best_params["hidden_size"]),
        "num_layers": int(best_params["num_layers"]),
        "batch_size": int(best_params["batch_size"]),
    }
    float_params: SampledFloatParams = {
        "learning_rate": float(best_params["learning_rate"]),
        "dropout": float(best_params["dropout"]),
    }
    return int_params, float_params, {}


def _extract_lightgbm_best_params(
    best_params: dict[str, float | int | str],
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Extract best parameters for LightGBM."""
    int_params: SampledIntParams = {
        "n_estimators": int(best_params["n_estimators"]),
        "num_leaves": int(best_params["num_leaves"]),
    }
    float_params: SampledFloatParams = {
        "learning_rate": float(best_params["learning_rate"]),
        "subsample": float(best_params["subsample"]),
        "colsample_bytree": float(best_params["colsample_bytree"]),
        "reg_alpha": float(best_params["reg_alpha"]),
        "reg_lambda": float(best_params["reg_lambda"]),
    }
    string_params: SampledStringParams = {}

    if "boosting_type" in best_params:
        string_params["boosting_type"] = str(best_params["boosting_type"])
        if best_params.get("boosting_type") == "dart":
            if "drop_rate" in best_params:
                float_params["drop_rate"] = float(best_params["drop_rate"])
            if "skip_drop" in best_params:
                float_params["skip_drop"] = float(best_params["skip_drop"])
            if "feature_fraction" in best_params:
                float_params["feature_fraction"] = float(best_params["feature_fraction"])

    return int_params, float_params, string_params


def _extract_cleargbm_best_params(
    best_params: dict[str, float | int | str],
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Extract best parameters for ClearGBM."""
    int_params: SampledIntParams = {
        "n_estimators": int(best_params["n_estimators"]),
        "max_depth": int(best_params["max_depth"]),
        "min_samples_split": int(best_params["min_samples_split"]),
        "min_samples_leaf": int(best_params["min_samples_leaf"]),
        "max_bins": int(best_params["max_bins"]),
    }
    float_params: SampledFloatParams = {
        "learning_rate": float(best_params["learning_rate"]),
        "subsample": float(best_params["subsample"]),
    }
    string_params: SampledStringParams = {}
    return int_params, float_params, string_params


def _extract_random_forest_best_params(
    best_params: dict[str, float | int | str],
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Extract best parameters for RandomForest."""
    int_params: SampledIntParams = {
        "n_estimators": int(best_params["n_estimators"]),
        "max_depth": int(best_params["max_depth"]),
        "min_samples_split": int(best_params["min_samples_split"]),
        "min_samples_leaf": int(best_params["min_samples_leaf"]),
    }
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {
        "max_features": str(best_params["max_features"]),
    }
    return int_params, float_params, string_params


def _extract_logreg_best_params(
    best_params: dict[str, float | int | str],
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Extract best parameters for LogReg."""
    int_params: SampledIntParams = {
        "max_iter": int(best_params["max_iter"]),
    }
    float_params: SampledFloatParams = {
        "C": float(best_params["C"]),
        "tol": float(best_params["tol"]),
    }
    string_params: SampledStringParams = {}
    if "penalty" in best_params:
        string_params["penalty"] = str(best_params["penalty"])
    if "solver" in best_params:
        string_params["solver"] = str(best_params["solver"])
    if "l1_ratio" in best_params:
        float_params["l1_ratio"] = float(best_params["l1_ratio"])
    return int_params, float_params, string_params


def _extract_best_params(
    search_space: SearchSpace,
    best_params: dict[str, float | int | str],
) -> tuple[SampledIntParams, SampledFloatParams, SampledStringParams]:
    """Extract best parameters from study results based on search space type."""
    if is_xgboost_search_space(search_space):
        return _extract_xgboost_best_params(best_params)
    if is_mlp_search_space(search_space):
        return _extract_mlp_best_params(best_params)
    if is_lstm_search_space(search_space):
        return _extract_lstm_best_params(best_params)
    if is_cleargbm_search_space(search_space):
        return _extract_cleargbm_best_params(best_params)
    if is_random_forest_search_space(search_space):
        return _extract_random_forest_best_params(best_params)
    if is_logreg_search_space(search_space):
        return _extract_logreg_best_params(best_params)
    # LightGBM is the only remaining space type; see _sample_params.
    assert is_lightgbm_search_space(search_space)
    return _extract_lightgbm_best_params(best_params)
