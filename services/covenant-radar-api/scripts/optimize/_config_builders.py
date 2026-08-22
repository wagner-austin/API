"""Per-backend train-config builders for model saving."""

from __future__ import annotations

from typing import Literal

from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams, SampledStringParams
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


def _build_xgboost_config(
    int_params: SampledIntParams,
    float_params: SampledFloatParams,
) -> TrainConfig:
    """Build XGBoost training config from sampled parameters.

    Args:
        int_params: Sampled integer parameters from optimization.
        float_params: Sampled float parameters from optimization.

    Returns:
        XGBoost training configuration.
    """
    return TrainConfig(
        device="auto",
        learning_rate=float_params["learning_rate"],
        max_depth=int_params["max_depth"],
        n_estimators=int_params["n_estimators"],
        subsample=float_params["subsample"],
        colsample_bytree=float_params["colsample_bytree"],
        reg_alpha=float_params["reg_alpha"],
        reg_lambda=float_params["reg_lambda"],
        random_state=42,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        early_stopping_rounds=10,
    )


def _build_mlp_config(
    int_params: SampledIntParams,
    float_params: SampledFloatParams,
) -> MLPConfig:
    """Build MLP training config from sampled parameters.

    Args:
        int_params: Sampled integer parameters from optimization.
        float_params: Sampled float parameters from optimization.

    Returns:
        MLP training configuration.
    """
    n_layers = int_params["n_layers"]
    hidden_size = int_params["hidden_size"]
    return MLPConfig(
        device="auto",
        precision="fp32",
        optimizer="adamw",
        hidden_sizes=tuple(hidden_size for _ in range(n_layers)),
        learning_rate=float_params["learning_rate"],
        batch_size=int_params["batch_size"],
        n_epochs=50,
        dropout=float_params["dropout"],
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        early_stopping_patience=10,
    )


def _build_lstm_config(
    int_params: SampledIntParams,
    float_params: SampledFloatParams,
) -> LSTMConfig:
    """Build LSTM training config from sampled parameters.

    Args:
        int_params: Sampled integer parameters from optimization.
        float_params: Sampled float parameters from optimization.

    Returns:
        LSTM training configuration.
    """
    return LSTMConfig(
        device="auto",
        precision="fp32",
        hidden_size=int_params["hidden_size"],
        num_layers=int_params["num_layers"],
        dropout=float_params["dropout"],
        bidirectional=False,
        sequence_length=5,
        learning_rate=float_params["learning_rate"],
        batch_size=int_params["batch_size"],
        n_epochs=50,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        early_stopping_patience=10,
    )


def _build_lightgbm_config(
    int_params: SampledIntParams,
    float_params: SampledFloatParams,
) -> LightGBMConfig:
    """Build LightGBM training config from sampled parameters.

    Args:
        int_params: Sampled integer parameters from optimization.
        float_params: Sampled float parameters from optimization.

    Returns:
        LightGBM training configuration.
    """
    return LightGBMConfig(
        device="auto",
        learning_rate=float_params["learning_rate"],
        max_depth=int_params["max_depth"],
        n_estimators=int_params["n_estimators"],
        num_leaves=int_params["num_leaves"],
        min_child_samples=int_params["min_child_samples"],
        subsample=float_params["subsample"],
        colsample_bytree=float_params["colsample_bytree"],
        reg_alpha=float_params["reg_alpha"],
        reg_lambda=float_params["reg_lambda"],
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        early_stopping_rounds=10,
    )


def _build_cleargbm_config(
    int_params: SampledIntParams,
    float_params: SampledFloatParams,
) -> ClearGBMConfig:
    """Build ClearGBM training config from sampled parameters.

    Args:
        int_params: Sampled integer parameters from optimization.
        float_params: Sampled float parameters from optimization.

    Returns:
        ClearGBM training configuration.
    """
    return ClearGBMConfig(
        n_estimators=int_params["n_estimators"],
        max_depth=int_params["max_depth"],
        learning_rate=float_params["learning_rate"],
        min_samples_split=int_params["min_samples_split"],
        min_samples_leaf=int_params["min_samples_leaf"],
        max_features=None,
        max_bins=int_params["max_bins"],
        subsample=float_params["subsample"],
        random_state=42,
        track_contributions=False,
        monotonic_constraints=None,
        reg_alpha=float_params["reg_alpha"],
        reg_lambda=float_params["reg_lambda"],
        n_jobs=-1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        early_stopping_rounds=10,
    )


def _narrow_logreg_solver(raw: str) -> LogRegSolver:
    """Narrow string to LogRegSolver literal type.

    Args:
        raw: Solver name string.

    Returns:
        Validated LogRegSolver literal.

    Raises:
        ValueError: If solver name is not valid.
    """
    if raw == "lbfgs":
        return "lbfgs"
    if raw == "liblinear":
        return "liblinear"
    if raw == "newton-cg":
        return "newton-cg"
    if raw == "newton-cholesky":
        return "newton-cholesky"
    if raw == "sag":
        return "sag"
    if raw == "saga":
        return "saga"
    raise ValueError(f"Invalid LogReg solver: {raw}")


def _narrow_logreg_penalty(raw: str) -> LogRegPenalty:
    """Narrow string to LogRegPenalty literal type.

    Args:
        raw: Penalty name string.

    Returns:
        Validated LogRegPenalty literal.

    Raises:
        ValueError: If penalty name is not valid.
    """
    if raw == "l1":
        return "l1"
    if raw == "l2":
        return "l2"
    if raw == "elasticnet":
        return "elasticnet"
    if raw == "none":
        return "none"
    raise ValueError(f"Invalid LogReg penalty: {raw}")


def _build_logreg_config(
    float_params: SampledFloatParams,
    string_params: SampledStringParams,
) -> LogRegConfig:
    """Build Logistic Regression training config from sampled parameters.

    Args:
        float_params: Sampled float parameters from optimization.
        string_params: Sampled string parameters from optimization.

    Returns:
        LogReg training configuration.
    """
    solver = _narrow_logreg_solver(string_params["solver"])
    penalty = _narrow_logreg_penalty(string_params["penalty"])

    return LogRegConfig(
        solver=solver,
        penalty=penalty,
        C=float_params["C"],
        max_iter=1000,
        tol=float_params["tol"],
        class_weight_balanced=True,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        l1_ratio=float_params["l1_ratio"],
    )


def _build_random_forest_config(
    int_params: SampledIntParams,
    float_params: SampledFloatParams,
    string_params: SampledStringParams,
) -> RandomForestConfig:
    """Build Random Forest training config from sampled parameters.

    Args:
        int_params: Sampled integer parameters from optimization.
        float_params: Sampled float parameters from optimization.
        string_params: Sampled string parameters from optimization.

    Returns:
        RandomForest training configuration.
    """
    # Parse max_features — narrow to the literal union type
    max_features_val: Literal["sqrt", "log2"] | float | int | None
    max_features_str = string_params.get("max_features")
    if max_features_str == "sqrt":
        max_features_val = "sqrt"
    elif max_features_str == "log2":
        max_features_val = "log2"
    elif "max_features_float" in float_params:
        max_features_val = float_params["max_features_float"]
    else:
        max_features_val = "sqrt"

    return RandomForestConfig(
        n_estimators=int_params["n_estimators"],
        max_depth=int_params.get("max_depth"),
        min_samples_split=int_params["min_samples_split"],
        min_samples_leaf=int_params["min_samples_leaf"],
        max_features=max_features_val,
        bootstrap=True,
        class_weight_balanced=True,
        n_jobs=-1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        oob_score=False,
    )
