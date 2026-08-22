"""Tests for covenant_ml types module."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml import TrainConfig
from covenant_ml.types import (
    DMatrixProtocol,
    EvalMetrics,
    FeatureImportance,
    TrainOutcome,
    TrainProgress,
    XGBBoosterProtocol,
    XGBClassifierFactory,
    XGBClassifierLoader,
    XGBModelProtocol,
    XGBParams,
)
from covenant_ml.types_regression import (
    RegressionMetrics,
    RegressionTrainOutcome,
    RegressionTrainProgress,
    RegressorBackendName,
)


def test_train_config_has_required_keys() -> None:
    """TrainConfig TypedDict has all required keys."""
    config: TrainConfig = {
        "device": "cpu",
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "early_stopping_rounds": 10,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
    }

    assert config["device"] == "cpu"
    assert config["learning_rate"] == 0.1
    assert config["max_depth"] == 6
    assert config["n_estimators"] == 100
    assert config["subsample"] == 0.8
    assert config["colsample_bytree"] == 0.8
    assert config["random_state"] == 42
    assert config["train_ratio"] == 0.7
    assert config["val_ratio"] == 0.15
    assert config["test_ratio"] == 0.15
    assert config["early_stopping_rounds"] == 10
    assert config["reg_alpha"] == 1.0
    assert config["reg_lambda"] == 5.0


def test_train_config_values_are_correct_types() -> None:
    """TrainConfig values have correct types at runtime."""
    config: TrainConfig = {
        "device": "cpu",
        "learning_rate": 0.05,
        "max_depth": 4,
        "n_estimators": 50,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "random_state": 123,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "early_stopping_rounds": 10,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
    }

    assert config["device"] in ("cpu", "cuda", "auto")
    # Verify float operations work
    assert config["learning_rate"] * 2 == 0.1
    assert config["subsample"] + 0.1 == 1.0

    # Verify int operations work
    assert config["max_depth"] + 1 == 5
    assert config["n_estimators"] // 10 == 5

    # Verify split ratio sum
    assert config["train_ratio"] + config["val_ratio"] + config["test_ratio"] == 1.0


class _FakeBooster:
    def __init__(self) -> None:
        self.saved: list[str] = []

    def save_model(self, fname: str) -> None:
        self.saved.append(fname)

    def predict(self, data: DMatrixProtocol) -> NDArray[np.float32]:
        _ = data
        result: NDArray[np.float32] = np.zeros(1, dtype=np.float32)
        result[0] = 0.5
        return result


class _FakeXGBModel:
    def __init__(
        self,
        *,
        n_jobs: int,
        tree_method: str,
        device: str,
        reg_alpha: float,
        reg_lambda: float,
    ) -> None:
        self.booster = _FakeBooster()
        self.params: XGBParams = XGBParams(
            n_jobs=n_jobs,
            tree_method=tree_method,
            device=device,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )
        self._feature_importances: NDArray[np.float32] = np.zeros(2, dtype=np.float32)
        self._feature_importances[0] = 0.5
        self._feature_importances[1] = 0.5

    @property
    def feature_importances_(self) -> NDArray[np.float32]:
        return self._feature_importances

    def fit(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        *,
        verbose: bool = False,
    ) -> XGBModelProtocol:
        _ = x_features, y_labels, verbose
        return self

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        _ = x
        result: NDArray[np.float64] = np.zeros((1, 2), dtype=np.float64)
        return result

    def get_xgb_params(self) -> XGBParams:
        return self.params

    def save_model(self, fname: str) -> None:
        self.booster.save_model(fname)

    def load_model(self, fname: str) -> None:
        _ = fname

    def get_booster(self) -> XGBBoosterProtocol:
        return self.booster


def _fake_classifier_factory(
    *,
    learning_rate: float,
    max_depth: int,
    n_estimators: int,
    subsample: float,
    colsample_bytree: float,
    random_state: int,
    objective: str,
    eval_metric: str,
    n_jobs: int,
    tree_method: str,
    device: str,
    scale_pos_weight: float | None = None,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
) -> XGBModelProtocol:
    _ = (
        learning_rate,
        max_depth,
        n_estimators,
        subsample,
        colsample_bytree,
        random_state,
        objective,
        eval_metric,
        n_jobs,
        tree_method,
        device,
        scale_pos_weight,
        reg_alpha,
        reg_lambda,
    )
    return _FakeXGBModel(
        n_jobs=n_jobs,
        tree_method=tree_method,
        device=device,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
    )


def _fake_loader() -> XGBModelProtocol:
    return _FakeXGBModel(
        n_jobs=2,
        tree_method="hist",
        device="cpu",
        reg_alpha=1.0,
        reg_lambda=5.0,
    )


def test_protocols_are_callable() -> None:
    """Ensure Protocol signatures are runtime-callable for coverage."""
    train_req: TrainConfig = {
        "device": "cpu",
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 1,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "early_stopping_rounds": 5,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
    }
    train_out: TrainOutcome = {
        "model_path": "p",
        "model_id": "id",
        "samples_total": 1,
        "samples_train": 1,
        "samples_val": 0,
        "samples_test": 0,
        "train_metrics": EvalMetrics(
            loss=0.0,
            ppl=1.0,
            auc=0.0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
        ),
        "val_metrics": EvalMetrics(
            loss=0.0,
            ppl=1.0,
            auc=0.0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
        ),
        "test_metrics": EvalMetrics(
            loss=0.0,
            ppl=1.0,
            auc=0.0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
        ),
        "best_val_auc": 0.0,
        "best_round": 1,
        "total_rounds": 1,
        "early_stopped": False,
        "config": train_req,
        "feature_importances": [
            FeatureImportance(name="feat1", importance=0.6, rank=1),
            FeatureImportance(name="feat2", importance=0.4, rank=2),
        ],
        "scale_pos_weight_computed": 1.0,
    }
    progress: TrainProgress = {
        "round": 1,
        "total_rounds": 1,
        "train_loss": 0.0,
        "train_auc": 0.0,
        "val_loss": None,
        "val_auc": None,
    }

    model = _FakeXGBModel(
        n_jobs=2,
        tree_method="hist",
        device="cpu",
        reg_alpha=1.0,
        reg_lambda=5.0,
    )
    booster = model.get_booster()
    booster.save_model("x.ubj")
    params = model.get_xgb_params()
    assert params["tree_method"] == "hist"
    assert params["device"] == "cpu"
    assert progress["round"] == 1
    assert train_out["model_id"] != ""

    factory: XGBClassifierFactory = _fake_classifier_factory
    loader: XGBClassifierLoader = _fake_loader
    created_model = factory(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        tree_method="hist",
        device="cpu",
        scale_pos_weight=None,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )
    loaded_model = loader()
    created_model.save_model("y.ubj")
    loaded_model.load_model("y.ubj")


# =============================================================================
# Regression Type Tests
# =============================================================================


def test_regressor_backend_name_values() -> None:
    """RegressorBackendName accepts all four regressor backends."""
    names: list[RegressorBackendName] = [
        "xgboost_reg",
        "lightgbm_reg",
        "mlp_reg",
        "lstm_reg",
    ]
    assert len(names) == 4
    assert "xgboost_reg" in names
    assert "lightgbm_reg" in names
    assert "mlp_reg" in names
    assert "lstm_reg" in names


def test_regression_metrics_has_all_fields() -> None:
    """RegressionMetrics TypedDict has all required fields."""
    metrics: RegressionMetrics = {
        "mse": 0.25,
        "rmse": 0.5,
        "mae": 0.4,
        "r_squared": 0.85,
        "mape": 0.12,
    }

    assert metrics["mse"] == 0.25
    assert metrics["rmse"] == 0.5
    assert metrics["mae"] == 0.4
    assert metrics["r_squared"] == 0.85
    assert metrics["mape"] == 0.12


def test_regression_train_progress_with_val() -> None:
    """RegressionTrainProgress works with validation RMSE."""
    progress: RegressionTrainProgress = {
        "round": 5,
        "total_rounds": 100,
        "train_rmse": 0.123,
        "val_rmse": 0.145,
    }

    assert progress["round"] == 5
    assert progress["total_rounds"] == 100
    assert progress["train_rmse"] == 0.123
    assert progress["val_rmse"] == 0.145


def test_regression_train_progress_without_val() -> None:
    """RegressionTrainProgress works with None validation RMSE."""
    progress: RegressionTrainProgress = {
        "round": 1,
        "total_rounds": 50,
        "train_rmse": 0.5,
        "val_rmse": None,
    }

    assert progress["val_rmse"] is None


def test_regression_train_outcome_with_xgboost_config() -> None:
    """RegressionTrainOutcome accepts TrainConfig (XGBoost regressor)."""
    config: TrainConfig = {
        "device": "cpu",
        "learning_rate": 0.1,
        "max_depth": 6,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "early_stopping_rounds": 10,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
    metrics: RegressionMetrics = {
        "mse": 0.01,
        "rmse": 0.1,
        "mae": 0.08,
        "r_squared": 0.95,
        "mape": 0.03,
    }
    outcome: RegressionTrainOutcome = {
        "model_path": "/tmp/model.ubj",
        "model_id": "reg-001",
        "samples_total": 1000,
        "samples_train": 700,
        "samples_val": 150,
        "samples_test": 150,
        "train_metrics": metrics,
        "val_metrics": metrics,
        "test_metrics": metrics,
        "best_val_rmse": 0.1,
        "best_round": 42,
        "total_rounds": 100,
        "early_stopped": True,
        "config": config,
        "feature_importances": [
            FeatureImportance(name="feat1", importance=0.6, rank=1),
        ],
    }

    assert outcome["model_id"] == "reg-001"
    assert outcome["best_val_rmse"] == 0.1
    assert outcome["early_stopped"] is True
    assert outcome["train_metrics"]["r_squared"] == 0.95
    assert len(outcome["feature_importances"]) == 1
