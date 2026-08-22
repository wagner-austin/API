"""Tests for worker/train_external_regression_job.py.

Tests use dependency injection via worker/_regression_hooks to verify actual
code paths. All code paths are tested with strong assertions on actual behavior.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from covenant_ml.types import (
    FeatureImportance,
)

from covenant_radar_api.worker.train_external_regression_job import (
    _build_lightgbm_reg_log,
    _build_xgboost_reg_log,
    _dispatch_regression_backend,
    _get_regression_active_filename,
    _get_regression_meta_filename,
    _importance_to_json,
    _regression_metrics_to_json,
    _write_regression_model_metadata,
)
from tests._train_external_regression_fixtures import (
    _make_fake_metrics,
)


class TestRegressionMetricsToJson:
    """Tests for _regression_metrics_to_json."""

    def test_converts_all_fields(self) -> None:
        """All RegressionMetrics fields are converted."""
        metrics = _make_fake_metrics()
        result = _regression_metrics_to_json(metrics)

        assert result["mse"] == 0.01
        assert result["rmse"] == 0.1
        assert result["mae"] == 0.08
        assert result["r_squared"] == 0.95
        assert result["mape"] == 5.0

    def test_returns_five_keys(self) -> None:
        """Result has exactly 5 keys."""
        metrics = _make_fake_metrics()
        result = _regression_metrics_to_json(metrics)
        assert len(result) == 5


class TestImportanceToJson:
    """Tests for _importance_to_json."""

    def test_converts_importance(self) -> None:
        """Converts FeatureImportance to JSON dict."""
        imp: FeatureImportance = {
            "name": "feature_0",
            "importance": 0.25,
            "rank": 1,
        }
        result = _importance_to_json(imp)
        assert result["name"] == "feature_0"
        assert result["importance"] == 0.25
        assert result["rank"] == 1


class TestBuildXGBoostRegLog:
    """Tests for _build_xgboost_reg_log."""

    def test_extracts_key_params(self) -> None:
        """Extracts key XGBoost parameters for logging."""
        from covenant_ml.types import TrainConfig

        config: TrainConfig = {
            "device": "cpu",
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 10,
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
        result = _build_xgboost_reg_log(config)
        assert result["learning_rate"] == 0.1
        assert result["n_estimators"] == 10
        assert result["max_depth"] == 3
        assert len(result) == 5


class TestBuildLightGBMRegLog:
    """Tests for _build_lightgbm_reg_log."""

    def test_extracts_key_params(self) -> None:
        """Extracts key LightGBM parameters for logging."""
        from covenant_ml.types import LightGBMConfig

        config: LightGBMConfig = {
            "device": "cpu",
            "learning_rate": 0.05,
            "max_depth": 5,
            "n_estimators": 100,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_state": 42,
            "early_stopping_rounds": 10,
        }
        result = _build_lightgbm_reg_log(config)
        assert result["num_leaves"] == 31
        assert result["reg_alpha"] == 0.1
        assert len(result) == 6


class TestGetRegressionActiveFilename:
    """Tests for _get_regression_active_filename."""

    def test_xgboost_reg(self) -> None:
        """XGBoost regressor returns .ubj filename."""
        assert _get_regression_active_filename("xgboost_reg") == "active_xgb_reg.ubj"

    def test_lightgbm_reg(self) -> None:
        """LightGBM regressor returns .txt filename."""
        assert _get_regression_active_filename("lightgbm_reg") == "active_lgbm_reg.txt"

    def test_unknown_raises_value_error(self) -> None:
        """Unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown regressor backend"):
            _get_regression_active_filename("mlp_reg")


class TestGetRegressionMetaFilename:
    """Tests for _get_regression_meta_filename."""

    def test_xgboost_reg_empty(self) -> None:
        """XGBoost regressor has no metadata (self-describing)."""
        assert _get_regression_meta_filename("xgboost_reg") == ""

    def test_lightgbm_reg(self) -> None:
        """LightGBM regressor has metadata."""
        assert _get_regression_meta_filename("lightgbm_reg") == "active_lgbm_reg_meta.json"


class TestWriteRegressionModelMetadata:
    """Tests for _write_regression_model_metadata."""

    def test_xgboost_reg_returns_none(self, tmp_path: Path) -> None:
        """XGBoost regressor returns None (no metadata needed)."""
        result = _write_regression_model_metadata("xgboost_reg", tmp_path)
        assert result is None

    def test_lightgbm_reg_writes_metadata(self, tmp_path: Path) -> None:
        """LightGBM regressor writes metadata file."""
        result = _write_regression_model_metadata("lightgbm_reg", tmp_path)
        expected_path = tmp_path / "active_lgbm_reg_meta.json"
        assert result == expected_path
        assert expected_path.exists()

        content = expected_path.read_text(encoding="utf-8")
        assert '"backend": "lightgbm_reg"' in content


class TestDispatchRegressionBackend:
    """Tests for _dispatch_regression_backend."""

    def test_xgboost_reg_dispatch(self) -> None:
        """XGBoost regressor dispatch returns log dict."""
        from covenant_ml.types import TrainConfig

        from covenant_radar_api.worker._train_external_regression_parsers import (
            XGBoostRegParseResult,
        )

        config: TrainConfig = {
            "device": "cpu",
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 10,
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
        parse_result: XGBoostRegParseResult = {
            "backend": "xgboost_reg",
            "config": config,
            "dataset": "financial_distress",
        }
        log_dict = _dispatch_regression_backend(parse_result)
        assert log_dict["learning_rate"] == 0.1

    def test_lightgbm_reg_dispatch(self) -> None:
        """LightGBM regressor dispatch returns log dict."""
        from covenant_ml.types import LightGBMConfig

        from covenant_radar_api.worker._train_external_regression_parsers import (
            LightGBMRegParseResult,
        )

        config: LightGBMConfig = {
            "device": "cpu",
            "learning_rate": 0.05,
            "max_depth": 5,
            "n_estimators": 100,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_state": 42,
            "early_stopping_rounds": 10,
        }
        parse_result: LightGBMRegParseResult = {
            "backend": "lightgbm_reg",
            "config": config,
            "dataset": "financial_distress",
        }
        log_dict = _dispatch_regression_backend(parse_result)
        assert log_dict["num_leaves"] == 31
