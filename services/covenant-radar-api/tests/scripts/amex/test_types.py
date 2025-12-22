"""Tests for AMEX pipeline types."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scripts.amex.types import (
    AMEXPipelineConfig,
    EnsembleResult,
    ModelOOFResult,
    PipelineResult,
    make_default_config,
)


def _float_array(*values: float) -> NDArray[np.float64]:
    """Create typed float64 array from values.

    Args:
        *values: Float values for the array.

    Returns:
        NDArray of float64.
    """
    return np.array(values, dtype=np.float64)


def _int_array(*values: int) -> NDArray[np.int64]:
    """Create typed int64 array from values.

    Args:
        *values: Int values for the array.

    Returns:
        NDArray of int64.
    """
    return np.array(values, dtype=np.int64)


class TestAMEXPipelineConfig:
    """Tests for AMEXPipelineConfig TypedDict."""

    def test_create_config(self) -> None:
        """AMEXPipelineConfig can be created with all fields."""
        config = AMEXPipelineConfig(
            backends=("lightgbm", "xgboost"),
            n_folds=5,
            n_estimators=1000,
            learning_rate=0.05,
            aggregation="statistics",
            include_rank_features=True,
            include_diff_features=True,
            include_window_features=True,
            window_sizes=(3, 6),
            random_state=42,
        )

        assert config["backends"] == ("lightgbm", "xgboost")
        assert config["n_folds"] == 5
        assert config["n_estimators"] == 1000
        assert config["learning_rate"] == 0.05
        assert config["aggregation"] == "statistics"
        assert config["include_rank_features"] is True
        assert config["include_diff_features"] is True
        assert config["include_window_features"] is True
        assert config["window_sizes"] == (3, 6)
        assert config["random_state"] == 42


class TestModelOOFResult:
    """Tests for ModelOOFResult TypedDict."""

    def test_create_model_oof_result(self) -> None:
        """ModelOOFResult can be created with all fields."""
        predictions = _float_array(0.1, 0.9, 0.5)
        fold_indices = _int_array(0, 0, 1)

        result = ModelOOFResult(
            model_name="lightgbm",
            oof_predictions=predictions,
            fold_indices=fold_indices,
            cv_scores=(0.81, 0.82),
            mean_cv_score=0.815,
        )

        assert result["model_name"] == "lightgbm"
        assert len(result["oof_predictions"]) == 3
        assert len(result["fold_indices"]) == 3
        assert result["cv_scores"] == (0.81, 0.82)
        assert result["mean_cv_score"] == 0.815


class TestEnsembleResult:
    """Tests for EnsembleResult TypedDict."""

    def test_create_ensemble_result(self) -> None:
        """EnsembleResult can be created with all fields."""
        result = EnsembleResult(
            model_names=("lightgbm", "xgboost"),
            weights=(0.6, 0.4),
            initial_score=0.80,
            optimized_score=0.82,
            improvement=0.02,
        )

        assert result["model_names"] == ("lightgbm", "xgboost")
        assert result["weights"] == (0.6, 0.4)
        assert result["initial_score"] == 0.80
        assert result["optimized_score"] == 0.82
        assert result["improvement"] == 0.02


class TestPipelineResult:
    """Tests for PipelineResult TypedDict."""

    def test_create_pipeline_result(self) -> None:
        """PipelineResult can be created with all fields."""
        model_result = ModelOOFResult(
            model_name="lightgbm",
            oof_predictions=_float_array(0.5, 0.5),
            fold_indices=_int_array(0, 1),
            cv_scores=(0.81,),
            mean_cv_score=0.81,
        )
        ensemble_result = EnsembleResult(
            model_names=("lightgbm",),
            weights=(1.0,),
            initial_score=0.80,
            optimized_score=0.81,
            improvement=0.01,
        )

        result = PipelineResult(
            n_samples_train=100,
            n_samples_test=50,
            n_features=20,
            model_results=(model_result,),
            ensemble_result=ensemble_result,
            submission_path="/path/to/submission.csv",
        )

        assert result["n_samples_train"] == 100
        assert result["n_samples_test"] == 50
        assert result["n_features"] == 20
        assert len(result["model_results"]) == 1
        assert result["submission_path"] == "/path/to/submission.csv"


class TestMakeDefaultConfig:
    """Tests for make_default_config function."""

    def test_returns_valid_config(self) -> None:
        """make_default_config returns a valid configuration."""
        config = make_default_config()

        assert config["backends"] == ("lightgbm", "xgboost")
        assert config["n_folds"] == 5
        assert config["n_estimators"] == 1000
        assert config["learning_rate"] == 0.05
        assert config["aggregation"] == "statistics"
        assert config["include_rank_features"] is True
        assert config["include_diff_features"] is True
        assert config["include_window_features"] is True
        assert config["window_sizes"] == (3, 6)
        assert config["random_state"] == 42
