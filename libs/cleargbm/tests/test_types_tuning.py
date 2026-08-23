"""Tests for cleargbm.types: training progress and tuning reports."""

from __future__ import annotations

import pytest

from cleargbm.types import (
    GradientBoostingConfig,
    JSONDict,
    JSONTypeError,
    TimingResult,
    TrainingProgress,
    TuningReport,
    decode_timing_result,
    decode_training_progress,
    decode_tuning_report,
    encode_timing_result,
    encode_training_progress,
    encode_tuning_report,
)

# =============================================================================
# TrainingProgress Tests
# =============================================================================


class TestTrainingProgress:
    """Tests for TrainingProgress encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode should preserve data."""
        original: TrainingProgress = {
            "tree_index": 50,
            "total_trees": 100,
            "train_loss": 0.35,
            "val_loss": 0.42,
        }
        encoded = encode_training_progress(original)
        decoded = decode_training_progress(encoded)

        assert decoded["tree_index"] == 50
        assert decoded["total_trees"] == 100
        assert decoded["train_loss"] == 0.35
        assert decoded["val_loss"] == 0.42

    def test_encode_decode_with_none_val_loss(self) -> None:
        """None val_loss should roundtrip."""
        original: TrainingProgress = {
            "tree_index": 10,
            "total_trees": 50,
            "train_loss": 0.5,
            "val_loss": None,
        }
        encoded = encode_training_progress(original)
        decoded = decode_training_progress(encoded)

        assert decoded["val_loss"] is None


# =============================================================================
# TimingResult Tests
# =============================================================================


class TestTimingResult:
    """Tests for TimingResult encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode should preserve data."""
        original: TimingResult = {
            "n_jobs": 4,
            "max_bins": 64,
            "max_depth": 4,
            "learning_rate": 0.1,
            "elapsed_seconds": 1.5,
            "trees_per_second": 3.33,
        }
        encoded = encode_timing_result(original)
        decoded = decode_timing_result(encoded)

        assert decoded["n_jobs"] == 4
        assert decoded["max_bins"] == 64
        assert decoded["max_depth"] == 4
        assert decoded["learning_rate"] == 0.1
        assert decoded["elapsed_seconds"] == 1.5
        assert decoded["trees_per_second"] == 3.33

    def test_decode_n_jobs_minus_one(self) -> None:
        """n_jobs=-1 (all cores) should decode correctly."""
        raw: JSONDict = {
            "n_jobs": -1,
            "max_bins": 64,
            "max_depth": 4,
            "learning_rate": 0.1,
            "elapsed_seconds": 1.0,
            "trees_per_second": 5.0,
        }
        decoded = decode_timing_result(raw)
        assert decoded["n_jobs"] == -1

    def test_decode_invalid_n_jobs(self) -> None:
        """Invalid n_jobs should raise ValueError."""
        raw: JSONDict = {
            "n_jobs": 0,
            "max_bins": 64,
            "max_depth": 4,
            "learning_rate": 0.1,
            "elapsed_seconds": 1.0,
            "trees_per_second": 5.0,
        }
        with pytest.raises(ValueError, match="n_jobs must be -1 or positive"):
            decode_timing_result(raw)

    def test_decode_invalid_max_bins(self) -> None:
        """Invalid max_bins should raise ValueError."""
        raw: JSONDict = {
            "n_jobs": 1,
            "max_bins": 0,
            "max_depth": 4,
            "learning_rate": 0.1,
            "elapsed_seconds": 1.0,
            "trees_per_second": 5.0,
        }
        with pytest.raises(ValueError, match="max_bins must be positive"):
            decode_timing_result(raw)

    def test_decode_negative_elapsed_seconds(self) -> None:
        """Negative elapsed_seconds should raise ValueError."""
        raw: JSONDict = {
            "n_jobs": 1,
            "max_bins": 64,
            "max_depth": 4,
            "learning_rate": 0.1,
            "elapsed_seconds": -1.0,
            "trees_per_second": 5.0,
        }
        with pytest.raises(ValueError, match="elapsed_seconds must be non-negative"):
            decode_timing_result(raw)

    def test_decode_missing_key(self) -> None:
        """Missing key should raise KeyError."""
        raw: JSONDict = {
            "n_jobs": 1,
            "max_bins": 64,
            # missing max_depth
            "learning_rate": 0.1,
            "elapsed_seconds": 1.0,
            "trees_per_second": 5.0,
        }
        with pytest.raises(KeyError):
            decode_timing_result(raw)


# =============================================================================
# TuningReport Tests
# =============================================================================


class TestTuningReport:
    """Tests for TuningReport encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode should preserve data."""
        config: GradientBoostingConfig = {
            "n_estimators": 5,
            "max_depth": 4,
            "learning_rate": 0.1,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "n_classes": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 42,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 2,
            "early_stopping_rounds": None,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        timing_result: TimingResult = {
            "n_jobs": 2,
            "max_bins": 64,
            "max_depth": 4,
            "learning_rate": 0.1,
            "elapsed_seconds": 1.5,
            "trees_per_second": 3.33,
        }
        original: TuningReport = {
            "best_config": config,
            "timing_results": (timing_result,),
            "sample_size": 1000,
            "n_features": 10,
            "recommended_n_jobs": 2,
            "recommended_max_bins": 64,
            "parallel_speedup": 1.8,
            "total_tune_time_seconds": 30.5,
        }
        encoded = encode_tuning_report(original)
        decoded = decode_tuning_report(encoded)

        assert decoded["best_config"]["n_jobs"] == 2
        assert len(decoded["timing_results"]) == 1
        assert decoded["timing_results"][0]["n_jobs"] == 2
        assert decoded["sample_size"] == 1000
        assert decoded["n_features"] == 10
        assert decoded["recommended_n_jobs"] == 2
        assert decoded["recommended_max_bins"] == 64
        assert decoded["parallel_speedup"] == 1.8
        assert decoded["total_tune_time_seconds"] == 30.5

    def test_decode_timing_results_not_list(self) -> None:
        """timing_results not a list should raise JSONTypeError."""
        config_raw: JSONDict = {
            "n_estimators": 5,
            "max_depth": 4,
            "learning_rate": 0.1,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "n_classes": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 42,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        raw: JSONDict = {
            "best_config": config_raw,
            "timing_results": "not a list",
            "sample_size": 100,
            "n_features": 5,
            "recommended_n_jobs": 1,
            "recommended_max_bins": 64,
            "parallel_speedup": 1.0,
            "total_tune_time_seconds": 10.0,
        }
        with pytest.raises(JSONTypeError, match="timing_results must be list"):
            decode_tuning_report(raw)

    def test_decode_timing_result_not_dict(self) -> None:
        """timing_results item not a dict should raise JSONTypeError."""
        config_raw: JSONDict = {
            "n_estimators": 5,
            "max_depth": 4,
            "learning_rate": 0.1,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "n_classes": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 42,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        raw: JSONDict = {
            "best_config": config_raw,
            "timing_results": ["not a dict"],
            "sample_size": 100,
            "n_features": 5,
            "recommended_n_jobs": 1,
            "recommended_max_bins": 64,
            "parallel_speedup": 1.0,
            "total_tune_time_seconds": 10.0,
        }
        with pytest.raises(JSONTypeError, match=r"timing_results\[0\] must be dict"):
            decode_tuning_report(raw)

    def test_decode_invalid_sample_size(self) -> None:
        """Invalid sample_size should raise ValueError."""
        config_raw: JSONDict = {
            "n_estimators": 5,
            "max_depth": 4,
            "learning_rate": 0.1,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "n_classes": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 42,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        raw: JSONDict = {
            "best_config": config_raw,
            "timing_results": [],
            "sample_size": 0,  # invalid
            "n_features": 5,
            "recommended_n_jobs": 1,
            "recommended_max_bins": 64,
            "parallel_speedup": 1.0,
            "total_tune_time_seconds": 10.0,
        }
        with pytest.raises(ValueError, match="sample_size must be positive"):
            decode_tuning_report(raw)

    def test_decode_negative_parallel_speedup(self) -> None:
        """Negative parallel_speedup should raise ValueError."""
        config_raw: JSONDict = {
            "n_estimators": 5,
            "max_depth": 4,
            "learning_rate": 0.1,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "colsample_bytree": None,
            "categorical_features": None,
            "n_classes": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 42,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        raw: JSONDict = {
            "best_config": config_raw,
            "timing_results": [],
            "sample_size": 100,
            "n_features": 5,
            "recommended_n_jobs": 1,
            "recommended_max_bins": 64,
            "parallel_speedup": -0.5,  # invalid
            "total_tune_time_seconds": 10.0,
        }
        with pytest.raises(ValueError, match="parallel_speedup must be non-negative"):
            decode_tuning_report(raw)

    def test_decode_best_config_not_dict(self) -> None:
        """best_config not a dict should raise JSONTypeError."""
        raw: JSONDict = {
            "best_config": "not a dict",
            "timing_results": [],
            "sample_size": 100,
            "n_features": 5,
            "recommended_n_jobs": 1,
            "recommended_max_bins": 64,
            "parallel_speedup": 1.0,
            "total_tune_time_seconds": 10.0,
        }
        with pytest.raises(JSONTypeError, match="best_config must be dict"):
            decode_tuning_report(raw)
