"""Tests for scripts/optimize/modes.py - run modes."""

from __future__ import annotations

import pytest
from covenant_ml.types import TrainConfig
from platform_core.logging import setup_rich_logging
from scripts._test_hooks import XGBoostOptimizationResult
from scripts.optimize.cli import DatasetName, FeaturePreset
from scripts.optimize.history import XGBoostHistoryEntry
from scripts.optimize.modes import _print_multi_dataset_summary, _print_preset_comparison_summary
from scripts.optimize.runner import RunResult


@pytest.fixture(autouse=True)
def setup_logging() -> None:
    """Ensure rich logging is setup for all tests in this module."""
    setup_rich_logging(level="WARNING", show_time=False)


def _make_optimization_result(
    dataset: DatasetName = "taiwan",
    best_val_auc: float = 0.85,
    n_samples: int = 1000,
    n_features: int = 100,
) -> XGBoostOptimizationResult:
    """Create a test XGBoostOptimizationResult."""
    return XGBoostOptimizationResult(
        backend="xgboost",
        status="complete",
        dataset=dataset,
        n_samples=n_samples,
        n_features=n_features,
        feature_preset="full",
        n_trials_complete=50,
        n_trials_pruned=5,
        n_trials_failed=0,
        best_trial_number=25,
        best_val_auc=best_val_auc,
        best_max_depth=6,
        best_n_estimators=100,
        best_learning_rate=0.1,
        best_reg_alpha=0.01,
        best_reg_lambda=0.01,
        best_subsample=0.8,
        best_colsample_bytree=0.8,
        duration_seconds=60.0,
        recommended_config=TrainConfig(
            device="cpu",
            learning_rate=0.1,
            max_depth=6,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            early_stopping_rounds=10,
            reg_alpha=0.01,
            reg_lambda=0.01,
        ),
    )


def _make_history_entry(
    dataset: str = "taiwan",
    feature_preset: str = "full",
    best_val_auc: float = 0.85,
) -> XGBoostHistoryEntry:
    """Create a test XGBoostHistoryEntry."""
    return XGBoostHistoryEntry(
        timestamp="2024-01-01T00:00:00Z",
        backend="xgboost",
        dataset=dataset,
        feature_preset=feature_preset,
        n_trials=50,
        n_samples=1000,
        n_features=100,
        best_val_auc=best_val_auc,
        best_trial_number=25,
        best_max_depth=6,
        best_n_estimators=100,
        best_learning_rate=0.1,
        best_reg_alpha=0.01,
        best_reg_lambda=0.01,
        best_subsample=0.8,
        best_colsample_bytree=0.8,
        duration_seconds=60.0,
    )


class TestPrintMultiDatasetSummary:
    """Tests for _print_multi_dataset_summary function."""

    def test_executes_with_no_history(self) -> None:
        """Test function executes when all_time_best is None (NEW path)."""
        result = _make_optimization_result(dataset="taiwan", best_val_auc=0.85)
        run_result: RunResult = RunResult(
            backend="xgboost",
            result=result,
            elapsed=60.0,
            previous_best=None,
            all_time_best=None,  # No history - should show NEW
            is_new_best=True,
        )

        dataset_name: DatasetName = "taiwan"
        results: list[tuple[DatasetName, RunResult]] = [(dataset_name, run_result)]

        # Should execute without error - covers the NEW delta path
        _print_multi_dataset_summary(results)

    def test_executes_with_positive_delta(self) -> None:
        """Test function executes with positive delta (improvement)."""
        result = _make_optimization_result(dataset="taiwan", best_val_auc=0.90)
        all_time_best = _make_history_entry(best_val_auc=0.85)

        run_result: RunResult = RunResult(
            backend="xgboost",
            result=result,
            elapsed=60.0,
            previous_best=None,
            all_time_best=all_time_best,
            is_new_best=True,
        )

        dataset_name: DatasetName = "taiwan"
        results: list[tuple[DatasetName, RunResult]] = [(dataset_name, run_result)]

        # Should execute without error - covers positive delta path
        _print_multi_dataset_summary(results)

    def test_executes_with_negative_delta(self) -> None:
        """Test function executes with negative delta (regression)."""
        result = _make_optimization_result(dataset="taiwan", best_val_auc=0.80)
        all_time_best = _make_history_entry(best_val_auc=0.85)

        run_result: RunResult = RunResult(
            backend="xgboost",
            result=result,
            elapsed=60.0,
            previous_best=None,
            all_time_best=all_time_best,
            is_new_best=False,
        )

        dataset_name: DatasetName = "taiwan"
        results: list[tuple[DatasetName, RunResult]] = [(dataset_name, run_result)]

        # Should execute without error - covers negative delta path
        _print_multi_dataset_summary(results)

    def test_executes_with_neutral_delta(self) -> None:
        """Test function executes with neutral delta (no significant change)."""
        result = _make_optimization_result(dataset="taiwan", best_val_auc=0.8501)
        all_time_best = _make_history_entry(best_val_auc=0.85)

        run_result: RunResult = RunResult(
            backend="xgboost",
            result=result,
            elapsed=60.0,
            previous_best=None,
            all_time_best=all_time_best,
            is_new_best=False,
        )

        dataset_name: DatasetName = "taiwan"
        results: list[tuple[DatasetName, RunResult]] = [(dataset_name, run_result)]

        # Should execute without error - covers neutral delta path
        _print_multi_dataset_summary(results)

    def test_executes_with_multiple_datasets(self) -> None:
        """Test function executes with multiple datasets."""
        result_taiwan = _make_optimization_result(dataset="taiwan", best_val_auc=0.85)
        result_us = _make_optimization_result(dataset="us", best_val_auc=0.90)
        result_polish = _make_optimization_result(dataset="polish", best_val_auc=0.88)

        results: list[tuple[DatasetName, RunResult]] = [
            (
                "taiwan",
                RunResult(
                    backend="xgboost",
                    result=result_taiwan,
                    elapsed=60.0,
                    previous_best=None,
                    all_time_best=None,
                    is_new_best=True,
                ),
            ),
            (
                "us",
                RunResult(
                    backend="xgboost",
                    result=result_us,
                    elapsed=70.0,
                    previous_best=None,
                    all_time_best=_make_history_entry(dataset="us", best_val_auc=0.88),
                    is_new_best=True,
                ),
            ),
            (
                "polish",
                RunResult(
                    backend="xgboost",
                    result=result_polish,
                    elapsed=80.0,
                    previous_best=None,
                    all_time_best=_make_history_entry(dataset="polish", best_val_auc=0.90),
                    is_new_best=False,
                ),
            ),
        ]

        # Should execute without error
        _print_multi_dataset_summary(results)


class TestPrintPresetComparisonSummary:
    """Tests for _print_preset_comparison_summary function."""

    def test_executes_with_all_presets(self) -> None:
        """Test function executes with all four presets."""
        results: list[tuple[FeaturePreset, float, int, float]] = [
            ("none", 0.80, 50, 30.0),
            ("log_only", 0.90, 60, 35.0),
            ("ratios_only", 0.85, 70, 40.0),
            ("full", 0.88, 100, 50.0),
        ]

        # Should execute without error - covers 1st, 2nd, 3rd, 4th ranking paths
        _print_preset_comparison_summary(results)

    def test_executes_with_tied_aucs(self) -> None:
        """Test function executes with tied AUC values."""
        results: list[tuple[FeaturePreset, float, int, float]] = [
            ("none", 0.85, 50, 30.0),
            ("log_only", 0.85, 60, 35.0),
            ("ratios_only", 0.85, 70, 40.0),
            ("full", 0.85, 100, 50.0),
        ]

        # Should execute without error
        _print_preset_comparison_summary(results)
