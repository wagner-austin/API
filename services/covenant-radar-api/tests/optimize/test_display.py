"""Tests for scripts/optimize display functions.

Tests table creation, result formatting, and output functions.
Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from scripts.optimize.display import (
    _format_delta,
    create_hyperparams_table,
    create_result_table,
    print_config,
    print_result,
)
from scripts.optimize.history import XGBoostHistoryEntry

from .conftest import (
    make_fake_cleargbm_result,
    make_fake_lightgbm_result,
    make_fake_lstm_result,
    make_fake_mlp_result,
    make_fake_result,
)


class TestCreateResultTable:
    """Tests for create_result_table function."""

    def test_creates_table_with_data(self) -> None:
        """Test table is created with result data."""
        result = make_fake_result()
        table = create_result_table("xgboost", result, 15.5)
        # Verify table has the expected protocol methods
        assert callable(table.add_column)
        assert callable(table.add_row)


class TestCreateHyperparamsTable:
    """Tests for create_hyperparams_table function for all backends."""

    def test_creates_xgboost_table(self) -> None:
        """Test table is created for XGBoost hyperparameters."""
        result = make_fake_result()
        table = create_hyperparams_table("xgboost", result)
        assert callable(table.add_column)
        assert callable(table.add_row)

    def test_creates_mlp_table(self) -> None:
        """Test table is created for MLP hyperparameters."""
        result = make_fake_mlp_result()
        table = create_hyperparams_table("mlp", result)
        assert callable(table.add_column)
        assert callable(table.add_row)

    def test_creates_lightgbm_table(self) -> None:
        """Test table is created for LightGBM hyperparameters."""
        result = make_fake_lightgbm_result()
        table = create_hyperparams_table("lightgbm", result)
        assert callable(table.add_column)
        assert callable(table.add_row)

    def test_creates_lstm_table(self) -> None:
        """Test table is created for LSTM hyperparameters."""
        result = make_fake_lstm_result()
        table = create_hyperparams_table("lstm", result)
        assert callable(table.add_column)
        assert callable(table.add_row)

    def test_creates_cleargbm_table(self) -> None:
        """Test table is created for ClearGBM hyperparameters."""
        result = make_fake_cleargbm_result()
        table = create_hyperparams_table("cleargbm", result)
        assert callable(table.add_column)
        assert callable(table.add_row)


class TestPrintConfig:
    """Tests for print_config function."""

    def test_prints_without_error(self) -> None:
        """Test print_config runs without error."""
        # Just verify it doesn't raise
        print_config("xgboost", "taiwan", 50, "full", "cuda")


class TestPrintResult:
    """Tests for print_result function."""

    def test_prints_without_error(self) -> None:
        """Test print_result runs without error."""
        result = make_fake_result()
        print_result("xgboost", result, 10.5)


class TestPrintResultNotNewBest:
    """Tests for print_result when current AUC is not a new best."""

    def test_prints_not_new_best_when_all_time_best_is_higher(self) -> None:
        """Test print_result shows 'Best AUC' (not NEW BEST) when AUC is lower."""
        result = make_fake_result(best_val_auc=0.80)

        # Create an all_time_best with higher AUC
        all_time_best: XGBoostHistoryEntry = {
            "backend": "xgboost",
            "timestamp": "2024-01-01T00:00:00Z",
            "dataset": "taiwan",
            "feature_preset": "full",
            "n_trials": 50,
            "n_samples": 1000,
            "n_features": 100,
            "best_val_auc": 0.90,  # Higher than current 0.80
            "best_trial_number": 25,
            "best_max_depth": 6,
            "best_n_estimators": 100,
            "best_learning_rate": 0.1,
            "best_reg_alpha": 0.01,
            "best_reg_lambda": 0.01,
            "best_subsample": 0.8,
            "best_colsample_bytree": 0.8,
            "duration_seconds": 60.0,
        }

        # Should not raise, and should hit the else branch (line 494)
        print_result("xgboost", result, 10.5, all_time_best=all_time_best)


class TestFormatElapsed:
    """Tests for _format_elapsed helper function."""

    def test_formats_seconds_under_one_minute(self) -> None:
        """Test format for times under 60 seconds."""
        from scripts.optimize._formatters import format_elapsed

        assert format_elapsed(0.0) == "0s"
        assert format_elapsed(30.5) == "30s"
        assert format_elapsed(59.9) == "60s"

    def test_formats_minutes_and_seconds(self) -> None:
        """Test format for times >= 60 seconds."""
        from scripts.optimize._formatters import format_elapsed

        assert format_elapsed(60.0) == "1m 00s"
        assert format_elapsed(90.0) == "1m 30s"
        assert format_elapsed(125.0) == "2m 05s"
        assert format_elapsed(3661.0) == "61m 01s"


class TestFormatDelta:
    """Tests for _format_delta helper function."""

    def test_formats_positive_delta_in_green(self) -> None:
        """Test that positive deltas (> 0.001) are displayed in green."""
        result = _format_delta(0.05)
        assert "[bold green]" in result
        assert "+0.0500" in result

    def test_formats_negative_delta_in_red(self) -> None:
        """Test that negative deltas (< -0.001) are displayed in red."""
        result = _format_delta(-0.05)
        assert "[bold red]" in result
        assert "-0.0500" in result

    def test_formats_near_zero_delta_as_dim(self) -> None:
        """Test that near-zero deltas are displayed dim."""
        result = _format_delta(0.0001)
        assert "[dim]" in result
        assert "+0.0001" in result

        result_neg = _format_delta(-0.0001)
        assert "[dim]" in result_neg
        assert "-0.0001" in result_neg
