"""Tests for submit CLI argument parsing.

Tests the command-line interface for the submit pipeline.
Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.submit.__main__ import (
    _ArgState,
    _handle_flag_arg,
    _handle_value_arg,
    _parse_aggregation,
    _parse_backend,
    parse_args,
    print_help,
)

from .conftest import get_captured_console


class TestParseBackend:
    """Tests for _parse_backend function."""

    def test_parse_valid_backends(self) -> None:
        """Test parsing valid backend names."""
        assert _parse_backend("lightgbm") == "lightgbm"
        assert _parse_backend("xgboost") == "xgboost"
        assert _parse_backend("mlp") == "mlp"
        assert _parse_backend("lstm") == "lstm"

    def test_parse_invalid_backend_raises(self) -> None:
        """Test that invalid backend raises SystemExit."""
        with pytest.raises(SystemExit):
            _parse_backend("invalid")


class TestParseAggregation:
    """Tests for _parse_aggregation function."""

    def test_parse_valid_aggregations(self) -> None:
        """Test parsing valid aggregation names."""
        assert _parse_aggregation("last") == "last"
        assert _parse_aggregation("first") == "first"
        assert _parse_aggregation("mean") == "mean"
        assert _parse_aggregation("statistics") == "statistics"

    def test_parse_invalid_aggregation_raises(self) -> None:
        """Test that invalid aggregation raises SystemExit."""
        with pytest.raises(SystemExit):
            _parse_aggregation("invalid")


class TestPrintHelp:
    """Tests for print_help function."""

    def test_print_help_outputs_usage(self) -> None:
        """Test that print_help outputs usage information."""
        print_help()

        captured = get_captured_console()
        assert len(captured.messages) == 1
        assert "Usage:" in captured.messages[0]
        assert "--backend" in captured.messages[0]


class TestArgState:
    """Tests for _ArgState class."""

    def test_argstate_initialization(self, tmp_path: Path) -> None:
        """Test _ArgState initializes with correct defaults."""
        state = _ArgState(tmp_path)

        assert state.backend == "lightgbm"
        assert state.n_estimators == 1000
        assert state.learning_rate == 0.05
        assert state.include_rank_features is True

    def test_argstate_to_parsed_args(self, tmp_path: Path) -> None:
        """Test _ArgState.to_parsed_args returns correct TypedDict."""
        state = _ArgState(tmp_path)
        state.backend = "xgboost"
        state.n_estimators = 500

        result = state.to_parsed_args()

        assert result["backend"] == "xgboost"
        assert result["n_estimators"] == 500


class TestHandleValueArg:
    """Tests for _handle_value_arg function."""

    def test_handle_backend_arg(self, tmp_path: Path) -> None:
        """Test handling --backend argument."""
        state = _ArgState(tmp_path)
        result = _handle_value_arg(state, "--backend", "xgboost")

        assert result is True
        assert state.backend == "xgboost"

    def test_handle_n_estimators_arg(self, tmp_path: Path) -> None:
        """Test handling --n-estimators argument."""
        state = _ArgState(tmp_path)
        result = _handle_value_arg(state, "--n-estimators", "500")

        assert result is True
        assert state.n_estimators == 500

    def test_handle_learning_rate_arg(self, tmp_path: Path) -> None:
        """Test handling --learning-rate argument."""
        state = _ArgState(tmp_path)
        result = _handle_value_arg(state, "--learning-rate", "0.1")

        assert result is True
        assert state.learning_rate == 0.1

    def test_handle_unknown_arg_returns_false(self, tmp_path: Path) -> None:
        """Test that unknown arguments return False."""
        state = _ArgState(tmp_path)
        result = _handle_value_arg(state, "--unknown", "value")

        assert result is False


class TestHandleFlagArg:
    """Tests for _handle_flag_arg function."""

    def test_handle_no_rank_features(self, tmp_path: Path) -> None:
        """Test handling --no-rank-features flag."""
        state = _ArgState(tmp_path)
        result = _handle_flag_arg(state, "--no-rank-features")

        assert result is True
        assert state.include_rank_features is False

    def test_handle_no_diff_features(self, tmp_path: Path) -> None:
        """Test handling --no-diff-features flag."""
        state = _ArgState(tmp_path)
        result = _handle_flag_arg(state, "--no-diff-features")

        assert result is True
        assert state.include_diff_features is False

    def test_handle_help_raises_systemexit(self, tmp_path: Path) -> None:
        """Test that --help raises SystemExit."""
        state = _ArgState(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            _handle_flag_arg(state, "--help")
        assert exc_info.value.code == 0

    def test_handle_unknown_flag_returns_false(self, tmp_path: Path) -> None:
        """Test that unknown flags return False."""
        state = _ArgState(tmp_path)
        result = _handle_flag_arg(state, "--unknown")

        assert result is False


class TestParseArgs:
    """Tests for parse_args function."""

    def test_parse_args_defaults(self) -> None:
        """Test parsing with default values."""
        result = parse_args([])

        assert result["backend"] == "lightgbm"
        assert result["n_estimators"] == 1000
        assert result["learning_rate"] == 0.05
        assert result["num_leaves"] == 31
        assert result["max_depth"] == -1
        assert result["aggregation"] == "statistics"
        assert result["include_rank_features"] is True
        assert result["include_diff_features"] is True

    def test_parse_args_backend(self) -> None:
        """Test parsing --backend argument."""
        result = parse_args(["--backend", "xgboost"])
        assert result["backend"] == "xgboost"

        result = parse_args(["-b", "mlp"])
        assert result["backend"] == "mlp"

    def test_parse_args_n_estimators(self) -> None:
        """Test parsing --n-estimators argument."""
        result = parse_args(["--n-estimators", "500"])
        assert result["n_estimators"] == 500

        result = parse_args(["-n", "200"])
        assert result["n_estimators"] == 200

    def test_parse_args_learning_rate(self) -> None:
        """Test parsing --learning-rate argument."""
        result = parse_args(["--learning-rate", "0.1"])
        assert result["learning_rate"] == 0.1

        result = parse_args(["-l", "0.01"])
        assert result["learning_rate"] == 0.01

    def test_parse_args_num_leaves(self) -> None:
        """Test parsing --num-leaves argument."""
        result = parse_args(["--num-leaves", "64"])
        assert result["num_leaves"] == 64

    def test_parse_args_max_depth(self) -> None:
        """Test parsing --max-depth argument."""
        result = parse_args(["--max-depth", "10"])
        assert result["max_depth"] == 10

    def test_parse_args_aggregation(self) -> None:
        """Test parsing --aggregation argument."""
        result = parse_args(["--aggregation", "last"])
        assert result["aggregation"] == "last"

        result = parse_args(["-a", "mean"])
        assert result["aggregation"] == "mean"

    def test_parse_args_no_rank_features(self) -> None:
        """Test parsing --no-rank-features flag."""
        result = parse_args(["--no-rank-features"])
        assert result["include_rank_features"] is False

    def test_parse_args_no_diff_features(self) -> None:
        """Test parsing --no-diff-features flag."""
        result = parse_args(["--no-diff-features"])
        assert result["include_diff_features"] is False

    def test_parse_args_train_dir(self, tmp_path: Path) -> None:
        """Test parsing --train-dir argument."""
        result = parse_args(["--train-dir", str(tmp_path)])
        assert result["train_dir"] == tmp_path

    def test_parse_args_test_dir(self, tmp_path: Path) -> None:
        """Test parsing --test-dir argument."""
        result = parse_args(["--test-dir", str(tmp_path)])
        assert result["test_dir"] == tmp_path

    def test_parse_args_output(self, tmp_path: Path) -> None:
        """Test parsing --output argument."""
        output = tmp_path / "out.csv"
        result = parse_args(["--output", str(output)])
        assert result["output_path"] == output

        result = parse_args(["-o", str(output)])
        assert result["output_path"] == output

    def test_parse_args_help_raises_systemexit(self) -> None:
        """Test that --help raises SystemExit."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--help"])
        assert exc_info.value.code == 0

        with pytest.raises(SystemExit) as exc_info:
            parse_args(["-h"])
        assert exc_info.value.code == 0

    def test_parse_args_combined(self) -> None:
        """Test parsing multiple arguments together."""
        result = parse_args(
            [
                "-b",
                "xgboost",
                "-n",
                "500",
                "-l",
                "0.1",
                "--num-leaves",
                "64",
                "--max-depth",
                "8",
                "-a",
                "last",
                "--no-rank-features",
                "--no-diff-features",
            ]
        )

        assert result["backend"] == "xgboost"
        assert result["n_estimators"] == 500
        assert result["learning_rate"] == 0.1
        assert result["num_leaves"] == 64
        assert result["max_depth"] == 8
        assert result["aggregation"] == "last"
        assert result["include_rank_features"] is False
        assert result["include_diff_features"] is False

    def test_parse_args_ignores_unknown_args(self) -> None:
        """Test that unknown arguments are ignored."""
        result = parse_args(["--unknown-flag", "--another", "value"])

        # Defaults should still be applied
        assert result["backend"] == "lightgbm"
        assert result["n_estimators"] == 1000
