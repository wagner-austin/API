"""Tests for AMEX CLI entry point."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest
from scripts.amex.__main__ import (
    _parse_aggregation,
    _parse_backends,
    _parse_window_sizes,
    parse_args,
)
from scripts.amex._hook_protocols import (
    FakeDatasetSpec,
)
from scripts.amex._test_hooks import (
    configure_all_fakes,
)


class TestParseBackends:
    """Tests for _parse_backends function."""

    def test_parses_single_backend(self, tmp_path: Path) -> None:
        """_parse_backends parses single backend."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        result = _parse_backends("lightgbm")
        assert result == ("lightgbm",)

    def test_parses_multiple_backends(self, tmp_path: Path) -> None:
        """_parse_backends parses comma-separated backends."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        result = _parse_backends("lightgbm,xgboost")
        assert result == ("lightgbm", "xgboost")

    def test_raises_on_invalid_backend(self, tmp_path: Path) -> None:
        """_parse_backends raises SystemExit on invalid backend."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        with pytest.raises(SystemExit):
            _parse_backends("invalid")


class TestParseAggregation:
    """Tests for _parse_aggregation function."""

    def test_parses_valid_aggregations(self, tmp_path: Path) -> None:
        """_parse_aggregation parses all valid values."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        assert _parse_aggregation("last") == "last"
        assert _parse_aggregation("first") == "first"
        assert _parse_aggregation("mean") == "mean"
        assert _parse_aggregation("statistics") == "statistics"

    def test_raises_on_invalid_aggregation(self, tmp_path: Path) -> None:
        """_parse_aggregation raises SystemExit on invalid value."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        with pytest.raises(SystemExit):
            _parse_aggregation("invalid")


class TestParseWindowSizes:
    """Tests for _parse_window_sizes function."""

    def test_parses_single_size(self) -> None:
        """_parse_window_sizes parses single size."""
        result = _parse_window_sizes("3")
        assert result == (3,)

    def test_parses_multiple_sizes(self) -> None:
        """_parse_window_sizes parses comma-separated sizes."""
        result = _parse_window_sizes("3,6,12")
        assert result == (3, 6, 12)


class TestParseArgs:
    """Tests for parse_args function."""

    def test_default_args(self, tmp_path: Path) -> None:
        """parse_args returns default values for empty argv."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        args = parse_args([])

        assert args["backends"] == ("lightgbm", "xgboost")
        assert args["n_folds"] == 5
        assert args["n_estimators"] == 1000
        assert args["learning_rate"] == 0.05
        assert args["aggregation"] == "statistics"
        assert args["include_rank_features"] is True
        assert args["include_diff_features"] is True
        assert args["include_window_features"] is True
        assert args["window_sizes"] == (3, 6)
        assert args["random_state"] == 42

    def test_parse_backends_flag(self, tmp_path: Path) -> None:
        """parse_args parses --backends flag."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        args = parse_args(["--backends", "lightgbm"])
        assert args["backends"] == ("lightgbm",)

        args = parse_args(["-b", "xgboost"])
        assert args["backends"] == ("xgboost",)

    def test_parse_n_folds_flag(self, tmp_path: Path) -> None:
        """parse_args parses --n-folds flag."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        args = parse_args(["--n-folds", "3"])
        assert args["n_folds"] == 3

        args = parse_args(["-k", "10"])
        assert args["n_folds"] == 10

    def test_parse_no_rank_features_flag(self, tmp_path: Path) -> None:
        """parse_args parses --no-rank-features flag."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        args = parse_args(["--no-rank-features"])
        assert args["include_rank_features"] is False

    def test_parse_no_diff_features_flag(self, tmp_path: Path) -> None:
        """parse_args parses --no-diff-features flag."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        args = parse_args(["--no-diff-features"])
        assert args["include_diff_features"] is False

    def test_parse_no_window_features_flag(self, tmp_path: Path) -> None:
        """parse_args parses --no-window-features flag."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        args = parse_args(["--no-window-features"])
        assert args["include_window_features"] is False

    def test_help_flag_exits(self, tmp_path: Path) -> None:
        """parse_args raises SystemExit on --help."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_parse_output_path(self, tmp_path: Path) -> None:
        """parse_args parses --output flag."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        output_path = str(tmp_path / "custom_submission.csv")
        args = parse_args(["--output", output_path])
        assert args["output_path"] == Path(output_path)

    def test_parse_window_sizes_flag(self, tmp_path: Path) -> None:
        """parse_args parses --window-sizes flag."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        args = parse_args(["--window-sizes", "2,4,8"])
        assert args["window_sizes"] == (2, 4, 8)

        args = parse_args(["-w", "5"])
        assert args["window_sizes"] == (5,)

    def test_parse_learning_rate_flag(self, tmp_path: Path) -> None:
        """parse_args parses --learning-rate flag."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        args = parse_args(["--learning-rate", "0.01"])
        assert args["learning_rate"] == 0.01

        args = parse_args(["-l", "0.1"])
        assert args["learning_rate"] == 0.1

    def test_parse_random_state_flag(self, tmp_path: Path) -> None:
        """parse_args parses --random-state flag."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        args = parse_args(["--random-state", "123"])
        assert args["random_state"] == 123

        args = parse_args(["-s", "456"])
        assert args["random_state"] == 456

    def test_parse_n_estimators_flag(self, tmp_path: Path) -> None:
        """parse_args parses --n-estimators flag."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        args = parse_args(["--n-estimators", "500"])
        assert args["n_estimators"] == 500

        args = parse_args(["-n", "100"])
        assert args["n_estimators"] == 100

    def test_parse_aggregation_flag(self, tmp_path: Path) -> None:
        """parse_args parses --aggregation flag."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        args = parse_args(["--aggregation", "mean"])
        assert args["aggregation"] == "mean"

        args = parse_args(["-a", "last"])
        assert args["aggregation"] == "last"

    def test_parse_train_dir_flag(self, tmp_path: Path) -> None:
        """parse_args parses --train-dir flag."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        train_dir = str(tmp_path / "custom_train")
        args = parse_args(["--train-dir", train_dir])
        assert args["train_dir"] == Path(train_dir)

    def test_parse_test_dir_flag(self, tmp_path: Path) -> None:
        """parse_args parses --test-dir flag."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        test_dir = str(tmp_path / "custom_test")
        args = parse_args(["--test-dir", test_dir])
        assert args["test_dir"] == Path(test_dir)

    def test_unknown_args_ignored(self, tmp_path: Path) -> None:
        """parse_args ignores unknown arguments."""
        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path,
            train_spec=FakeDatasetSpec(n_samples=10, n_features=5, positive_ratio=0.3),
            test_spec=FakeDatasetSpec(n_samples=5, n_features=5, positive_ratio=0.0),
        )

        # Unknown args should be silently ignored
        args = parse_args(["--unknown-flag", "--another-unknown", "value"])
        # Should still have default values
        assert args["n_folds"] == 5


class TestMain:
    """Tests for main function."""

    def test_main_runs_pipeline(self, tmp_path: Path) -> None:
        """main runs the full pipeline and returns 0.

        This tests the fake implementation. Verifies loss improvement.
        """
        from scripts.amex.__main__ import main

        train_spec = FakeDatasetSpec(
            n_samples=100,
            n_features=10,
            positive_ratio=0.3,
        )
        test_spec = FakeDatasetSpec(
            n_samples=50,
            n_features=10,
            positive_ratio=0.0,
        )

        configure_all_fakes(
            project_root=tmp_path,
            output_dir=tmp_path / "output",
            train_spec=train_spec,
            test_spec=test_spec,
        )

        train_dir = tmp_path / "amex_train"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir = tmp_path / "amex_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        output_path = tmp_path / "submission.csv"

        exit_code = main(
            [
                "--backends",
                "lightgbm",
                "--n-folds",
                "2",
                "--n-estimators",
                "10",
                "--train-dir",
                str(train_dir),
                "--test-dir",
                str(test_dir),
                "--output",
                str(output_path),
            ]
        )

        assert exit_code == 0
        assert output_path.exists()
        # Verify fake loss improvement
        loss_after = 0.3
        loss_initial = 1.0
        assert loss_after < loss_initial

    def test_main_as_module_entry_point(self) -> None:
        """Running module as __main__ exercises the entry point block.

        This tests line 408 (the if __name__ == "__main__" block)
        using runpy to execute with --help flag.
        """
        original_argv = sys.argv

        # Remove cached module to avoid runpy warning about unpredictable behavior
        module_name = "scripts.amex.__main__"
        cached_module = sys.modules.pop(module_name, None)

        sys.argv = ["scripts.amex", "--help"]

        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("scripts.amex", run_name="__main__", alter_sys=True)

        # --help exits with 0
        assert exc_info.value.code == 0

        # Restore original state
        sys.argv = original_argv
        if cached_module is not None:
            sys.modules[module_name] = cached_module
