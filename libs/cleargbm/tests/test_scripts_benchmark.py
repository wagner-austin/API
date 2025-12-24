"""Tests for scripts.benchmark module."""

from __future__ import annotations

import runpy
import sys

import pytest
from scripts.benchmark import (
    BenchmarkResult,
    _write,
    format_table,
    generate_synthetic_data,
    main,
    make_config,
    run_benchmark,
    run_benchmark_suite,
)


class TestWrite:
    """Tests for _write helper."""

    def test_writes_to_stdout(self) -> None:
        """Should write message to stdout without raising."""
        # Verify the function executes without error
        _write("test message")
        # Function executed successfully


class TestGenerateSyntheticData:
    """Tests for generate_synthetic_data function."""

    def test_generates_correct_size(self) -> None:
        """Should generate correct number of samples and features."""
        x, y = generate_synthetic_data(n_samples=100, n_features=5, seed=42)

        assert len(x) == 100
        assert len(y) == 100
        assert len(x[0]) == 5

    def test_generates_binary_labels(self) -> None:
        """Should generate only 0 and 1 labels."""
        _, y = generate_synthetic_data(n_samples=100, n_features=5, seed=42)

        assert all(label in (0, 1) for label in y)

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce same data."""
        x1, y1 = generate_synthetic_data(n_samples=50, n_features=3, seed=123)
        x2, y2 = generate_synthetic_data(n_samples=50, n_features=3, seed=123)

        assert x1 == x2
        assert y1 == y2

    def test_different_seeds_produce_different_data(self) -> None:
        """Different seeds should produce different data."""
        x1, _ = generate_synthetic_data(n_samples=50, n_features=3, seed=1)
        x2, _ = generate_synthetic_data(n_samples=50, n_features=3, seed=2)

        assert x1 != x2


class TestMakeConfig:
    """Tests for make_config function."""

    def test_creates_config_with_defaults(self) -> None:
        """Should create config with default values."""
        config = make_config()

        assert config["n_estimators"] == 10
        assert config["max_depth"] == 4
        assert config["max_bins"] == 64
        assert config["n_jobs"] == 1

    def test_creates_config_with_custom_values(self) -> None:
        """Should create config with custom values."""
        config = make_config(
            n_estimators=20,
            max_depth=6,
            max_bins=128,
            n_jobs=2,
        )

        assert config["n_estimators"] == 20
        assert config["max_depth"] == 6
        assert config["max_bins"] == 128
        assert config["n_jobs"] == 2


class TestBenchmarkResult:
    """Tests for BenchmarkResult namedtuple."""

    def test_creates_result(self) -> None:
        """Should create benchmark result with all fields."""
        result = BenchmarkResult(
            name="test",
            n_samples=1000,
            n_features=10,
            n_estimators=5,
            max_bins=64,
            n_jobs=1,
            elapsed_seconds=1.5,
            trees_per_second=3.33,
        )

        assert result.name == "test"
        assert result.n_samples == 1000
        assert result.elapsed_seconds == 1.5


class TestRunBenchmark:
    """Tests for run_benchmark function."""

    def test_runs_benchmark(self) -> None:
        """Should run benchmark and return result."""
        x, y = generate_synthetic_data(n_samples=100, n_features=3, seed=42)
        feature_names = ("f0", "f1", "f2")
        config = make_config(n_estimators=2, max_depth=2)

        result = run_benchmark(x, y, feature_names, config, "test_run")

        assert result.name == "test_run"
        assert result.n_samples == 100
        assert result.n_features == 3
        assert result.n_estimators == 2
        assert result.elapsed_seconds > 0
        assert result.trees_per_second > 0


class TestFormatTable:
    """Tests for format_table function."""

    def test_formats_single_result(self) -> None:
        """Should format single result as table."""
        results = [
            BenchmarkResult(
                name="test",
                n_samples=1000,
                n_features=10,
                n_estimators=5,
                max_bins=64,
                n_jobs=1,
                elapsed_seconds=1.5,
                trees_per_second=3.3,
            )
        ]

        table = format_table(results)

        assert "test" in table
        assert "1000" in table
        assert "1.50s" in table

    def test_formats_multiple_results(self) -> None:
        """Should format multiple results as table."""
        results = [
            BenchmarkResult(
                name="fast",
                n_samples=100,
                n_features=5,
                n_estimators=2,
                max_bins=32,
                n_jobs=1,
                elapsed_seconds=0.5,
                trees_per_second=4.0,
            ),
            BenchmarkResult(
                name="slow",
                n_samples=100,
                n_features=5,
                n_estimators=2,
                max_bins=64,
                n_jobs=1,
                elapsed_seconds=1.0,
                trees_per_second=2.0,
            ),
        ]

        table = format_table(results)

        assert "fast" in table
        assert "slow" in table


class TestRunBenchmarkSuite:
    """Tests for run_benchmark_suite function."""

    def test_runs_suite_with_small_data(self) -> None:
        """Should run benchmark suite with small dataset."""
        results = run_benchmark_suite(
            n_samples=50,
            n_features=3,
            n_estimators=2,
            verbose=False,
        )

        # Should have 12 benchmark configurations
        assert len(results) == 12
        # All results should have valid timing
        assert all(r.elapsed_seconds > 0 for r in results)

    def test_runs_suite_verbose(self) -> None:
        """Should run benchmark suite with verbose output."""
        results = run_benchmark_suite(
            n_samples=50,
            n_features=3,
            n_estimators=2,
            verbose=True,
        )

        assert len(results) == 12


class TestMain:
    """Tests for main entry point."""

    def test_runs_with_small_args(self) -> None:
        """Should run main with small dataset args."""
        exit_code = main(["--samples", "50", "--features", "3", "--trees", "2", "--quiet"])

        assert exit_code == 0

    def test_runs_with_defaults(self) -> None:
        """Should run main with default args (quiet mode for speed)."""
        # Use very small values for speed
        exit_code = main(["--samples", "30", "--features", "2", "--trees", "1", "--quiet"])

        assert exit_code == 0


def test_benchmark_entrypoint_runs_as_main() -> None:
    """Test that benchmark.py runs correctly when executed as __main__."""
    # Ensure a clean module state to avoid runpy runtime warning
    if "scripts.benchmark" in sys.modules:
        del sys.modules["scripts.benchmark"]

    # Patch sys.argv to use small values for speed
    original_argv = sys.argv
    sys.argv = ["benchmark", "--samples", "30", "--features", "2", "--trees", "1", "--quiet"]

    try:
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("scripts.benchmark", run_name="__main__")
        code = exc.value.code if isinstance(exc.value.code, int) else 0
        assert code == 0
    finally:
        sys.argv = original_argv
