"""Tests for scripts.autotune module."""

from __future__ import annotations

import runpy
import sys

import numpy as np
import pytest
from scripts.autotune import (
    _write,
    format_report,
    generate_sample_data,
    main,
    make_config,
    run_autotune,
    time_config,
)


class TestWrite:
    """Tests for _write helper."""

    def test_writes_to_stdout(self) -> None:
        """Should write message to stdout without raising."""
        _write("test message")


class TestGenerateSampleData:
    """Tests for generate_sample_data function."""

    def test_generates_correct_size(self) -> None:
        """Should generate correct number of samples and features."""
        x, y = generate_sample_data(n_samples=100, n_features=5, seed=42)

        n_samples: int = x.shape[0]
        n_features: int = x.shape[1]
        n_labels: int = y.shape[0]
        assert n_samples == 100
        assert n_labels == 100
        assert n_features == 5

    def test_generates_binary_labels(self) -> None:
        """Should generate only 0 and 1 labels."""
        _, y = generate_sample_data(n_samples=100, n_features=5, seed=42)

        n_labels: int = y.shape[0]
        for i in range(n_labels):
            label: int = y.item(i)
            assert label in (0, 1)

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce same data."""
        x1, y1 = generate_sample_data(n_samples=50, n_features=3, seed=123)
        x2, y2 = generate_sample_data(n_samples=50, n_features=3, seed=123)

        assert np.array_equal(x1, x2)
        assert np.array_equal(y1, y2)

    def test_different_seeds_produce_different_data(self) -> None:
        """Different seeds should produce different data."""
        x1, _ = generate_sample_data(n_samples=50, n_features=3, seed=1)
        x2, _ = generate_sample_data(n_samples=50, n_features=3, seed=2)

        assert not np.array_equal(x1, x2)


class TestMakeConfig:
    """Tests for make_config function."""

    def test_creates_config_with_defaults(self) -> None:
        """Should create config with default values."""
        config = make_config()

        assert config["n_estimators"] == 5
        assert config["max_depth"] == 4
        assert config["max_bins"] == 64
        assert config["n_jobs"] == 1

    def test_creates_config_with_custom_values(self) -> None:
        """Should create config with custom values."""
        config = make_config(
            n_estimators=10,
            max_depth=6,
            max_bins=128,
            n_jobs=2,
        )

        assert config["n_estimators"] == 10
        assert config["max_depth"] == 6
        assert config["max_bins"] == 128
        assert config["n_jobs"] == 2


class TestTimeConfig:
    """Tests for time_config function."""

    def test_times_config(self) -> None:
        """Should time a configuration and return TimingResult."""
        x, y = generate_sample_data(n_samples=50, n_features=3, seed=42)
        feature_names = ("f0", "f1", "f2")
        config = make_config(n_estimators=2, max_depth=2)

        result = time_config(x, y, feature_names, config)

        assert result["n_jobs"] == 1
        assert result["max_bins"] == 64
        assert result["elapsed_seconds"] > 0
        assert result["trees_per_second"] > 0


class TestRunAutotune:
    """Tests for run_autotune function."""

    def test_runs_autotune_small_grid(self) -> None:
        """Should run autotune with small grid."""
        x, y = generate_sample_data(n_samples=50, n_features=3, seed=42)

        report = run_autotune(
            x=x,
            y=y,
            n_estimators=2,
            n_jobs_grid=(1,),
            max_bins_grid=(32,),
            verbose=False,
        )

        assert report["sample_size"] == 50
        assert report["n_features"] == 3
        assert len(report["timing_results"]) == 1
        assert report["recommended_n_jobs"] == 1
        assert report["recommended_max_bins"] == 32

    def test_runs_autotune_with_multiple_configs(self) -> None:
        """Should run autotune with multiple configurations."""
        x, y = generate_sample_data(n_samples=50, n_features=3, seed=42)

        report = run_autotune(
            x=x,
            y=y,
            n_estimators=2,
            n_jobs_grid=(1, 2),
            max_bins_grid=(32, 64),
            verbose=False,
        )

        # Should have 2 * 2 = 4 timing results
        assert len(report["timing_results"]) == 4
        assert report["parallel_speedup"] >= 0

    def test_runs_autotune_verbose(self) -> None:
        """Should run autotune with verbose output."""
        x, y = generate_sample_data(n_samples=50, n_features=3, seed=42)

        report = run_autotune(
            x=x,
            y=y,
            n_estimators=2,
            n_jobs_grid=(1,),
            max_bins_grid=(32,),
            verbose=True,
        )

        assert report["sample_size"] == 50

    def test_raises_on_empty_grid(self) -> None:
        """Should raise ValueError when grid is empty."""
        x, y = generate_sample_data(n_samples=50, n_features=3, seed=42)

        with pytest.raises(ValueError, match="No timing results collected"):
            run_autotune(
                x=x,
                y=y,
                n_estimators=2,
                n_jobs_grid=(),
                max_bins_grid=(),
                verbose=False,
            )


class TestFormatReport:
    """Tests for format_report function."""

    def test_formats_report(self) -> None:
        """Should format TuningReport as readable text."""
        x, y = generate_sample_data(n_samples=50, n_features=3, seed=42)

        report = run_autotune(
            x=x,
            y=y,
            n_estimators=2,
            n_jobs_grid=(1,),
            max_bins_grid=(32,),
            verbose=False,
        )

        formatted = format_report(report)

        assert "Autotune Report" in formatted
        assert "50 samples" in formatted
        assert "n_jobs" in formatted
        assert "max_bins" in formatted


class TestMain:
    """Tests for main entry point."""

    def test_runs_with_small_args(self) -> None:
        """Should run main with small dataset args."""
        exit_code = main(["--samples", "30", "--features", "2", "--trees", "1", "--quiet"])

        assert exit_code == 0

    def test_runs_with_defaults_quiet(self) -> None:
        """Should run main with defaults in quiet mode."""
        exit_code = main(["--samples", "30", "--features", "2", "--trees", "1", "--quiet"])

        assert exit_code == 0

    def test_runs_with_verbose_mode(self) -> None:
        """Should run main with verbose output (no --quiet flag)."""
        exit_code = main(["--samples", "30", "--features", "2", "--trees", "1"])

        assert exit_code == 0


def test_autotune_entrypoint_runs_as_main() -> None:
    """Test that autotune.py runs correctly when executed as __main__."""
    # Ensure a clean module state
    if "scripts.autotune" in sys.modules:
        del sys.modules["scripts.autotune"]

    # Patch sys.argv to use small values for speed
    original_argv = sys.argv
    sys.argv = ["autotune", "--samples", "30", "--features", "2", "--trees", "1", "--quiet"]

    try:
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("scripts.autotune", run_name="__main__")
        assert exc.value.code == 0
    finally:
        sys.argv = original_argv
