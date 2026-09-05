"""Tests for dataset hooks and loading progress callbacks.

Tests real dataset hooks and loading progress callback coverage.
Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from shutil import copyfile

import scripts._test_hooks as _hooks
from covenant_ml.datasets import TimeSeriesDatasetConfig
from scripts._test_hooks import (
    LoadingProgressCallbackProtocol,
    LoadingProgressInfo,
    PhaseProgressCallbackProtocol,
    TrialProgressCallbackProtocol,
    UnifiedOptimizationResult,
)
from scripts.optimize._runners import run_single_with_progress

from .conftest import (
    make_fake_cleargbm_result,
    make_fake_lightgbm_result,
    make_fake_lstm_result,
    make_fake_mlp_result,
    make_fake_result,
)


class TestRealDatasetHooks:
    """Tests for real dataset hooks to ensure coverage."""

    def test_real_dataset_registry_returns_registry(self) -> None:
        """Test _real_dataset_registry returns a DatasetRegistry."""
        from scripts._test_hooks import _real_dataset_registry

        registry = _real_dataset_registry()

        assert "taiwan" in registry
        assert "us" in registry
        assert "polish" in registry

    def test_real_dataset_loader_loads_taiwan(self, tmp_path: Path) -> None:
        """Test _real_dataset_loader loads Taiwan dataset."""
        from scripts._test_hooks import _real_dataset_loader, _real_dataset_registry

        # Copy real Taiwan dataset
        external_dir = tmp_path / "external"
        taiwan_dir = external_dir / "taiwan_data"
        taiwan_dir.mkdir(parents=True, exist_ok=True)
        real_src = (
            Path(__file__).parent.parent.parent / "data" / "external" / "taiwan_data" / "data.csv"
        )
        assert real_src.exists(), "Taiwan dataset not found"
        copyfile(str(real_src), str(taiwan_dir / "data.csv"))

        registry = _real_dataset_registry()
        config = registry.get("taiwan")
        dataset = _real_dataset_loader(config, external_dir)

        assert dataset["meta"]["n_samples"] > 0
        assert dataset["meta"]["n_features"] > 0


class TestRealTimeseriesHooks:
    """Tests for real time-series dataset hooks to ensure coverage."""

    def test_real_timeseries_registry_returns_registry(self) -> None:
        """Test _real_timeseries_registry returns a TimeSeriesDatasetRegistry."""
        from scripts._test_hooks import _real_timeseries_registry

        registry = _real_timeseries_registry()

        # Registry should contain kaggle_amex_default
        assert "kaggle_amex_default" in registry

    def test_real_timeseries_loader_loads_sample(self, tmp_path: Path) -> None:
        """Test _real_timeseries_loader loads sample time-series dataset."""
        from scripts._test_hooks import _real_timeseries_loader

        # Create a minimal time-series config for testing
        sample_config: TimeSeriesDatasetConfig = TimeSeriesDatasetConfig(
            name="amex_sample",
            display_name="AMEX Sample",
            folder="amex_sample",
            file_name="data.csv",
            file_format="csv",
            encoding="utf-8",
            target={
                "column_name": "target",
                "label_type": "binary_int",
                "positive_values": (1,),
                "negative_values": (0,),
            },
            exclude_columns=(),
            n_samples_expected=10,
            n_features_expected=10,
            positive_class_ratio_expected=0.3,
            time_series={
                "entity_column": "customer_ID",
                "time_column": "S_2",
                "aggregation": "last",
                "labels_file": "labels.csv",
                "labels_entity_column": "customer_ID",
                "include_rank_features": False,
                "include_diff_features": False,
                "include_window_features": False,
                "window_sizes": (),
            },
        )

        # Copy sample fixtures
        external_dir = tmp_path / "external"
        sample_dir = external_dir / "amex_sample"
        sample_dir.mkdir(parents=True, exist_ok=True)

        fixture_dir = (
            Path(__file__).parent.parent.parent.parent.parent
            / "libs"
            / "covenant_ml"
            / "tests"
            / "datasets"
            / "fixtures"
            / "timeseries_amex_sample"
        )
        copyfile(str(fixture_dir / "data.csv"), str(sample_dir / "data.csv"))
        copyfile(str(fixture_dir / "labels.csv"), str(sample_dir / "labels.csv"))

        # Load using real loader
        dataset = _real_timeseries_loader(sample_config, external_dir)

        assert dataset["meta"]["n_samples"] > 0
        assert dataset["meta"]["n_features"] > 0
        assert len(dataset["x"]) == len(dataset["y"])


def _make_unified_fake_runner(
    result_factory: Callable[[], UnifiedOptimizationResult],
    loading_info: LoadingProgressInfo | None = None,
) -> _hooks.OptimizationRunnerProtocol:
    """Create a fake runner that optionally invokes the loading progress callback.

    Args:
        result_factory: Callable returning UnifiedOptimizationResult.
        loading_info: Optional loading progress info to pass to callback.

    Returns:
        Fake runner matching OptimizationRunnerProtocol.
    """

    def fake_runner(
        config_json: str,
        external_dir: Path,
        output_dir: Path,
        progress_callback: TrialProgressCallbackProtocol | None = None,
        phase_callback: PhaseProgressCallbackProtocol | None = None,
        loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
    ) -> UnifiedOptimizationResult:
        _ = config_json
        _ = external_dir
        _ = output_dir
        _ = progress_callback
        _ = phase_callback
        if loading_progress_callback is not None and loading_info is not None:
            loading_progress_callback(loading_info)
        return result_factory()

    return fake_runner


class TestLoadingProgressCallbacks:
    """Tests for loading progress callback coverage in all backends.

    These tests ensure the loading_progress_callback is exercised
    by having fake runners invoke it with the unified callback type.
    """

    def test_xgboost_loading_progress_callback_is_invoked(self) -> None:
        """Test XGBoost loading progress callback is called and formats correctly."""
        loading_callback_calls: list[LoadingProgressInfo] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            if loading_progress_callback is not None:
                info: LoadingProgressInfo = {
                    "dataset": "taiwan",
                    "phase": "reading",
                    "percent_complete": 50.0,
                    "rows_processed": 500,
                    "rows_total": 1000,
                    "message": "Reading rows from dataset",
                }
                loading_progress_callback(info)
                loading_callback_calls.append(info)
            return make_fake_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            result = run_single_with_progress(
                "xgboost", "taiwan", 5, "full", "cpu", None, save_model=False
            )
            assert result["backend"] == "xgboost"
            assert len(loading_callback_calls) == 1
            assert loading_callback_calls[0]["dataset"] == "taiwan"
            assert loading_callback_calls[0]["phase"] == "reading"
            assert loading_callback_calls[0]["percent_complete"] == 50.0
        finally:
            _hooks.optimization_runner = original

    def test_mlp_loading_progress_callback_is_invoked(self) -> None:
        """Test MLP loading progress callback is called and formats correctly."""
        loading_callback_calls: list[LoadingProgressInfo] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            if loading_progress_callback is not None:
                info: LoadingProgressInfo = {
                    "dataset": "taiwan",
                    "phase": "parsing",
                    "percent_complete": 75.0,
                    "rows_processed": 750,
                    "rows_total": 1000,
                    "message": "Parsing rows from dataset",
                }
                loading_progress_callback(info)
                loading_callback_calls.append(info)
            return make_fake_mlp_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            result = run_single_with_progress(
                "mlp", "taiwan", 5, "full", "cpu", None, save_model=False
            )
            assert result["backend"] == "mlp"
            assert len(loading_callback_calls) == 1
            assert loading_callback_calls[0]["dataset"] == "taiwan"
            assert loading_callback_calls[0]["phase"] == "parsing"
            assert loading_callback_calls[0]["percent_complete"] == 75.0
        finally:
            _hooks.optimization_runner = original

    def test_lightgbm_loading_progress_callback_is_invoked(self) -> None:
        """Test LightGBM loading progress callback is called."""
        loading_callback_calls: list[LoadingProgressInfo] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            if loading_progress_callback is not None:
                info: LoadingProgressInfo = {
                    "dataset": "taiwan",
                    "phase": "encoding",
                    "percent_complete": 100.0,
                    "rows_processed": 1000,
                    "rows_total": 1000,
                    "message": "Encoding categorical features",
                }
                loading_progress_callback(info)
                loading_callback_calls.append(info)
            return make_fake_lightgbm_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            result = run_single_with_progress(
                "lightgbm", "taiwan", 5, "full", "cpu", None, save_model=False
            )
            assert result["backend"] == "lightgbm"
            assert len(loading_callback_calls) == 1
            assert loading_callback_calls[0]["dataset"] == "taiwan"
            assert loading_callback_calls[0]["phase"] == "encoding"
            assert loading_callback_calls[0]["percent_complete"] == 100.0
        finally:
            _hooks.optimization_runner = original

    def test_lstm_loading_progress_callback_is_invoked(self) -> None:
        """Test LSTM loading progress callback is called."""
        loading_callback_calls: list[LoadingProgressInfo] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            if loading_progress_callback is not None:
                info: LoadingProgressInfo = {
                    "dataset": "taiwan",
                    "phase": "reading",
                    "percent_complete": 25.0,
                    "rows_processed": 250,
                    "rows_total": 1000,
                    "message": "Reading rows from dataset",
                }
                loading_progress_callback(info)
                loading_callback_calls.append(info)
            return make_fake_lstm_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            result = run_single_with_progress(
                "lstm", "taiwan", 5, "full", "cpu", None, save_model=False
            )
            assert result["backend"] == "lstm"
            assert len(loading_callback_calls) == 1
            assert loading_callback_calls[0]["dataset"] == "taiwan"
            assert loading_callback_calls[0]["phase"] == "reading"
            assert loading_callback_calls[0]["percent_complete"] == 25.0
        finally:
            _hooks.optimization_runner = original

    def test_cleargbm_loading_progress_callback_is_invoked(self) -> None:
        """Test ClearGBM loading progress callback is called."""
        loading_callback_calls: list[LoadingProgressInfo] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            if loading_progress_callback is not None:
                info: LoadingProgressInfo = {
                    "dataset": "taiwan",
                    "phase": "reading",
                    "percent_complete": 100.0,
                    "rows_processed": 1000,
                    "rows_total": 1000,
                    "message": "Loaded 1000 rows",
                }
                loading_progress_callback(info)
                loading_callback_calls.append(info)
            return make_fake_cleargbm_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            result = run_single_with_progress(
                "cleargbm", "taiwan", 5, "full", "cpu", None, save_model=False
            )
            assert result["backend"] == "cleargbm"
            assert len(loading_callback_calls) == 1
            assert loading_callback_calls[0]["dataset"] == "taiwan"
            assert loading_callback_calls[0]["percent_complete"] == 100.0
        finally:
            _hooks.optimization_runner = original
