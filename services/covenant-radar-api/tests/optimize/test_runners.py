"""Tests for scripts/optimize/_runners.py - unified runner with progress.

Tests run_single_with_progress function that handles optimization with
progress bar integration and history tracking.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

import scripts._test_hooks as _hooks
from covenant_ml.backends.registry import BackendRegistration, ClassifierRegistry
from covenant_ml.datasets import DatasetRegistry
from covenant_ml.types import BackendName
from scripts._test_hooks import (
    LoadingProgressCallbackProtocol,
    LoadingProgressInfo,
    PhaseProgressCallbackProtocol,
    PhaseProgressInfo,
    TrialProgressCallbackProtocol,
    TrialProgressInfo,
    UnifiedOptimizationResult,
)
from scripts.optimize._runners import run_single_with_progress

from .conftest import (
    FakeSaveModelBackend,
    make_fake_cleargbm_result,
    make_fake_dataset_config,
    make_fake_lightgbm_result,
    make_fake_loaded_dataset,
    make_fake_lstm_result,
    make_fake_mlp_result,
    make_fake_result,
)


def _make_phase_infos(
    backend: BackendName,
    dataset: str = "taiwan",
) -> list[PhaseProgressInfo]:
    """Create the four standard phase progress infos.

    Args:
        backend: Backend name.
        dataset: Dataset name.

    Returns:
        List of PhaseProgressInfo for all four phases.
    """
    return [
        {
            "phase": "loading_data",
            "backend": backend,
            "dataset": dataset,
            "n_samples": 0,
            "n_features": 0,
        },
        {
            "phase": "feature_engineering",
            "backend": backend,
            "dataset": dataset,
            "n_samples": 1000,
            "n_features": 100,
        },
        {
            "phase": "optimizing",
            "backend": backend,
            "dataset": dataset,
            "n_samples": 1000,
            "n_features": 150,
        },
        {
            "phase": "saving",
            "backend": backend,
            "dataset": dataset,
            "n_samples": 1000,
            "n_features": 150,
        },
    ]


def _make_trial_info(backend: BackendName) -> TrialProgressInfo:
    """Create a standard trial progress info.

    Args:
        backend: Backend name.

    Returns:
        TrialProgressInfo for a single trial.
    """
    return {
        "backend": backend,
        "trial_number": 1,
        "n_trials_total": 5,
        "current_value": 0.80,
        "best_value": 0.80,
        "best_trial": 1,
        "is_best": True,
    }


class TestRunSingleWithProgress:
    """Tests for run_single_with_progress function."""

    def test_runs_xgboost_backend(self) -> None:
        """Test run_single_with_progress uses XGBoost backend correctly."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            nonlocal call_count
            call_count += 1
            if phase_callback is not None:
                for info in _make_phase_infos("xgboost"):
                    phase_callback(info)
            if progress_callback is not None:
                progress_callback(_make_trial_info("xgboost"))
            return make_fake_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            result = run_single_with_progress(
                "xgboost", "taiwan", 5, "full", "cpu", None, save_model=False
            )
            assert call_count == 1
            assert result["backend"] == "xgboost"
            assert result["result"]["backend"] == "xgboost"
        finally:
            _hooks.optimization_runner = original

    def test_runs_mlp_backend(self) -> None:
        """Test run_single_with_progress uses MLP backend correctly."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            nonlocal call_count
            call_count += 1
            if phase_callback is not None:
                for info in _make_phase_infos("mlp"):
                    phase_callback(info)
            if progress_callback is not None:
                progress_callback(_make_trial_info("mlp"))
            return make_fake_mlp_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            result = run_single_with_progress(
                "mlp", "taiwan", 5, "full", "cpu", None, save_model=False
            )
            assert call_count == 1
            assert result["backend"] == "mlp"
            assert result["result"]["backend"] == "mlp"
        finally:
            _hooks.optimization_runner = original

    def test_runs_lightgbm_backend(self) -> None:
        """Test run_single_with_progress uses LightGBM backend correctly."""
        call_count = 0
        phase_calls: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            nonlocal call_count
            call_count += 1
            if phase_callback is not None:
                for info in _make_phase_infos("lightgbm"):
                    phase_callback(info)
                phase_calls.append("called")
            if progress_callback is not None:
                progress_callback(_make_trial_info("lightgbm"))
            return make_fake_lightgbm_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            result = run_single_with_progress(
                "lightgbm", "taiwan", 5, "full", "cpu", None, save_model=False
            )
            assert call_count == 1
            assert result["backend"] == "lightgbm"
            assert result["result"]["backend"] == "lightgbm"
            assert len(phase_calls) == 1
        finally:
            _hooks.optimization_runner = original

    def test_runs_lstm_backend(self) -> None:
        """Test run_single_with_progress uses LSTM backend correctly."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            nonlocal call_count
            call_count += 1
            if phase_callback is not None:
                for info in _make_phase_infos("lstm"):
                    phase_callback(info)
            if progress_callback is not None:
                progress_callback(_make_trial_info("lstm"))
            return make_fake_lstm_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            result = run_single_with_progress(
                "lstm", "taiwan", 5, "full", "cpu", None, save_model=False
            )
            assert call_count == 1
            assert result["backend"] == "lstm"
            assert result["result"]["backend"] == "lstm"
        finally:
            _hooks.optimization_runner = original

    def test_runs_cleargbm_backend(self) -> None:
        """Test run_single_with_progress uses ClearGBM backend correctly."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            nonlocal call_count
            call_count += 1
            if phase_callback is not None:
                for info in _make_phase_infos("cleargbm"):
                    phase_callback(info)
            if progress_callback is not None:
                progress_callback(_make_trial_info("cleargbm"))
            if loading_progress_callback is not None:
                loading_info: LoadingProgressInfo = {
                    "dataset": "taiwan",
                    "phase": "reading",
                    "percent_complete": 100.0,
                    "rows_processed": 1000,
                    "rows_total": 1000,
                    "message": "Loaded 1000 rows",
                }
                loading_progress_callback(loading_info)
            return make_fake_cleargbm_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            result = run_single_with_progress(
                "cleargbm", "taiwan", 5, "full", "cpu", None, save_model=False
            )
            assert call_count == 1
            assert result["backend"] == "cleargbm"
            assert result["result"]["backend"] == "cleargbm"
        finally:
            _hooks.optimization_runner = original

    def test_run_single_with_save_model_true(self, tmp_path: Path) -> None:
        """Test run_single_with_progress with save_model=True.

        This covers the save_model branch in run_single_with_progress.
        """
        fake_backend = FakeSaveModelBackend()
        fake_registry = ClassifierRegistry()
        fake_registry.register("xgboost", BackendRegistration(lambda: fake_backend))
        fake_dataset_reg = DatasetRegistry((make_fake_dataset_config("taiwan"),))

        orig_runner = _hooks.optimization_runner
        orig_backend_reg = _hooks.backend_registry_factory
        orig_dataset_reg = _hooks.dataset_registry_factory
        orig_loader = _hooks.dataset_loader

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            _ = progress_callback
            _ = phase_callback
            return make_fake_result()

        _hooks.optimization_runner = fake_runner
        _hooks.backend_registry_factory = lambda: fake_registry
        _hooks.dataset_registry_factory = lambda: fake_dataset_reg
        _hooks.dataset_loader = lambda cfg, ext_dir: make_fake_loaded_dataset()

        try:
            (tmp_path / "data" / "external").mkdir(parents=True, exist_ok=True)
            (tmp_path / "models").mkdir(parents=True, exist_ok=True)

            result = run_single_with_progress(
                "xgboost", "taiwan", 5, "full", "cpu", None, save_model=True, project_root=tmp_path
            )

            assert result["backend"] == "xgboost"
            assert result["result"]["backend"] == "xgboost"

            model_dir = tmp_path / "models" / "xgboost"
            assert model_dir.exists()
        finally:
            _hooks.optimization_runner = orig_runner
            _hooks.backend_registry_factory = orig_backend_reg
            _hooks.dataset_registry_factory = orig_dataset_reg
            _hooks.dataset_loader = orig_loader
