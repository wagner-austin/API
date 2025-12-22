"""Tests for scripts/optimize/_runners.py - backend-specific runners with progress.

Tests run_single_with_progress function that handles optimization with
progress bar integration and history tracking.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

import scripts._test_hooks as _hooks
from covenant_ml.backends.registry import BackendRegistration, ClassifierRegistry
from covenant_ml.datasets import DatasetRegistry
from scripts._test_hooks import (
    LightGBMOptimizationResult,
    LSTMOptimizationResult,
    MLPOptimizationResult,
    XGBoostOptimizationResult,
)
from scripts.optimize._runners import run_single_with_progress

from .conftest import (
    FakeSaveModelBackend,
    make_fake_dataset_config,
    make_fake_lightgbm_result,
    make_fake_loaded_dataset,
    make_fake_lstm_result,
    make_fake_mlp_result,
    make_fake_result,
)


class TestRunSingleWithProgress:
    """Tests for run_single_with_progress function."""

    def test_runs_xgboost_backend(self) -> None:
        """Test run_single_with_progress uses XGBoost backend correctly."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
            phase_callback: _hooks.XGBoostPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.XGBoostLoadingProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            nonlocal call_count
            call_count += 1
            # Exercise the phase callback if provided
            if phase_callback is not None:
                phase_callback(
                    {
                        "phase": "loading_data",
                        "dataset": "taiwan",
                        "n_samples": 0,
                        "n_features": 0,
                    }
                )
                phase_callback(
                    {
                        "phase": "feature_engineering",
                        "dataset": "taiwan",
                        "n_samples": 1000,
                        "n_features": 100,
                    }
                )
                phase_callback(
                    {
                        "phase": "optimizing",
                        "dataset": "taiwan",
                        "n_samples": 1000,
                        "n_features": 150,
                    }
                )
                phase_callback(
                    {
                        "phase": "saving",
                        "dataset": "taiwan",
                        "n_samples": 1000,
                        "n_features": 150,
                    }
                )
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial_number": 1,
                        "n_trials_total": 5,
                        "current_auc": 0.80,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": True,
                        "best_learning_rate": 0.1,
                        "best_max_depth": 6,
                        "best_n_estimators": 100,
                    }
                )
            return make_fake_result()

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            result = run_single_with_progress(
                "xgboost", "taiwan", 5, "full", "cpu", None, save_model=False
            )
            assert call_count == 1
            assert result["backend"] == "xgboost"
            assert result["result"]["backend"] == "xgboost"
        finally:
            _hooks.xgboost_runner = original

    def test_runs_mlp_backend(self) -> None:
        """Test run_single_with_progress uses MLP backend correctly."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.MLPTrialProgressCallbackProtocol | None = None,
            phase_callback: _hooks.MLPPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.MLPLoadingProgressCallbackProtocol | None = None,
        ) -> MLPOptimizationResult:
            nonlocal call_count
            call_count += 1
            # Exercise the phase callback if provided
            if phase_callback is not None:
                phase_callback(
                    {
                        "phase": "loading_data",
                        "dataset": "taiwan",
                        "n_samples": 0,
                        "n_features": 0,
                    }
                )
                phase_callback(
                    {
                        "phase": "feature_engineering",
                        "dataset": "taiwan",
                        "n_samples": 1000,
                        "n_features": 100,
                    }
                )
                phase_callback(
                    {
                        "phase": "optimizing",
                        "dataset": "taiwan",
                        "n_samples": 1000,
                        "n_features": 150,
                    }
                )
                phase_callback(
                    {
                        "phase": "saving",
                        "dataset": "taiwan",
                        "n_samples": 1000,
                        "n_features": 150,
                    }
                )
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial_number": 1,
                        "n_trials_total": 5,
                        "current_auc": 0.80,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": True,
                        "best_learning_rate": 0.001,
                        "best_n_layers": 3,
                        "best_hidden_size": 128,
                        "best_dropout": 0.2,
                    }
                )
            return make_fake_mlp_result()

        original = _hooks.mlp_runner
        _hooks.mlp_runner = fake_runner
        try:
            result = run_single_with_progress(
                "mlp", "taiwan", 5, "full", "cpu", None, save_model=False
            )
            assert call_count == 1
            assert result["backend"] == "mlp"
            assert result["result"]["backend"] == "mlp"
        finally:
            _hooks.mlp_runner = original

    def test_runs_lightgbm_backend(self) -> None:
        """Test run_single_with_progress uses LightGBM backend correctly."""
        call_count = 0
        phase_calls: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LightGBMTrialProgressCallbackProtocol | None = None,
            phase_callback: _hooks.LightGBMPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.LightGBMLoadingProgressCallbackProtocol | None = None,
        ) -> LightGBMOptimizationResult:
            nonlocal call_count
            call_count += 1
            # Exercise the phase callback if provided
            if phase_callback is not None:
                phase_callback(
                    {
                        "phase": "loading_data",
                        "dataset": "taiwan",
                        "n_samples": 0,
                        "n_features": 0,
                    }
                )
                phase_callback(
                    {
                        "phase": "feature_engineering",
                        "dataset": "taiwan",
                        "n_samples": 1000,
                        "n_features": 24,
                    }
                )
                phase_callback(
                    {
                        "phase": "optimizing",
                        "dataset": "taiwan",
                        "n_samples": 1000,
                        "n_features": 100,
                    }
                )
                phase_callback(
                    {
                        "phase": "saving",
                        "dataset": "taiwan",
                        "n_samples": 1000,
                        "n_features": 100,
                    }
                )
                phase_calls.append("called")
            if progress_callback is not None:
                # Note: LightGBM progress info doesn't include best_max_depth
                # because it's fixed at -1 (num_leaves controls complexity)
                progress_callback(
                    {
                        "trial_number": 1,
                        "n_trials_total": 5,
                        "current_auc": 0.80,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": True,
                        "best_learning_rate": 0.1,
                        "best_n_estimators": 100,
                        "best_num_leaves": 31,
                    }
                )
            return make_fake_lightgbm_result()

        original = _hooks.lightgbm_runner
        _hooks.lightgbm_runner = fake_runner
        try:
            result = run_single_with_progress(
                "lightgbm", "taiwan", 5, "full", "cpu", None, save_model=False
            )
            assert call_count == 1
            assert result["backend"] == "lightgbm"
            assert result["result"]["backend"] == "lightgbm"
            assert len(phase_calls) == 1  # Phase callback was called
        finally:
            _hooks.lightgbm_runner = original

    def test_runs_lstm_backend(self) -> None:
        """Test run_single_with_progress uses LSTM backend correctly."""
        call_count = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LSTMTrialProgressCallbackProtocol | None = None,
            phase_callback: _hooks.LSTMPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.LSTMLoadingProgressCallbackProtocol | None = None,
        ) -> LSTMOptimizationResult:
            nonlocal call_count
            call_count += 1
            # Exercise the phase callback if provided
            if phase_callback is not None:
                phase_callback(
                    {
                        "phase": "loading_data",
                        "dataset": "taiwan",
                        "n_samples": 0,
                        "n_features": 0,
                    }
                )
                phase_callback(
                    {
                        "phase": "feature_engineering",
                        "dataset": "taiwan",
                        "n_samples": 1000,
                        "n_features": 100,
                    }
                )
                phase_callback(
                    {
                        "phase": "optimizing",
                        "dataset": "taiwan",
                        "n_samples": 1000,
                        "n_features": 150,
                    }
                )
                phase_callback(
                    {
                        "phase": "saving",
                        "dataset": "taiwan",
                        "n_samples": 1000,
                        "n_features": 150,
                    }
                )
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial_number": 1,
                        "n_trials_total": 5,
                        "current_auc": 0.80,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": True,
                        "best_learning_rate": 0.001,
                        "best_num_layers": 2,
                        "best_hidden_size": 64,
                        "best_dropout": 0.2,
                    }
                )
            return make_fake_lstm_result()

        original = _hooks.lstm_runner
        _hooks.lstm_runner = fake_runner
        try:
            result = run_single_with_progress(
                "lstm", "taiwan", 5, "full", "cpu", None, save_model=False
            )
            assert call_count == 1
            assert result["backend"] == "lstm"
            assert result["result"]["backend"] == "lstm"
        finally:
            _hooks.lstm_runner = original

    def test_run_single_with_save_model_true(self, tmp_path: Path) -> None:
        """Test run_single_with_progress with save_model=True.

        This covers the save_model branch in run_single_backend (modes.py line 363).
        """
        # Set up fake registries using shared classes
        fake_backend = FakeSaveModelBackend()
        fake_registry = ClassifierRegistry()
        fake_registry.register("xgboost", BackendRegistration(lambda: fake_backend))
        fake_dataset_reg = DatasetRegistry((make_fake_dataset_config("taiwan"),))

        # Store originals
        orig_runner = _hooks.xgboost_runner
        orig_backend_reg = _hooks.backend_registry_factory
        orig_dataset_reg = _hooks.dataset_registry_factory
        orig_loader = _hooks.dataset_loader

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
            phase_callback: _hooks.XGBoostPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.XGBoostLoadingProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            _ = progress_callback  # Available for progress reporting
            _ = phase_callback  # Available for phase reporting
            return make_fake_result()

        _hooks.xgboost_runner = fake_runner
        _hooks.backend_registry_factory = lambda: fake_registry
        _hooks.dataset_registry_factory = lambda: fake_dataset_reg
        _hooks.dataset_loader = lambda cfg, ext_dir: make_fake_loaded_dataset()

        try:
            # Create necessary directories
            (tmp_path / "data" / "external").mkdir(parents=True, exist_ok=True)
            (tmp_path / "models").mkdir(parents=True, exist_ok=True)

            # Run with save_model=True
            result = run_single_with_progress(
                "xgboost", "taiwan", 5, "full", "cpu", None, save_model=True, project_root=tmp_path
            )

            # Verify the result is returned correctly
            assert result["backend"] == "xgboost"
            assert result["result"]["backend"] == "xgboost"

            # Verify model was saved
            model_dir = tmp_path / "models" / "xgboost"
            assert model_dir.exists()
        finally:
            _hooks.xgboost_runner = orig_runner
            _hooks.backend_registry_factory = orig_backend_reg
            _hooks.dataset_registry_factory = orig_dataset_reg
            _hooks.dataset_loader = orig_loader
