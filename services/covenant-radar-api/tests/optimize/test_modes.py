"""Tests for scripts/optimize mode functions.

Tests compare_presets, run_single_with_progress, and run_all_datasets modes.
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
from scripts.optimize.modes import (
    compare_presets,
    run_all_datasets,
    run_single_with_progress,
)

from .conftest import (
    FakeSaveModelBackend,
    make_fake_dataset_config,
    make_fake_lightgbm_result,
    make_fake_loaded_dataset,
    make_fake_lstm_result,
    make_fake_mlp_result,
    make_fake_result,
)


class TestComparePresets:
    """Tests for compare_presets function."""

    def test_runs_all_presets_xgboost(self) -> None:
        """Test compare_presets runs all four presets with XGBoost."""
        presets_called: list[str] = []
        callback_calls = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
            phase_callback: _hooks.XGBoostPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.XGBoostLoadingProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            nonlocal callback_calls
            _ = phase_callback  # Available for phase reporting
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
                callback_calls += 1
                progress_callback(
                    {
                        "trial_number": 2,
                        "n_trials_total": 5,
                        "current_auc": 0.75,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": False,
                        "best_learning_rate": 0.1,
                        "best_max_depth": 6,
                        "best_n_estimators": 100,
                    }
                )
                callback_calls += 1
            # Extract preset from config
            if "none" in config_json and "log_only" not in config_json:
                presets_called.append("none")
                return make_fake_result(feature_preset="none", best_val_auc=0.75, n_features=20)
            if "log_only" in config_json:
                presets_called.append("log_only")
                return make_fake_result(feature_preset="log_only", best_val_auc=0.80, n_features=40)
            if "ratios_only" in config_json:
                presets_called.append("ratios_only")
                return make_fake_result(
                    feature_preset="ratios_only", best_val_auc=0.82, n_features=500
                )
            presets_called.append("full")
            return make_fake_result(feature_preset="full", best_val_auc=0.85, n_features=800)

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            compare_presets("xgboost", "taiwan", 10, "cpu", None, save_model=False)
            assert len(presets_called) == 4
            assert "none" in presets_called
            assert "log_only" in presets_called
            assert "ratios_only" in presets_called
            assert "full" in presets_called
            assert callback_calls == 8  # 2 calls per preset * 4 presets
        finally:
            _hooks.xgboost_runner = original

    def test_runs_all_presets_mlp(self) -> None:
        """Test compare_presets runs all four presets with MLP."""
        presets_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.MLPTrialProgressCallbackProtocol | None = None,
            phase_callback: _hooks.MLPPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.MLPLoadingProgressCallbackProtocol | None = None,
        ) -> MLPOptimizationResult:
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
            _ = phase_callback  # Available for phase reporting
            # Extract preset from config
            if "none" in config_json and "log_only" not in config_json:
                presets_called.append("none")
            elif "log_only" in config_json:
                presets_called.append("log_only")
            elif "ratios_only" in config_json:
                presets_called.append("ratios_only")
            else:
                presets_called.append("full")
            return make_fake_mlp_result()

        original = _hooks.mlp_runner
        _hooks.mlp_runner = fake_runner
        try:
            compare_presets("mlp", "taiwan", 10, "cpu", None, save_model=False)
            assert len(presets_called) == 4
        finally:
            _hooks.mlp_runner = original

    def test_runs_all_presets_lightgbm(self) -> None:
        """Test compare_presets runs all four presets with LightGBM."""
        presets_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LightGBMTrialProgressCallbackProtocol | None = None,
            phase_callback: _hooks.LightGBMPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.LightGBMLoadingProgressCallbackProtocol | None = None,
        ) -> LightGBMOptimizationResult:
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
            if "none" in config_json and "log_only" not in config_json:
                presets_called.append("none")
            elif "log_only" in config_json:
                presets_called.append("log_only")
            elif "ratios_only" in config_json:
                presets_called.append("ratios_only")
            else:
                presets_called.append("full")
            return make_fake_lightgbm_result()

        original = _hooks.lightgbm_runner
        _hooks.lightgbm_runner = fake_runner
        try:
            compare_presets("lightgbm", "taiwan", 10, "cpu", None, save_model=False)
            assert len(presets_called) == 4
        finally:
            _hooks.lightgbm_runner = original

    def test_runs_all_presets_lstm(self) -> None:
        """Test compare_presets runs all four presets with LSTM."""
        presets_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LSTMTrialProgressCallbackProtocol | None = None,
            phase_callback: _hooks.LSTMPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.LSTMLoadingProgressCallbackProtocol | None = None,
        ) -> LSTMOptimizationResult:
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
            _ = phase_callback  # Available for phase reporting
            if "none" in config_json and "log_only" not in config_json:
                presets_called.append("none")
            elif "log_only" in config_json:
                presets_called.append("log_only")
            elif "ratios_only" in config_json:
                presets_called.append("ratios_only")
            else:
                presets_called.append("full")
            return make_fake_lstm_result()

        original = _hooks.lstm_runner
        _hooks.lstm_runner = fake_runner
        try:
            compare_presets("lstm", "taiwan", 10, "cpu", None, save_model=False)
            assert len(presets_called) == 4
        finally:
            _hooks.lstm_runner = original

    def test_compare_presets_with_save_model_true(self, tmp_path: Path) -> None:
        """Test compare_presets with save_model=True.

        This covers the save_model branch in compare_presets (modes.py line 453).
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

        presets_called: list[str] = []

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
            # Track which preset was called
            if "none" in config_json and "log_only" not in config_json:
                presets_called.append("none")
                return make_fake_result(feature_preset="none", best_val_auc=0.75, n_features=20)
            if "log_only" in config_json:
                presets_called.append("log_only")
                return make_fake_result(feature_preset="log_only", best_val_auc=0.80, n_features=40)
            if "ratios_only" in config_json:
                presets_called.append("ratios_only")
                return make_fake_result(
                    feature_preset="ratios_only", best_val_auc=0.82, n_features=500
                )
            presets_called.append("full")
            return make_fake_result(feature_preset="full", best_val_auc=0.85, n_features=800)

        _hooks.xgboost_runner = fake_runner
        _hooks.backend_registry_factory = lambda: fake_registry
        _hooks.dataset_registry_factory = lambda: fake_dataset_reg
        _hooks.dataset_loader = lambda cfg, ext_dir: make_fake_loaded_dataset()

        try:
            # Create necessary directories
            (tmp_path / "data" / "external").mkdir(parents=True, exist_ok=True)
            (tmp_path / "models").mkdir(parents=True, exist_ok=True)

            # Run with save_model=True
            compare_presets(
                "xgboost", "taiwan", 5, "cpu", None, save_model=True, project_root=tmp_path
            )

            # Verify all presets were called
            assert len(presets_called) == 4
            assert "none" in presets_called
            assert "log_only" in presets_called
            assert "ratios_only" in presets_called
            assert "full" in presets_called

            # Verify model was saved (at least one preset creates a model)
            model_dir = tmp_path / "models" / "xgboost"
            assert model_dir.exists()
        finally:
            _hooks.xgboost_runner = orig_runner
            _hooks.backend_registry_factory = orig_backend_reg
            _hooks.dataset_registry_factory = orig_dataset_reg
            _hooks.dataset_loader = orig_loader


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


class TestRunAllDatasets:
    """Tests for run_all_datasets function."""

    def test_runs_all_three_datasets(self) -> None:
        """Test run_all_datasets runs on taiwan, us, and polish with varying AUCs."""
        datasets_called: list[str] = []
        callback_calls = 0

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
            phase_callback: _hooks.XGBoostPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.XGBoostLoadingProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            nonlocal callback_calls
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
                callback_calls += 1
                progress_callback(
                    {
                        "trial_number": 2,
                        "n_trials_total": 5,
                        "current_auc": 0.75,
                        "best_auc": 0.80,
                        "best_trial": 1,
                        "is_best": False,
                        "best_learning_rate": 0.1,
                        "best_max_depth": 6,
                        "best_n_estimators": 100,
                    }
                )
                callback_calls += 1
            # Return different AUC values to cover both best and non-best formatting
            if "taiwan" in config_json:
                datasets_called.append("taiwan")
                return make_fake_result(dataset="taiwan", best_val_auc=0.90)  # Best
            if '"us"' in config_json:
                datasets_called.append("us")
                return make_fake_result(dataset="us", best_val_auc=0.85)  # Not best
            datasets_called.append("polish")
            return make_fake_result(dataset="polish", best_val_auc=0.82)  # Not best

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            run_all_datasets("xgboost", 10, "full", "cpu", None, save_model=False)
            assert len(datasets_called) == 3
            assert "taiwan" in datasets_called
            assert "us" in datasets_called
            assert "polish" in datasets_called
            assert callback_calls == 6  # 2 calls per dataset * 3 datasets
        finally:
            _hooks.xgboost_runner = original
