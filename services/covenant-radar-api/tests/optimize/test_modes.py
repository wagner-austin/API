"""Tests for scripts/optimize mode functions.

Tests compare_presets, run_all_datasets, and run_multiple_backends modes.
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
    run_multiple_backends,
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


class TestRunMultipleBackends:
    """Tests for run_multiple_backends function."""

    def test_runs_multiple_backends_with_summary(self) -> None:
        """Test run_multiple_backends runs each backend and displays comparison."""
        backends_called: list[str] = []

        def fake_xgboost_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
            phase_callback: _hooks.XGBoostPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.XGBoostLoadingProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            backends_called.append("xgboost")
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial_number": 1,
                        "n_trials_total": 5,
                        "current_auc": 0.85,
                        "best_auc": 0.85,
                        "best_trial": 1,
                        "is_best": True,
                        "best_learning_rate": 0.1,
                        "best_max_depth": 6,
                        "best_n_estimators": 100,
                    }
                )
            return make_fake_result(dataset="taiwan", best_val_auc=0.85)

        def fake_lightgbm_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LightGBMTrialProgressCallbackProtocol | None = None,
            phase_callback: _hooks.LightGBMPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.LightGBMLoadingProgressCallbackProtocol | None = None,
        ) -> LightGBMOptimizationResult:
            backends_called.append("lightgbm")
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial_number": 1,
                        "n_trials_total": 5,
                        "current_auc": 0.90,
                        "best_auc": 0.90,
                        "best_trial": 1,
                        "is_best": True,
                        "best_learning_rate": 0.05,
                        "best_num_leaves": 31,
                        "best_n_estimators": 150,
                    }
                )
            return make_fake_lightgbm_result(dataset="taiwan", best_val_auc=0.90)

        original_xgb = _hooks.xgboost_runner
        original_lgb = _hooks.lightgbm_runner
        _hooks.xgboost_runner = fake_xgboost_runner
        _hooks.lightgbm_runner = fake_lightgbm_runner
        try:
            run_multiple_backends(
                ("lightgbm", "xgboost"),
                "taiwan",
                5,
                "full",
                "cpu",
                None,
                save_model=False,
            )
            assert len(backends_called) == 2
            assert "lightgbm" in backends_called
            assert "xgboost" in backends_called
        finally:
            _hooks.xgboost_runner = original_xgb
            _hooks.lightgbm_runner = original_lgb

    def test_runs_single_backend_no_summary(self) -> None:
        """Test single backend doesn't print summary table."""
        backends_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
            phase_callback: _hooks.XGBoostPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.XGBoostLoadingProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            backends_called.append("xgboost")
            if progress_callback is not None:
                progress_callback(
                    {
                        "trial_number": 1,
                        "n_trials_total": 5,
                        "current_auc": 0.85,
                        "best_auc": 0.85,
                        "best_trial": 1,
                        "is_best": True,
                        "best_learning_rate": 0.1,
                        "best_max_depth": 6,
                        "best_n_estimators": 100,
                    }
                )
            return make_fake_result(dataset="taiwan", best_val_auc=0.85)

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            # Single backend shouldn't print comparison summary
            run_multiple_backends(
                ("xgboost",),
                "taiwan",
                5,
                "full",
                "cpu",
                None,
                save_model=False,
            )
            assert len(backends_called) == 1
        finally:
            _hooks.xgboost_runner = original

    def test_runs_all_four_backends(self) -> None:
        """Test run_multiple_backends with all 4 backends."""
        backends_called: list[str] = []

        def fake_xgboost_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
            phase_callback: _hooks.XGBoostPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.XGBoostLoadingProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            backends_called.append("xgboost")
            return make_fake_result(dataset="taiwan", best_val_auc=0.85)

        def fake_lightgbm_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LightGBMTrialProgressCallbackProtocol | None = None,
            phase_callback: _hooks.LightGBMPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.LightGBMLoadingProgressCallbackProtocol | None = None,
        ) -> LightGBMOptimizationResult:
            backends_called.append("lightgbm")
            return make_fake_lightgbm_result(dataset="taiwan", best_val_auc=0.90)

        def fake_mlp_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.MLPTrialProgressCallbackProtocol | None = None,
            phase_callback: _hooks.MLPPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.MLPLoadingProgressCallbackProtocol | None = None,
        ) -> MLPOptimizationResult:
            backends_called.append("mlp")
            return make_fake_mlp_result(dataset="taiwan", best_val_auc=0.80)

        def fake_lstm_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.LSTMTrialProgressCallbackProtocol | None = None,
            phase_callback: _hooks.LSTMPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.LSTMLoadingProgressCallbackProtocol | None = None,
        ) -> LSTMOptimizationResult:
            backends_called.append("lstm")
            return make_fake_lstm_result(dataset="taiwan", best_val_auc=0.78)

        original_xgb = _hooks.xgboost_runner
        original_lgb = _hooks.lightgbm_runner
        original_mlp = _hooks.mlp_runner
        original_lstm = _hooks.lstm_runner
        _hooks.xgboost_runner = fake_xgboost_runner
        _hooks.lightgbm_runner = fake_lightgbm_runner
        _hooks.mlp_runner = fake_mlp_runner
        _hooks.lstm_runner = fake_lstm_runner
        try:
            run_multiple_backends(
                ("xgboost", "lightgbm", "mlp", "lstm"),
                "taiwan",
                5,
                "full",
                "cpu",
                None,
                save_model=False,
            )
            assert len(backends_called) == 4
            assert "xgboost" in backends_called
            assert "lightgbm" in backends_called
            assert "mlp" in backends_called
            assert "lstm" in backends_called
        finally:
            _hooks.xgboost_runner = original_xgb
            _hooks.lightgbm_runner = original_lgb
            _hooks.mlp_runner = original_mlp
            _hooks.lstm_runner = original_lstm

    def test_with_custom_project_root(self, tmp_path: Path) -> None:
        """Test run_multiple_backends with custom project_root (covers else branch)."""
        # Create required directories
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / "external").mkdir(parents=True, exist_ok=True)

        backends_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: _hooks.XGBoostProgressCallbackProtocol | None = None,
            phase_callback: _hooks.XGBoostPhaseCallbackProtocol | None = None,
            loading_progress_callback: _hooks.XGBoostLoadingProgressCallbackProtocol | None = None,
        ) -> XGBoostOptimizationResult:
            backends_called.append("xgboost")
            return make_fake_result()

        original = _hooks.xgboost_runner
        _hooks.xgboost_runner = fake_runner
        try:
            run_multiple_backends(
                ("xgboost",),
                "taiwan",
                5,
                "full",
                "cpu",
                None,
                save_model=False,
                project_root=tmp_path,  # Custom project_root to cover else branch
            )
            assert len(backends_called) == 1
        finally:
            _hooks.xgboost_runner = original
