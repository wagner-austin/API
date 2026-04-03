"""Tests for scripts/optimize mode functions.

Tests compare_presets, run_all_datasets, and run_multiple_backends modes.
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
    PhaseProgressCallbackProtocol,
    TrialProgressCallbackProtocol,
    TrialProgressInfo,
    UnifiedOptimizationResult,
)
from scripts.optimize.modes import (
    compare_presets,
    run_all_datasets,
    run_multiple_backends,
)

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


def _make_trial_info(backend: BackendName) -> TrialProgressInfo:
    """Create a standard trial progress info.

    Args:
        backend: Backend name.

    Returns:
        TrialProgressInfo for a best trial.
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


def _make_non_best_trial_info(backend: BackendName) -> TrialProgressInfo:
    """Create a non-best trial progress info.

    Args:
        backend: Backend name.

    Returns:
        TrialProgressInfo for a non-best trial.
    """
    return {
        "backend": backend,
        "trial_number": 2,
        "n_trials_total": 5,
        "current_value": 0.75,
        "best_value": 0.80,
        "best_trial": 1,
        "is_best": False,
    }


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
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            nonlocal callback_calls
            _ = phase_callback
            if progress_callback is not None:
                progress_callback(_make_trial_info("xgboost"))
                callback_calls += 1
                progress_callback(_make_non_best_trial_info("xgboost"))
                callback_calls += 1
            if "none" in config_json and "log_only" not in config_json:
                presets_called.append("none")
                return make_fake_result(feature_preset="none", best_value=0.75, n_features=20)
            if "log_only" in config_json:
                presets_called.append("log_only")
                return make_fake_result(feature_preset="log_only", best_value=0.80, n_features=40)
            if "ratios_only" in config_json:
                presets_called.append("ratios_only")
                return make_fake_result(
                    feature_preset="ratios_only", best_value=0.82, n_features=500
                )
            presets_called.append("full")
            return make_fake_result(feature_preset="full", best_value=0.85, n_features=800)

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            compare_presets(("xgboost",), "taiwan", 10, "cpu", None, save_model=False)
            assert len(presets_called) == 4
            assert "none" in presets_called
            assert "log_only" in presets_called
            assert "ratios_only" in presets_called
            assert "full" in presets_called
            assert callback_calls == 8  # 2 calls per preset * 4 presets
        finally:
            _hooks.optimization_runner = original

    def test_runs_all_presets_mlp(self) -> None:
        """Test compare_presets runs all four presets with MLP."""
        presets_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            if progress_callback is not None:
                progress_callback(_make_trial_info("mlp"))
            _ = phase_callback
            if "none" in config_json and "log_only" not in config_json:
                presets_called.append("none")
            elif "log_only" in config_json:
                presets_called.append("log_only")
            elif "ratios_only" in config_json:
                presets_called.append("ratios_only")
            else:
                presets_called.append("full")
            return make_fake_mlp_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            compare_presets(("mlp",), "taiwan", 10, "cpu", None, save_model=False)
            assert len(presets_called) == 4
        finally:
            _hooks.optimization_runner = original

    def test_runs_all_presets_lightgbm(self) -> None:
        """Test compare_presets runs all four presets with LightGBM."""
        presets_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            if progress_callback is not None:
                progress_callback(_make_trial_info("lightgbm"))
            if "none" in config_json and "log_only" not in config_json:
                presets_called.append("none")
            elif "log_only" in config_json:
                presets_called.append("log_only")
            elif "ratios_only" in config_json:
                presets_called.append("ratios_only")
            else:
                presets_called.append("full")
            return make_fake_lightgbm_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            compare_presets(("lightgbm",), "taiwan", 10, "cpu", None, save_model=False)
            assert len(presets_called) == 4
        finally:
            _hooks.optimization_runner = original

    def test_runs_all_presets_lstm(self) -> None:
        """Test compare_presets runs all four presets with LSTM."""
        presets_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            if progress_callback is not None:
                progress_callback(_make_trial_info("lstm"))
            _ = phase_callback
            if "none" in config_json and "log_only" not in config_json:
                presets_called.append("none")
            elif "log_only" in config_json:
                presets_called.append("log_only")
            elif "ratios_only" in config_json:
                presets_called.append("ratios_only")
            else:
                presets_called.append("full")
            return make_fake_lstm_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            compare_presets(("lstm",), "taiwan", 10, "cpu", None, save_model=False)
            assert len(presets_called) == 4
        finally:
            _hooks.optimization_runner = original

    def test_runs_all_presets_cleargbm(self) -> None:
        """Test compare_presets runs all four presets with ClearGBM."""
        presets_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            if progress_callback is not None:
                progress_callback(_make_trial_info("cleargbm"))
            _ = phase_callback
            if "none" in config_json and "log_only" not in config_json:
                presets_called.append("none")
            elif "log_only" in config_json:
                presets_called.append("log_only")
            elif "ratios_only" in config_json:
                presets_called.append("ratios_only")
            else:
                presets_called.append("full")
            return make_fake_cleargbm_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            compare_presets(("cleargbm",), "taiwan", 10, "cpu", None, save_model=False)
            assert len(presets_called) == 4
        finally:
            _hooks.optimization_runner = original

    def test_compare_presets_with_save_model_true(self, tmp_path: Path) -> None:
        """Test compare_presets with save_model=True.

        This covers the save_model branch in compare_presets.
        """
        fake_backend = FakeSaveModelBackend()
        fake_registry = ClassifierRegistry()
        fake_registry.register("xgboost", BackendRegistration(lambda: fake_backend))
        fake_dataset_reg = DatasetRegistry((make_fake_dataset_config("taiwan"),))

        orig_runner = _hooks.optimization_runner
        orig_backend_reg = _hooks.backend_registry_factory
        orig_dataset_reg = _hooks.dataset_registry_factory
        orig_loader = _hooks.dataset_loader

        presets_called: list[str] = []

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
            if "none" in config_json and "log_only" not in config_json:
                presets_called.append("none")
                return make_fake_result(feature_preset="none", best_value=0.75, n_features=20)
            if "log_only" in config_json:
                presets_called.append("log_only")
                return make_fake_result(feature_preset="log_only", best_value=0.80, n_features=40)
            if "ratios_only" in config_json:
                presets_called.append("ratios_only")
                return make_fake_result(
                    feature_preset="ratios_only", best_value=0.82, n_features=500
                )
            presets_called.append("full")
            return make_fake_result(feature_preset="full", best_value=0.85, n_features=800)

        _hooks.optimization_runner = fake_runner
        _hooks.backend_registry_factory = lambda: fake_registry
        _hooks.dataset_registry_factory = lambda: fake_dataset_reg
        _hooks.dataset_loader = lambda cfg, ext_dir: make_fake_loaded_dataset()

        try:
            (tmp_path / "data" / "external").mkdir(parents=True, exist_ok=True)
            (tmp_path / "models").mkdir(parents=True, exist_ok=True)

            compare_presets(
                ("xgboost",), "taiwan", 5, "cpu", None, save_model=True, project_root=tmp_path
            )

            assert len(presets_called) == 4
            assert "none" in presets_called
            assert "log_only" in presets_called
            assert "ratios_only" in presets_called
            assert "full" in presets_called

            model_dir = tmp_path / "models" / "xgboost"
            assert model_dir.exists()
        finally:
            _hooks.optimization_runner = orig_runner
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
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            nonlocal callback_calls
            if progress_callback is not None:
                progress_callback(_make_trial_info("xgboost"))
                callback_calls += 1
                progress_callback(_make_non_best_trial_info("xgboost"))
                callback_calls += 1
            if "taiwan" in config_json:
                datasets_called.append("taiwan")
                return make_fake_result(dataset="taiwan", best_value=0.90)
            if '"us"' in config_json:
                datasets_called.append("us")
                return make_fake_result(dataset="us", best_value=0.85)
            datasets_called.append("polish")
            return make_fake_result(dataset="polish", best_value=0.82)

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            run_all_datasets("xgboost", 10, "full", "cpu", None, save_model=False)
            assert len(datasets_called) == 3
            assert "taiwan" in datasets_called
            assert "us" in datasets_called
            assert "polish" in datasets_called
            assert callback_calls == 6  # 2 calls per dataset * 3 datasets
        finally:
            _hooks.optimization_runner = original


class TestRunMultipleBackends:
    """Tests for run_multiple_backends function."""

    def test_runs_multiple_backends_with_summary(self) -> None:
        """Test run_multiple_backends runs each backend and displays comparison."""
        backends_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            if '"xgboost"' in config_json:
                backends_called.append("xgboost")
                if progress_callback is not None:
                    progress_callback(_make_trial_info("xgboost"))
                return make_fake_result(dataset="taiwan", best_value=0.85)
            backends_called.append("lightgbm")
            if progress_callback is not None:
                progress_callback(_make_trial_info("lightgbm"))
            return make_fake_lightgbm_result(dataset="taiwan", best_value=0.90)

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
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
            _hooks.optimization_runner = original

    def test_runs_single_backend_no_summary(self) -> None:
        """Test single backend doesn't print summary table."""
        backends_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            backends_called.append("xgboost")
            if progress_callback is not None:
                progress_callback(_make_trial_info("xgboost"))
            return make_fake_result(dataset="taiwan", best_value=0.85)

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
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
            _hooks.optimization_runner = original

    def test_runs_all_four_backends(self) -> None:
        """Test run_multiple_backends with four backends."""
        backends_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            if '"xgboost"' in config_json:
                backends_called.append("xgboost")
                return make_fake_result(dataset="taiwan", best_value=0.85)
            if '"lightgbm"' in config_json:
                backends_called.append("lightgbm")
                return make_fake_lightgbm_result(dataset="taiwan", best_value=0.90)
            if '"mlp"' in config_json:
                backends_called.append("mlp")
                return make_fake_mlp_result(dataset="taiwan", best_value=0.80)
            backends_called.append("lstm")
            return make_fake_lstm_result(dataset="taiwan", best_value=0.78)

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
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
            _hooks.optimization_runner = original

    def test_with_custom_project_root(self, tmp_path: Path) -> None:
        """Test run_multiple_backends with custom project_root (covers else branch)."""
        (tmp_path / "models").mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / "external").mkdir(parents=True, exist_ok=True)

        backends_called: list[str] = []

        def fake_runner(
            config_json: str,
            external_dir: Path,
            output_dir: Path,
            progress_callback: TrialProgressCallbackProtocol | None = None,
            phase_callback: PhaseProgressCallbackProtocol | None = None,
            loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
        ) -> UnifiedOptimizationResult:
            backends_called.append("xgboost")
            return make_fake_result()

        original = _hooks.optimization_runner
        _hooks.optimization_runner = fake_runner
        try:
            run_multiple_backends(
                ("xgboost",),
                "taiwan",
                5,
                "full",
                "cpu",
                None,
                save_model=False,
                project_root=tmp_path,
            )
            assert len(backends_called) == 1
        finally:
            _hooks.optimization_runner = original
