"""Tests for worker/optimize_job.py unified hyperparameter optimization job.

Tests use dependency injection via worker/_test_hooks to verify actual code paths.
All code paths are tested with strong assertions on actual behavior.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from covenant_ml.datasets import (
    DatasetConfig,
    LoadedDataset,
)
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.optimizer import (
    SearchSpace,
)
from covenant_ml.optimizer.protocol import ObjectiveProtocol, TrialCallbackProtocol
from covenant_ml.optimizer.types import (
    OptimizationConfig,
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    TrialResult,
)
from covenant_ml.types import (
    BackendName,
)
from numpy.typing import NDArray
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
)

from covenant_radar_api.worker import _test_hooks as worker_hooks
from covenant_radar_api.worker._test_hooks import ObjectiveWithFeatureCount
from covenant_radar_api.worker.optimize_job import (
    run_optimization,
)
from covenant_radar_api.worker.optimize_types import (
    LoadingProgressInfo,
    PhaseProgressInfo,
    TrialProgressInfo,
    UnifiedOptimizeParseResult,
)
from tests._optimize_job_fixtures import (
    _FAKE_SEARCH_SPACE,
    _FakeBackend,
    _FakeObjective,
    _FakeOptimizer,
    _make_config_json,
    _make_fake_backend_registry,
    _make_fake_dataset,
    _make_fake_optimizer_registry,
    _make_fake_standard_registry,
    _make_fake_timeseries_registry,
)


class TestRunOptimization:
    """Tests for run_optimization using fake worker_hooks."""

    def setup_method(self) -> None:
        """Install all fake hooks before each test."""
        self._orig_registry = worker_hooks.registry_factory
        self._orig_optimizer = worker_hooks.optimizer_registry_factory
        self._orig_objective = worker_hooks.objective_factory
        self._orig_dataset_registry = worker_hooks.dataset_registry_factory
        self._orig_ts_registry = worker_hooks.timeseries_registry_factory
        self._orig_dataset_loader = worker_hooks.dataset_loader

        # Install fake dataset registries
        worker_hooks.dataset_registry_factory = _make_fake_standard_registry
        worker_hooks.timeseries_registry_factory = _make_fake_timeseries_registry

    def teardown_method(self) -> None:
        """Restore all original hooks after each test."""
        worker_hooks.registry_factory = self._orig_registry
        worker_hooks.optimizer_registry_factory = self._orig_optimizer
        worker_hooks.objective_factory = self._orig_objective
        worker_hooks.dataset_registry_factory = self._orig_dataset_registry
        worker_hooks.timeseries_registry_factory = self._orig_ts_registry
        worker_hooks.dataset_loader = self._orig_dataset_loader

    def _install_fakes(
        self,
        backend: _FakeBackend | None = None,
        optimizer: _FakeOptimizer | None = None,
        objective: _FakeObjective | None = None,
        dataset: LoadedDataset | None = None,
    ) -> tuple[_FakeBackend, _FakeOptimizer, _FakeObjective]:
        """Install fake hooks and return the fakes for assertion.

        Args:
            backend: Optional fake backend (defaults to new _FakeBackend).
            optimizer: Optional fake optimizer (defaults to new _FakeOptimizer).
            objective: Optional fake objective (defaults to new _FakeObjective).
            dataset: Optional fake dataset (defaults to _make_fake_dataset).

        Returns:
            Tuple of (fake_backend, fake_optimizer, fake_objective).
        """
        fake_backend = backend or _FakeBackend()
        fake_optimizer = optimizer or _FakeOptimizer()
        fake_objective = objective or _FakeObjective()
        fake_dataset = dataset or _make_fake_dataset()

        fake_registry = _make_fake_backend_registry(fake_backend)
        fake_optimizer_registry = _make_fake_optimizer_registry(fake_optimizer)

        worker_hooks.registry_factory = lambda: fake_registry

        worker_hooks.optimizer_registry_factory = lambda: fake_optimizer_registry

        def _fake_objective_factory(
            backend_name: BackendName,
            x: NDArray[np.float64],
            y: NDArray[np.int64],
            feature_names: list[str],
            config: UnifiedOptimizeParseResult,
        ) -> ObjectiveWithFeatureCount:
            del backend_name, x, y, feature_names, config
            return fake_objective

        worker_hooks.objective_factory = _fake_objective_factory

        def _fake_loader(
            config: DatasetConfig,
            external_dir: Path,
            progress_callback: ProgressCallbackProtocol | None = None,
        ) -> LoadedDataset:
            del config, external_dir, progress_callback
            return fake_dataset

        worker_hooks.dataset_loader = _fake_loader

        return fake_backend, fake_optimizer, fake_objective

    def test_run_optimization_returns_result(self, tmp_path: Path) -> None:
        """run_optimization returns UnifiedOptimizationResult."""
        _, _, _ = self._install_fakes()
        config_json = _make_config_json()
        result = run_optimization(config_json, tmp_path, tmp_path / "output")

        assert result["backend"] == "xgboost"
        assert result["status"] == "complete"
        assert result["dataset"] == "taiwan"
        assert result["n_samples"] == 100
        assert result["n_features"] == 10
        assert result["best_value"] == 0.85
        assert result["n_trials_complete"] == 10

    def test_optimizer_called_once(self, tmp_path: Path) -> None:
        """Optimizer.optimize is called exactly once."""
        _, fake_optimizer, _ = self._install_fakes()
        config_json = _make_config_json()
        run_optimization(config_json, tmp_path, tmp_path / "output")

        assert fake_optimizer.optimize_call_count == 1

    def test_backend_search_space_used(self, tmp_path: Path) -> None:
        """Backend's get_default_search_space is called and passed to optimizer."""
        fake_backend, fake_optimizer, _ = self._install_fakes()
        config_json = _make_config_json()
        run_optimization(config_json, tmp_path, tmp_path / "output")

        assert fake_backend.get_search_space_called
        assert fake_optimizer.last_search_space == _FAKE_SEARCH_SPACE

    def test_optimization_config_populated(self, tmp_path: Path) -> None:
        """OptimizationConfig is correctly built from parsed config."""
        _, fake_optimizer, _ = self._install_fakes()
        config_json = _make_config_json(n_trials=25, timeout_seconds=1800, random_state=99)
        run_optimization(config_json, tmp_path, tmp_path / "output")

        config = fake_optimizer.last_config
        if config is None:
            pytest.fail("last_config must be set after optimize")
        assert config["n_trials"] == 25
        assert config["timeout_seconds"] == 1800
        assert config["random_state"] == 99

    def test_results_saved_to_output_dir(self, tmp_path: Path) -> None:
        """Optimization results are saved as JSON files."""
        _, _, _ = self._install_fakes()
        output_dir = tmp_path / "optuna"
        config_json = _make_config_json()
        run_optimization(config_json, tmp_path, output_dir)

        result_file = output_dir / "taiwan_xgboost_optuna_result.json"
        config_file = output_dir / "taiwan_xgboost_optimal_config.json"
        assert result_file.exists()
        assert config_file.exists()

        raw = load_json_str(result_file.read_text())
        result_data = narrow_json_to_dict(raw)
        assert result_data["backend"] == "xgboost"
        assert result_data["dataset"] == "taiwan"

    def test_phase_callbacks_called(self, tmp_path: Path) -> None:
        """Phase callback receives all 4 phases in order."""
        _, _, _ = self._install_fakes()
        phases: list[str] = []

        def _phase_cb(info: PhaseProgressInfo) -> None:
            phases.append(info["phase"])

        config_json = _make_config_json()
        run_optimization(
            config_json,
            tmp_path,
            tmp_path / "output",
            phase_callback=_phase_cb,
        )

        assert phases == ["loading_data", "feature_engineering", "optimizing", "saving"]

    def test_phase_callbacks_include_backend_and_dataset(self, tmp_path: Path) -> None:
        """Phase callback info includes correct backend and dataset."""
        _, _, _ = self._install_fakes()
        infos: list[PhaseProgressInfo] = []

        def _phase_cb(info: PhaseProgressInfo) -> None:
            infos.append(info)

        config_json = _make_config_json(backend="lightgbm", dataset="us")
        run_optimization(
            config_json,
            tmp_path,
            tmp_path / "output",
            phase_callback=_phase_cb,
        )

        for info in infos:
            assert info["backend"] == "lightgbm"
            assert info["dataset"] == "us"

    def test_loading_progress_callback(self, tmp_path: Path) -> None:
        """Loading progress callback is invoked when provided."""
        _, _, _ = self._install_fakes()

        # Override the dataset loader to call the progress callback
        fake_dataset = _make_fake_dataset()

        def _loading_loader(
            config: DatasetConfig,
            external_dir: Path,
            progress_callback: ProgressCallbackProtocol | None = None,
        ) -> LoadedDataset:
            del config, external_dir
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "reading",
                        "bytes_read": 500,
                        "bytes_total": 1000,
                        "percent_complete": 50.0,
                        "rows_processed": 50,
                        "rows_total": 100,
                        "message": "Reading CSV",
                    }
                )
            return fake_dataset

        worker_hooks.dataset_loader = _loading_loader

        loading_infos: list[LoadingProgressInfo] = []

        def _loading_cb(info: LoadingProgressInfo) -> None:
            loading_infos.append(info)

        config_json = _make_config_json()
        run_optimization(
            config_json,
            tmp_path,
            tmp_path / "output",
            loading_progress_callback=_loading_cb,
        )

        assert len(loading_infos) == 1
        assert loading_infos[0]["phase"] == "reading"
        assert loading_infos[0]["percent_complete"] == 50.0
        assert loading_infos[0]["dataset"] == "taiwan"

    def test_progress_callback_on_trial(self, tmp_path: Path) -> None:
        """Trial progress callback receives info when optimizer calls trial_callback."""
        # Create an optimizer that calls the trial callback
        summary = OptimizationSummary(
            best_trial_number=0,
            best_value=0.88,
            best_int_params=SampledIntParams(max_depth=6),
            best_float_params=SampledFloatParams(learning_rate=0.05),
            best_string_params=SampledStringParams(),
            n_trials_total=5,
            n_trials_complete=5,
            n_trials_pruned=0,
            n_trials_failed=0,
            total_duration_seconds=2.0,
        )

        class _CallbackOptimizer(_FakeOptimizer):
            """Optimizer that invokes the trial callback."""

            def optimize(
                self,
                x_features: NDArray[np.float64],
                y_labels: NDArray[np.int64],
                feature_names: list[str],
                search_space: SearchSpace,
                config: OptimizationConfig,
                objective: ObjectiveProtocol,
                trial_callback: TrialCallbackProtocol | None = None,
            ) -> OptimizationSummary:
                """Call trial callback then return summary."""
                del x_features, y_labels, feature_names, search_space, objective
                self._optimize_call_count += 1
                self._last_config = config
                if trial_callback is not None:
                    trial_callback(
                        TrialResult(
                            trial_number=0,
                            int_params=SampledIntParams(max_depth=6),
                            float_params=SampledFloatParams(learning_rate=0.05),
                            string_params=SampledStringParams(),
                            value=0.88,
                            state="complete",
                            duration_seconds=0.5,
                        )
                    )
                return summary

        callback_optimizer = _CallbackOptimizer()
        self._install_fakes(optimizer=callback_optimizer)

        trial_infos: list[TrialProgressInfo] = []

        def _trial_cb(info: TrialProgressInfo) -> None:
            trial_infos.append(info)

        config_json = _make_config_json(n_trials=5)
        run_optimization(
            config_json,
            tmp_path,
            tmp_path / "output",
            progress_callback=_trial_cb,
        )

        assert len(trial_infos) == 1
        assert trial_infos[0]["trial_number"] == 0
        assert trial_infos[0]["current_value"] == 0.88
        assert trial_infos[0]["is_best"] is True
        assert trial_infos[0]["backend"] == "xgboost"

    def test_result_includes_best_params(self, tmp_path: Path) -> None:
        """Result includes best hyperparameters from optimizer summary."""
        summary = OptimizationSummary(
            best_trial_number=7,
            best_value=0.91,
            best_int_params=SampledIntParams(max_depth=8, n_estimators=300),
            best_float_params=SampledFloatParams(learning_rate=0.03, reg_alpha=0.5),
            best_string_params=SampledStringParams(booster="gbtree"),
            n_trials_total=50,
            n_trials_complete=48,
            n_trials_pruned=1,
            n_trials_failed=1,
            total_duration_seconds=60.0,
        )
        fake_optimizer = _FakeOptimizer(result=summary)
        self._install_fakes(optimizer=fake_optimizer)

        config_json = _make_config_json(n_trials=50)
        result = run_optimization(config_json, tmp_path, tmp_path / "output")

        assert result["best_trial_number"] == 7
        assert result["best_value"] == 0.91
        assert result["best_int_params"]["max_depth"] == 8
        assert result["best_int_params"]["n_estimators"] == 300
        assert result["best_float_params"]["learning_rate"] == 0.03
        assert result["best_string_params"]["booster"] == "gbtree"
        assert result["n_trials_complete"] == 48
        assert result["n_trials_pruned"] == 1
        assert result["n_trials_failed"] == 1
        assert result["duration_seconds"] == 60.0

    def test_feature_preset_passed_through(self, tmp_path: Path) -> None:
        """Feature preset from config appears in result."""
        _, _, _ = self._install_fakes()
        config_json = _make_config_json(feature_preset="full")
        result = run_optimization(config_json, tmp_path, tmp_path / "output")

        assert result["feature_preset"] == "full"

    def test_no_callbacks_ok(self, tmp_path: Path) -> None:
        """run_optimization works without any callbacks."""
        _, _, _ = self._install_fakes()
        config_json = _make_config_json()
        result = run_optimization(config_json, tmp_path, tmp_path / "output")

        assert result["status"] == "complete"

    def test_n_features_from_objective(self, tmp_path: Path) -> None:
        """n_features in result comes from objective.n_features."""
        fake_objective = _FakeObjective(n_features=42)
        self._install_fakes(objective=fake_objective)
        config_json = _make_config_json()
        result = run_optimization(config_json, tmp_path, tmp_path / "output")

        assert result["n_features"] == 42

    def test_trial_callback_non_best_trial(self, tmp_path: Path) -> None:
        """Trial callback correctly reports is_best=False for non-best trials."""
        summary = OptimizationSummary(
            best_trial_number=1,
            best_value=0.90,
            best_int_params=SampledIntParams(max_depth=6),
            best_float_params=SampledFloatParams(learning_rate=0.05),
            best_string_params=SampledStringParams(),
            n_trials_total=3,
            n_trials_complete=3,
            n_trials_pruned=0,
            n_trials_failed=0,
            total_duration_seconds=2.0,
        )

        class _MultiTrialOptimizer(_FakeOptimizer):
            """Optimizer that calls trial callback with best and non-best trials."""

            def optimize(
                self,
                x_features: NDArray[np.float64],
                y_labels: NDArray[np.int64],
                feature_names: list[str],
                search_space: SearchSpace,
                config: OptimizationConfig,
                objective: ObjectiveProtocol,
                trial_callback: TrialCallbackProtocol | None = None,
            ) -> OptimizationSummary:
                """Call trial callback with multiple trials."""
                del x_features, y_labels, feature_names, search_space, objective
                self._optimize_call_count += 1
                self._last_config = config
                if trial_callback is not None:
                    # First trial: best
                    trial_callback(
                        TrialResult(
                            trial_number=0,
                            int_params=SampledIntParams(max_depth=5),
                            float_params=SampledFloatParams(learning_rate=0.1),
                            string_params=SampledStringParams(),
                            value=0.85,
                            state="complete",
                            duration_seconds=0.5,
                        )
                    )
                    # Second trial: new best
                    trial_callback(
                        TrialResult(
                            trial_number=1,
                            int_params=SampledIntParams(max_depth=6),
                            float_params=SampledFloatParams(learning_rate=0.05),
                            string_params=SampledStringParams(),
                            value=0.90,
                            state="complete",
                            duration_seconds=0.5,
                        )
                    )
                    # Third trial: not best
                    trial_callback(
                        TrialResult(
                            trial_number=2,
                            int_params=SampledIntParams(max_depth=4),
                            float_params=SampledFloatParams(learning_rate=0.2),
                            string_params=SampledStringParams(),
                            value=0.80,
                            state="complete",
                            duration_seconds=0.5,
                        )
                    )
                return summary

        multi_optimizer = _MultiTrialOptimizer()
        self._install_fakes(optimizer=multi_optimizer)

        trial_infos: list[TrialProgressInfo] = []

        def _trial_cb(info: TrialProgressInfo) -> None:
            trial_infos.append(info)

        config_json = _make_config_json(n_trials=3)
        run_optimization(
            config_json,
            tmp_path,
            tmp_path / "output",
            progress_callback=_trial_cb,
        )

        assert len(trial_infos) == 3
        # First trial is best (first one always is)
        assert trial_infos[0]["is_best"] is True
        assert trial_infos[0]["best_value"] == 0.85
        # Second trial is new best
        assert trial_infos[1]["is_best"] is True
        assert trial_infos[1]["best_value"] == 0.90
        # Third trial is NOT best
        assert trial_infos[2]["is_best"] is False
        assert trial_infos[2]["best_value"] == 0.90
        assert trial_infos[2]["current_value"] == 0.80

    def test_trial_callback_without_progress_callback(self, tmp_path: Path) -> None:
        """Trial callback works correctly without external progress callback."""
        summary = OptimizationSummary(
            best_trial_number=0,
            best_value=0.85,
            best_int_params=SampledIntParams(max_depth=5),
            best_float_params=SampledFloatParams(learning_rate=0.1),
            best_string_params=SampledStringParams(),
            n_trials_total=2,
            n_trials_complete=2,
            n_trials_pruned=0,
            n_trials_failed=0,
            total_duration_seconds=1.0,
        )

        class _NoProgressOptimizer(_FakeOptimizer):
            """Optimizer that calls trial callback without progress callback."""

            def optimize(
                self,
                x_features: NDArray[np.float64],
                y_labels: NDArray[np.int64],
                feature_names: list[str],
                search_space: SearchSpace,
                config: OptimizationConfig,
                objective: ObjectiveProtocol,
                trial_callback: TrialCallbackProtocol | None = None,
            ) -> OptimizationSummary:
                """Call trial callback to exercise internal state tracking."""
                del x_features, y_labels, feature_names, search_space, objective
                self._optimize_call_count += 1
                self._last_config = config
                if trial_callback is not None:
                    trial_callback(
                        TrialResult(
                            trial_number=0,
                            int_params=SampledIntParams(max_depth=5),
                            float_params=SampledFloatParams(learning_rate=0.1),
                            string_params=SampledStringParams(),
                            value=0.85,
                            state="complete",
                            duration_seconds=0.5,
                        )
                    )
                return summary

        no_progress_optimizer = _NoProgressOptimizer()
        self._install_fakes(optimizer=no_progress_optimizer)

        config_json = _make_config_json(n_trials=2)
        # Run WITHOUT progress_callback (tests the None branch)
        result = run_optimization(
            config_json,
            tmp_path,
            tmp_path / "output",
        )

        assert result["status"] == "complete"
        assert result["best_value"] == 0.85
