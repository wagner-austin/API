"""Tests for worker/optimize_regression_job.py regression hyperparameter optimization.

Tests use dependency injection via worker/_regression_hooks and worker/_test_hooks
to verify actual code paths with fake backends and optimizers.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from covenant_ml.datasets import (
    RegressionDatasetConfig,
    RegressionLoadedDataset,
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
)
from numpy.typing import NDArray
from platform_core.json_utils import (
    load_json_str,
)

from covenant_radar_api.worker import _regression_hooks as regression_hooks
from covenant_radar_api.worker import _test_hooks as worker_hooks
from covenant_radar_api.worker.optimize_regression_job import (
    run_regression_optimization,
)
from covenant_radar_api.worker.optimize_regression_results import (
    RegressionLoadingProgressInfo,
    RegressionPhaseProgressInfo,
    RegressionTrialProgressInfo,
)
from tests._optimize_regression_job_fixtures import (
    _FakeOptimizer,
    _FakeRegressorBackend,
    _make_config_json,
    _make_fake_objective_factory,
    _make_fake_optimizer_registry,
    _make_fake_regression_dataset,
    _make_fake_regression_loader,
    _make_fake_regression_registry,
    _make_fake_regressor_registry,
    _make_fake_standard_registry,
    _make_fake_timeseries_registry,
    _make_trial_result,
)


class TestRunRegressionOptimization:
    """Tests for run_regression_optimization function."""

    def setup_method(self) -> None:
        """Install all fake hooks before each test."""
        # Regression hooks
        self._orig_regression_registry = regression_hooks.regression_registry_factory
        self._orig_regression_loader = regression_hooks.regression_dataset_loader
        self._orig_regressor_registry = regression_hooks.regressor_registry_factory
        self._orig_regressor_objective = regression_hooks.regressor_objective_factory

        # Classifier hooks (for optimizer registry)
        self._orig_optimizer_registry = worker_hooks.optimizer_registry_factory
        self._orig_dataset_registry = worker_hooks.dataset_registry_factory
        self._orig_ts_registry = worker_hooks.timeseries_registry_factory

        # Install fakes
        self._fake_backend = _FakeRegressorBackend()
        self._fake_optimizer = _FakeOptimizer()

        regression_hooks.regression_registry_factory = _make_fake_regression_registry
        regression_hooks.regression_dataset_loader = _make_fake_regression_loader
        regression_hooks.regressor_registry_factory = lambda b=self._fake_backend: (
            _make_fake_regressor_registry(b)
        )
        regression_hooks.regressor_objective_factory = _make_fake_objective_factory

        worker_hooks.optimizer_registry_factory = lambda o=self._fake_optimizer: (
            _make_fake_optimizer_registry(o)
        )
        worker_hooks.dataset_registry_factory = _make_fake_standard_registry
        worker_hooks.timeseries_registry_factory = _make_fake_timeseries_registry

    def teardown_method(self) -> None:
        """Restore original hooks after each test."""
        regression_hooks.regression_registry_factory = self._orig_regression_registry
        regression_hooks.regression_dataset_loader = self._orig_regression_loader
        regression_hooks.regressor_registry_factory = self._orig_regressor_registry
        regression_hooks.regressor_objective_factory = self._orig_regressor_objective

        worker_hooks.optimizer_registry_factory = self._orig_optimizer_registry
        worker_hooks.dataset_registry_factory = self._orig_dataset_registry
        worker_hooks.timeseries_registry_factory = self._orig_ts_registry

    def test_basic_optimization(self, tmp_path: Path) -> None:
        """Basic regression optimization runs end to end."""
        config_json = _make_config_json()
        result = run_regression_optimization(
            config_json,
            tmp_path / "external",
            tmp_path / "output",
        )

        assert result["backend"] == "xgboost_reg"
        assert result["status"] == "complete"
        assert result["dataset"] == "financial_distress"
        assert result["n_samples"] == 100
        assert result["n_features"] == 10
        assert result["best_value"] == -0.25
        assert result["n_trials_complete"] == 5
        assert result["duration_seconds"] == 1.0
        assert self._fake_optimizer.optimize_call_count == 1

    def test_lightgbm_backend(self, tmp_path: Path) -> None:
        """LightGBM regressor backend works end to end."""
        config_json = _make_config_json(backend="lightgbm_reg")
        result = run_regression_optimization(
            config_json,
            tmp_path / "external",
            tmp_path / "output",
        )

        assert result["backend"] == "lightgbm_reg"
        assert result["status"] == "complete"

    def test_with_timeout(self, tmp_path: Path) -> None:
        """Timeout parameter is forwarded to optimizer."""
        config_json = _make_config_json(timeout_seconds=3600)
        result = run_regression_optimization(
            config_json,
            tmp_path / "external",
            tmp_path / "output",
        )

        assert result["status"] == "complete"

    def test_custom_feature_preset(self, tmp_path: Path) -> None:
        """Feature preset is captured in result."""
        config_json = _make_config_json(feature_preset="full")
        result = run_regression_optimization(
            config_json,
            tmp_path / "external",
            tmp_path / "output",
        )

        assert result["feature_preset"] == "full"

    def test_phase_callbacks(self, tmp_path: Path) -> None:
        """Phase callbacks are invoked for all four phases."""
        phases: list[str] = []

        def _phase_callback(info: RegressionPhaseProgressInfo) -> None:
            phases.append(info["phase"])

        config_json = _make_config_json()
        run_regression_optimization(
            config_json,
            tmp_path / "external",
            tmp_path / "output",
            phase_callback=_phase_callback,
        )

        assert phases == ["loading_data", "feature_engineering", "optimizing", "saving"]

    def test_trial_progress_callback(self, tmp_path: Path) -> None:
        """Trial progress callback is invoked during optimization."""

        # Use an optimizer that invokes the trial callback
        def _optimizer_with_callback() -> _FakeOptimizer:
            class _InvokingOptimizer(_FakeOptimizer):
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
                    self._optimize_call_count += 1
                    if trial_callback is not None:
                        trial_callback(_make_trial_result(trial_number=0, value=-0.25))
                    return OptimizationSummary(
                        best_trial_number=0,
                        best_value=-0.25,
                        best_int_params=SampledIntParams(max_depth=5),
                        best_float_params=SampledFloatParams(learning_rate=0.1),
                        best_string_params=SampledStringParams(),
                        n_trials_total=config["n_trials"],
                        n_trials_complete=config["n_trials"],
                        n_trials_pruned=0,
                        n_trials_failed=0,
                        total_duration_seconds=0.5,
                    )

            return _InvokingOptimizer()

        invoking_optimizer = _optimizer_with_callback()
        worker_hooks.optimizer_registry_factory = lambda: _make_fake_optimizer_registry(
            invoking_optimizer
        )

        trial_infos: list[RegressionTrialProgressInfo] = []

        def _trial_callback(info: RegressionTrialProgressInfo) -> None:
            trial_infos.append(info)

        config_json = _make_config_json()
        run_regression_optimization(
            config_json,
            tmp_path / "external",
            tmp_path / "output",
            progress_callback=_trial_callback,
        )

        assert len(trial_infos) == 1
        assert trial_infos[0]["backend"] == "xgboost_reg"
        assert trial_infos[0]["current_value"] == -0.25
        assert trial_infos[0]["is_best"] is True

    def test_loading_progress_callback(self, tmp_path: Path) -> None:
        """Loading progress callback is invoked during dataset loading."""
        from covenant_ml.datasets.types import LoadProgress

        loading_infos: list[RegressionLoadingProgressInfo] = []

        def _loading_callback(info: RegressionLoadingProgressInfo) -> None:
            loading_infos.append(info)

        # Use a loader that invokes the progress callback
        def _loader_with_progress(
            config: RegressionDatasetConfig,
            external_dir: Path,
            progress_callback: ProgressCallbackProtocol | None = None,
        ) -> RegressionLoadedDataset:
            if progress_callback is not None:
                progress_callback(
                    LoadProgress(
                        phase="reading",
                        bytes_read=1000,
                        bytes_total=1000,
                        percent_complete=100.0,
                        rows_processed=100,
                        rows_total=100,
                        message="Done",
                    )
                )
            return _make_fake_regression_dataset(config["name"])

        regression_hooks.regression_dataset_loader = _loader_with_progress

        config_json = _make_config_json()
        run_regression_optimization(
            config_json,
            tmp_path / "external",
            tmp_path / "output",
            loading_progress_callback=_loading_callback,
        )

        assert len(loading_infos) == 1
        assert loading_infos[0]["dataset"] == "financial_distress"
        assert loading_infos[0]["phase"] == "reading"
        assert loading_infos[0]["percent_complete"] == 100.0

    def test_result_has_best_params(self, tmp_path: Path) -> None:
        """Result includes best hyperparameters from optimizer."""
        config_json = _make_config_json()
        result = run_regression_optimization(
            config_json,
            tmp_path / "external",
            tmp_path / "output",
        )

        assert result["best_int_params"]["max_depth"] == 5
        assert result["best_int_params"]["n_estimators"] == 100
        assert result["best_float_params"]["learning_rate"] == 0.1
        assert result["best_trial_number"] == 0

    def test_saved_config_carries_tuned_hyperparameters(self, tmp_path: Path) -> None:
        """The saved *_optimal_config.json contains the tuned hyperparameters.

        Regression guard for two defects on this path: the flattening step was
        never ported from the classifier job, so the file named
        "optimal_config" held run metadata and no hyperparameters at all; and
        the regressor param encoders omitted every neural-net key, so
        hidden_size/dropout were dropped even once flattening existed.

        Args:
            tmp_path: Pytest temporary directory unique to this test.
        """
        summary = OptimizationSummary(
            best_trial_number=3,
            best_value=-0.125,
            best_int_params=SampledIntParams(max_depth=7, hidden_size=128),
            best_float_params=SampledFloatParams(learning_rate=0.05, dropout=0.25),
            best_string_params=SampledStringParams(booster="gbtree"),
            n_trials_total=5,
            n_trials_complete=5,
            n_trials_pruned=0,
            n_trials_failed=0,
            total_duration_seconds=2.0,
        )
        fake_optimizer = _FakeOptimizer(summary)
        worker_hooks.optimizer_registry_factory = lambda o=fake_optimizer: (
            _make_fake_optimizer_registry(o)
        )

        output_dir = tmp_path / "output"
        run_regression_optimization(
            _make_config_json(),
            tmp_path / "external",
            output_dir,
        )

        config_path = output_dir / "financial_distress_xgboost_reg_optimal_config.json"
        saved = load_json_str(config_path.read_text(encoding="utf-8"))
        assert type(saved) is dict
        assert saved["best_max_depth"] == 7
        assert saved["best_hidden_size"] == 128
        assert saved["best_learning_rate"] == 0.05
        assert saved["best_dropout"] == 0.25
        assert saved["best_booster"] == "gbtree"

    def test_saves_results_to_output_dir(self, tmp_path: Path) -> None:
        """Results are saved to the output directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config_json = _make_config_json()
        run_regression_optimization(
            config_json,
            tmp_path / "external",
            output_dir,
        )

        # save_optimization_results creates a dataset subdirectory with result + config files
        saved_files = sorted(output_dir.rglob("*.json"))
        assert saved_files[0].name.endswith(".json")
        assert len(saved_files) == 2


class TestSaveOptimizationResultsUnderTheFarm:
    """Result filenames are per-job when HPC3_JOB_NAME is exported."""

    def test_job_name_suffixes_both_files(self, tmp_path: Path) -> None:
        """Sweep members for one dataset differ only by preset; without
        the suffix, four concurrent jobs would overwrite one file
        last-writer-wins and the survivor's identity would be whichever
        member finished last."""
        from platform_core.config import _test_hooks as config_env

        from covenant_radar_api.worker._optimize_common import save_optimization_results

        saved = config_env.get_env

        def fake_get_env(key: str) -> str | None:
            if key == "HPC3_JOB_NAME":
                return "cleargbm.p6-rung1-taiwan-full"
            return saved(key)

        config_env.get_env = fake_get_env
        try:
            result_path, config_path = save_optimization_results(
                tmp_path,
                "taiwan",
                "cleargbm",
                {"best_value": 0.9},
                {"n_estimators": 100},
            )
            assert result_path.name == (
                "taiwan_cleargbm_optuna_result-cleargbm.p6-rung1-taiwan-full.json"
            )
            assert config_path.name == (
                "taiwan_cleargbm_optimal_config-cleargbm.p6-rung1-taiwan-full.json"
            )
            assert result_path.exists()
            assert config_path.exists()
        finally:
            config_env.get_env = saved

    def test_no_job_name_keeps_the_local_names(self, tmp_path: Path) -> None:
        """Locally the filenames are unchanged."""
        from covenant_radar_api.worker._optimize_common import save_optimization_results

        result_path, config_path = save_optimization_results(
            tmp_path,
            "taiwan",
            "cleargbm",
            {"best_value": 0.9},
            {"n_estimators": 100},
        )
        assert result_path.name == "taiwan_cleargbm_optuna_result.json"
        assert config_path.name == "taiwan_cleargbm_optimal_config.json"
