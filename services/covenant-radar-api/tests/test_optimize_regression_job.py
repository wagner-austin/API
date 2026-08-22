"""Tests for worker/optimize_regression_job.py regression hyperparameter optimization.

Tests use dependency injection via worker/_regression_hooks and worker/_test_hooks
to verify actual code paths with fake backends and optimizers.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONTypeError,
    dump_json_str,
)

from covenant_radar_api.worker import _regression_hooks as regression_hooks
from covenant_radar_api.worker import _test_hooks as worker_hooks
from covenant_radar_api.worker.optimize_regression_job import (
    _make_regression_trial_callback,
    _parse_regression_optimize_config,
    _report_regression_phase,
)
from covenant_radar_api.worker.optimize_regression_results import (
    RegressionOptimizePhase,
    RegressionPhaseProgressInfo,
    RegressionTrialProgressInfo,
)
from tests._optimize_regression_job_fixtures import (
    _FakeOptimizer,
    _FakeRegressorBackend,
    _make_config_json,
    _make_fake_objective_factory,
    _make_fake_optimizer_registry,
    _make_fake_regression_loader,
    _make_fake_regression_registry,
    _make_fake_regressor_registry,
    _make_fake_standard_registry,
    _make_fake_timeseries_registry,
    _make_trial_result,
)


class TestParseRegressionOptimizeConfig:
    """Tests for _parse_regression_optimize_config function."""

    def setup_method(self) -> None:
        """Install fake regression registry before each test."""
        self._orig_regression_registry = regression_hooks.regression_registry_factory
        regression_hooks.regression_registry_factory = _make_fake_regression_registry

    def teardown_method(self) -> None:
        """Restore original hooks after each test."""
        regression_hooks.regression_registry_factory = self._orig_regression_registry

    def test_minimal_config_returns_defaults(self) -> None:
        """Minimal config uses defaults for all optional fields."""
        config_json = _make_config_json()
        result = _parse_regression_optimize_config(config_json)

        assert result["backend"] == "xgboost_reg"
        assert result["dataset"] == "financial_distress"
        assert result["n_trials"] == 5
        assert result["timeout_seconds"] is None
        assert result["device"] == "auto"
        assert result["feature_preset"] == "none"
        assert result["random_state"] == 42
        assert result["early_stopping_rounds"] == 10
        assert result["n_jobs"] == -1
        assert result["precision"] == "fp32"
        assert result["nn_optimizer"] == "adamw"
        assert result["n_epochs"] == 50
        assert result["early_stopping_patience"] == 10
        assert result["sequence_length"] == 5
        assert result["bidirectional"] is False

    def test_full_config_all_fields(self) -> None:
        """Full config with all fields specified."""
        config_json = _make_config_json(
            backend="lightgbm_reg",
            dataset="financial_distress",
            n_trials=50,
            timeout_seconds=3600,
            device="cuda",
            feature_preset="full",
            random_state=123,
            early_stopping_rounds=20,
            n_jobs=4,
            precision="fp16",
            optimizer="adam",
            n_epochs=100,
            early_stopping_patience=20,
            sequence_length=10,
            bidirectional=True,
        )
        result = _parse_regression_optimize_config(config_json)

        assert result["backend"] == "lightgbm_reg"
        assert result["dataset"] == "financial_distress"
        assert result["n_trials"] == 50
        assert result["timeout_seconds"] == 3600
        assert result["device"] == "cuda"
        assert result["feature_preset"] == "full"
        assert result["random_state"] == 123
        assert result["early_stopping_rounds"] == 20
        assert result["n_jobs"] == 4
        assert result["precision"] == "fp16"
        assert result["nn_optimizer"] == "adam"
        assert result["n_epochs"] == 100
        assert result["early_stopping_patience"] == 20
        assert result["sequence_length"] == 10
        assert result["bidirectional"] is True

    def test_optimizer_wire_key_matches_classifier(self) -> None:
        """The optimizer is read from the 'optimizer' wire key, not 'nn_optimizer'.

        Regression guard: this path previously read raw['nn_optimizer'] while the
        classifier path, the external-train parser, and the API all send
        'optimizer'. Because parse_nn_optimizer defaults None to 'adamw', a
        client-supplied optimizer was silently discarded with no error.
        """
        for wire_value in ("adam", "sgd"):
            config_json = _make_config_json(backend="mlp_reg", optimizer=wire_value)
            result = _parse_regression_optimize_config(config_json)
            assert result["nn_optimizer"] == wire_value

    def test_all_four_backends_accepted(self) -> None:
        """All 4 regressor backends are accepted."""
        backends: tuple[str, ...] = ("xgboost_reg", "lightgbm_reg", "mlp_reg", "lstm_reg")
        for backend in backends:
            config_json = _make_config_json(backend=backend)
            result = _parse_regression_optimize_config(config_json)
            assert result["backend"] == backend

    def test_invalid_backend_raises(self) -> None:
        """Invalid backend raises ValueError."""
        config_json = _make_config_json(backend="invalid")
        with pytest.raises(ValueError, match="backend must be one of"):
            _parse_regression_optimize_config(config_json)

    def test_missing_dataset_raises(self) -> None:
        """Missing dataset field raises JSONTypeError."""
        config_json = dump_json_str({"backend": "xgboost_reg", "n_trials": 5})
        with pytest.raises(JSONTypeError, match="Missing required field 'dataset'"):
            _parse_regression_optimize_config(config_json)

    def test_invalid_dataset_raises(self) -> None:
        """Invalid dataset name raises ValueError."""
        config_json = _make_config_json(dataset="nonexistent")
        with pytest.raises(ValueError, match="dataset must be one of"):
            _parse_regression_optimize_config(config_json)

    def test_missing_n_trials_raises(self) -> None:
        """Missing n_trials raises JSONTypeError."""
        config_json = dump_json_str({"backend": "xgboost_reg", "dataset": "financial_distress"})
        with pytest.raises(JSONTypeError, match="Missing required field 'n_trials'"):
            _parse_regression_optimize_config(config_json)

    def test_non_dict_config_raises(self) -> None:
        """Non-object JSON raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="config must be a JSON object"):
            _parse_regression_optimize_config('"just a string"')

    def test_invalid_timeout_type_raises(self) -> None:
        """Non-integer timeout raises JSONTypeError."""
        config_json = _make_config_json(timeout_seconds="fast")
        with pytest.raises(JSONTypeError, match="timeout_seconds must be an integer"):
            _parse_regression_optimize_config(config_json)

    def test_null_timeout_allowed(self) -> None:
        """Null timeout results in None."""
        config_json = _make_config_json(timeout_seconds=None)
        result = _parse_regression_optimize_config(config_json)
        assert result["timeout_seconds"] is None

    def test_device_cpu(self) -> None:
        """CPU device is accepted."""
        config_json = _make_config_json(device="cpu")
        result = _parse_regression_optimize_config(config_json)
        assert result["device"] == "cpu"

    def test_feature_preset_log_only(self) -> None:
        """log_only feature preset is accepted."""
        config_json = _make_config_json(feature_preset="log_only")
        result = _parse_regression_optimize_config(config_json)
        assert result["feature_preset"] == "log_only"


class TestReportRegressionPhase:
    """Tests for _report_regression_phase function."""

    def test_calls_callback_with_correct_info(self) -> None:
        """Callback receives correctly populated RegressionPhaseProgressInfo."""
        received: list[RegressionPhaseProgressInfo] = []

        def _callback(info: RegressionPhaseProgressInfo) -> None:
            received.append(info)

        _report_regression_phase(
            _callback,
            "loading_data",
            "xgboost_reg",
            "financial_distress",
            100,
            10,
        )

        assert len(received) == 1
        info = received[0]
        assert info["phase"] == "loading_data"
        assert info["backend"] == "xgboost_reg"
        assert info["dataset"] == "financial_distress"
        assert info["n_samples"] == 100
        assert info["n_features"] == 10

    def test_none_callback_is_safe(self) -> None:
        """None callback does not raise."""
        _report_regression_phase(None, "optimizing", "xgboost_reg", "test", 50, 5)

    def test_all_phases(self) -> None:
        """All four phases can be reported."""
        received: list[RegressionPhaseProgressInfo] = []

        def _callback(info: RegressionPhaseProgressInfo) -> None:
            received.append(info)

        phases: tuple[RegressionOptimizePhase, ...] = (
            "loading_data",
            "feature_engineering",
            "optimizing",
            "saving",
        )
        for phase in phases:
            _report_regression_phase(_callback, phase, "lightgbm_reg", "test", 0, 0)

        assert len(received) == 4
        assert [r["phase"] for r in received] == list(phases)


class TestMakeRegressionTrialCallback:
    """Tests for _make_regression_trial_callback function."""

    def test_tracks_best_value(self) -> None:
        """Trial callback tracks best value correctly."""
        received: list[RegressionTrialProgressInfo] = []

        def _progress(info: RegressionTrialProgressInfo) -> None:
            received.append(info)

        callback = _make_regression_trial_callback("xgboost_reg", 5, _progress)

        # First trial is always best
        callback(_make_trial_result(trial_number=0, value=-0.5))
        assert received[0]["is_best"] is True
        assert received[0]["best_value"] == -0.5
        assert received[0]["best_trial"] == 0

        # Worse trial (more negative = worse for neg RMSE)
        callback(_make_trial_result(trial_number=1, value=-0.8))
        assert received[1]["is_best"] is False
        assert received[1]["best_value"] == -0.5
        assert received[1]["best_trial"] == 0

        # Better trial (less negative = better)
        callback(_make_trial_result(trial_number=2, value=-0.3))
        assert received[2]["is_best"] is True
        assert received[2]["best_value"] == -0.3
        assert received[2]["best_trial"] == 2

    def test_none_callback_is_safe(self) -> None:
        """None progress callback does not raise."""
        callback = _make_regression_trial_callback("xgboost_reg", 5, None)
        callback(_make_trial_result(trial_number=0, value=-0.5))

    def test_backend_name_propagated(self) -> None:
        """Backend name is included in progress info."""
        received: list[RegressionTrialProgressInfo] = []

        def _progress(info: RegressionTrialProgressInfo) -> None:
            received.append(info)

        callback = _make_regression_trial_callback("lightgbm_reg", 10, _progress)
        callback(_make_trial_result(trial_number=0, value=-0.5))

        assert received[0]["backend"] == "lightgbm_reg"
        assert received[0]["n_trials_total"] == 10

    def test_current_value_always_reported(self) -> None:
        """Current trial value is always reported regardless of best status."""
        received: list[RegressionTrialProgressInfo] = []

        def _progress(info: RegressionTrialProgressInfo) -> None:
            received.append(info)

        callback = _make_regression_trial_callback("xgboost_reg", 3, _progress)
        callback(_make_trial_result(trial_number=0, value=-0.5))
        callback(_make_trial_result(trial_number=1, value=-0.9))

        assert received[0]["current_value"] == -0.5
        assert received[1]["current_value"] == -0.9


class TestProcessRegressionOptimizeJob:
    """Tests for process_regression_optimize_job RQ entry point."""

    def setup_method(self) -> None:
        """Install all fake hooks before each test."""
        self._orig_regression_registry = regression_hooks.regression_registry_factory
        self._orig_regression_loader = regression_hooks.regression_dataset_loader
        self._orig_regressor_registry = regression_hooks.regressor_registry_factory
        self._orig_regressor_objective = regression_hooks.regressor_objective_factory
        self._orig_optimizer_registry = worker_hooks.optimizer_registry_factory
        self._orig_dataset_registry = worker_hooks.dataset_registry_factory
        self._orig_ts_registry = worker_hooks.timeseries_registry_factory

        regression_hooks.regression_registry_factory = _make_fake_regression_registry
        regression_hooks.regression_dataset_loader = _make_fake_regression_loader
        worker_hooks.dataset_registry_factory = _make_fake_standard_registry
        worker_hooks.timeseries_registry_factory = _make_fake_timeseries_registry

    def teardown_method(self) -> None:
        """Restore all original hooks after each test."""
        regression_hooks.regression_registry_factory = self._orig_regression_registry
        regression_hooks.regression_dataset_loader = self._orig_regression_loader
        regression_hooks.regressor_registry_factory = self._orig_regressor_registry
        regression_hooks.regressor_objective_factory = self._orig_regressor_objective
        worker_hooks.optimizer_registry_factory = self._orig_optimizer_registry
        worker_hooks.dataset_registry_factory = self._orig_dataset_registry
        worker_hooks.timeseries_registry_factory = self._orig_ts_registry

    def test_process_regression_optimize_job_returns_encoded_result(
        self,
        tmp_path: Path,
    ) -> None:
        """process_regression_optimize_job returns JSON-serializable dict."""
        from platform_core.config import _test_hooks as config_hooks
        from platform_core.testing import FakeEnv

        from covenant_radar_api.worker.optimize_regression_job import (
            process_regression_optimize_job,
        )

        fake_backend = _FakeRegressorBackend()
        fake_optimizer = _FakeOptimizer()

        regression_hooks.regressor_registry_factory = (
            lambda b=fake_backend: _make_fake_regressor_registry(b)
        )
        regression_hooks.regressor_objective_factory = _make_fake_objective_factory
        worker_hooks.optimizer_registry_factory = (
            lambda o=fake_optimizer: _make_fake_optimizer_registry(o)
        )

        fake_env = FakeEnv(
            {
                "APP__DATA_ROOT": str(tmp_path),
                "APP__MODELS_ROOT": str(tmp_path / "models"),
                "DATABASE_URL": "postgresql://test@localhost/test",
                "REDIS_URL": "redis://localhost:6379/0",
            }
        )

        orig_get_env = config_hooks.get_env
        config_hooks.get_env = fake_env
        try:
            config_json = _make_config_json()
            result = process_regression_optimize_job(config_json)

            assert result["backend"] == "xgboost_reg"
            assert result["status"] == "complete"
            assert result["dataset"] == "financial_distress"
        finally:
            config_hooks.get_env = orig_get_env
