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
from covenant_ml.types import (
    BackendName,
)
from numpy.typing import NDArray
from platform_core.json_utils import (
    JSONTypeError,
    dump_json_str,
)

from covenant_radar_api.worker import _test_hooks as worker_hooks
from covenant_radar_api.worker._optimize_common import (
    parse_bidirectional,
    parse_nn_optimizer,
    parse_precision,
)
from covenant_radar_api.worker._test_hooks import ObjectiveWithFeatureCount
from covenant_radar_api.worker.optimize_job import (
    _parse_optimize_config,
)
from covenant_radar_api.worker.optimize_types import (
    UnifiedOptimizeParseResult,
)
from tests._optimize_job_fixtures import (
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


class TestParsePrecision:
    """Tests for parse_precision function."""

    def test_defaults_to_fp32(self) -> None:
        """None input returns 'fp32'."""
        assert parse_precision(None) == "fp32"

    def test_accepts_fp32(self) -> None:
        """'fp32' is accepted."""
        assert parse_precision("fp32") == "fp32"

    def test_accepts_fp16(self) -> None:
        """'fp16' is accepted."""
        assert parse_precision("fp16") == "fp16"

    def test_accepts_bf16(self) -> None:
        """'bf16' is accepted."""
        assert parse_precision("bf16") == "bf16"

    def test_accepts_auto(self) -> None:
        """'auto' is accepted."""
        assert parse_precision("auto") == "auto"

    def test_rejects_invalid_string(self) -> None:
        """Invalid precision raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="precision must be one of"):
            parse_precision("fp64")

    def test_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="precision must be a string"):
            parse_precision(16)


class TestParseNnOptimizer:
    """Tests for parse_nn_optimizer function."""

    def test_defaults_to_adamw(self) -> None:
        """None input returns 'adamw'."""
        assert parse_nn_optimizer(None) == "adamw"

    def test_accepts_adamw(self) -> None:
        """'adamw' is accepted."""
        assert parse_nn_optimizer("adamw") == "adamw"

    def test_accepts_adam(self) -> None:
        """'adam' is accepted."""
        assert parse_nn_optimizer("adam") == "adam"

    def test_accepts_sgd(self) -> None:
        """'sgd' is accepted."""
        assert parse_nn_optimizer("sgd") == "sgd"

    def test_rejects_invalid_string(self) -> None:
        """Invalid optimizer raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="optimizer must be one of"):
            parse_nn_optimizer("rmsprop")

    def test_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="optimizer must be a string"):
            parse_nn_optimizer(42)


class TestParseBidirectional:
    """Tests for parse_bidirectional function."""

    def test_defaults_to_false(self) -> None:
        """None input returns False."""
        assert parse_bidirectional(None) is False

    def test_accepts_true(self) -> None:
        """True is accepted."""
        assert parse_bidirectional(True) is True

    def test_accepts_false(self) -> None:
        """False is accepted."""
        assert parse_bidirectional(False) is False

    def test_rejects_non_bool(self) -> None:
        """Non-boolean input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="bidirectional must be a boolean"):
            parse_bidirectional("yes")


class TestParseOptimizeConfig:
    """Tests for _parse_optimize_config function."""

    def setup_method(self) -> None:
        """Install fake dataset registries before each test."""
        self._orig_dataset_registry = worker_hooks.dataset_registry_factory
        self._orig_ts_registry = worker_hooks.timeseries_registry_factory
        worker_hooks.dataset_registry_factory = _make_fake_standard_registry
        worker_hooks.timeseries_registry_factory = _make_fake_timeseries_registry

    def teardown_method(self) -> None:
        """Restore original hooks after each test."""
        worker_hooks.dataset_registry_factory = self._orig_dataset_registry
        worker_hooks.timeseries_registry_factory = self._orig_ts_registry

    def test_minimal_config_returns_defaults(self) -> None:
        """Minimal config uses defaults for all optional fields."""
        config_json = _make_config_json()
        result = _parse_optimize_config(config_json)

        assert result["backend"] == "xgboost"
        assert result["dataset"] == "taiwan"
        assert result["n_trials"] == 10
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
            backend="mlp",
            dataset="us",
            n_trials=100,
            timeout_seconds=3600,
            device="cuda",
            feature_preset="full",
            random_state=123,
            early_stopping_rounds=20,
            n_jobs=4,
            precision="fp16",
            optimizer="adam",
            n_epochs=100,
            early_stopping_patience=15,
            sequence_length=10,
            bidirectional=True,
        )
        result = _parse_optimize_config(config_json)

        assert result["backend"] == "mlp"
        assert result["dataset"] == "us"
        assert result["n_trials"] == 100
        assert result["timeout_seconds"] == 3600
        assert result["device"] == "cuda"
        assert result["feature_preset"] == "full"
        assert result["random_state"] == 123
        assert result["early_stopping_rounds"] == 20
        assert result["n_jobs"] == 4
        assert result["precision"] == "fp16"
        assert result["nn_optimizer"] == "adam"
        assert result["n_epochs"] == 100
        assert result["early_stopping_patience"] == 15
        assert result["sequence_length"] == 10
        assert result["bidirectional"] is True

    def test_all_seven_backends_accepted(self) -> None:
        """All 7 backends are accepted."""
        backends: tuple[str, ...] = (
            "xgboost",
            "mlp",
            "lstm",
            "lightgbm",
            "cleargbm",
            "logreg",
            "random_forest",
        )
        for backend in backends:
            config_json = _make_config_json(backend=backend)
            result = _parse_optimize_config(config_json)
            assert result["backend"] == backend

    def test_invalid_backend_raises(self) -> None:
        """Invalid backend raises ValueError."""
        config_json = _make_config_json(backend="invalid")
        with pytest.raises(ValueError, match="backend must be one of"):
            _parse_optimize_config(config_json)

    def test_missing_dataset_raises(self) -> None:
        """Missing dataset field raises JSONTypeError."""
        config_json = dump_json_str({"backend": "xgboost", "n_trials": 10})
        with pytest.raises(JSONTypeError):
            _parse_optimize_config(config_json)

    def test_invalid_dataset_raises(self) -> None:
        """Invalid dataset name raises ValueError."""
        config_json = _make_config_json(dataset="nonexistent")
        with pytest.raises(ValueError, match="dataset must be one of"):
            _parse_optimize_config(config_json)

    def test_missing_n_trials_raises(self) -> None:
        """Missing n_trials field raises JSONTypeError."""
        config_json = dump_json_str({"backend": "xgboost", "dataset": "taiwan"})
        with pytest.raises(JSONTypeError):
            _parse_optimize_config(config_json)

    def test_non_object_config_raises(self) -> None:
        """Non-object config raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="config must be a JSON object"):
            _parse_optimize_config('"just a string"')

    def test_invalid_precision_raises(self) -> None:
        """Invalid precision raises JSONTypeError."""
        config_json = _make_config_json(precision="fp64")
        with pytest.raises(JSONTypeError, match="precision must be one of"):
            _parse_optimize_config(config_json)

    def test_invalid_nn_optimizer_raises(self) -> None:
        """Invalid optimizer raises JSONTypeError."""
        config_json = _make_config_json(optimizer="rmsprop")
        with pytest.raises(JSONTypeError, match="optimizer must be one of"):
            _parse_optimize_config(config_json)

    def test_invalid_bidirectional_raises(self) -> None:
        """Invalid bidirectional raises JSONTypeError."""
        config_json = _make_config_json(bidirectional="yes")
        with pytest.raises(JSONTypeError, match="bidirectional must be a boolean"):
            _parse_optimize_config(config_json)

    def test_invalid_timeout_type_raises(self) -> None:
        """Non-integer timeout_seconds raises JSONTypeError."""
        config_json = _make_config_json(timeout_seconds="an_hour")
        with pytest.raises(JSONTypeError, match="timeout_seconds must be an integer"):
            _parse_optimize_config(config_json)

    def test_null_timeout_is_none(self) -> None:
        """Explicit null timeout_seconds results in None."""
        config_json = _make_config_json(timeout_seconds=None)
        result = _parse_optimize_config(config_json)
        assert result["timeout_seconds"] is None


class TestProcessOptimizeJob:
    """Tests for process_optimize_job RQ entry point."""

    def setup_method(self) -> None:
        """Install all fake hooks before each test."""
        self._orig_registry = worker_hooks.registry_factory
        self._orig_optimizer = worker_hooks.optimizer_registry_factory
        self._orig_objective = worker_hooks.objective_factory
        self._orig_dataset_registry = worker_hooks.dataset_registry_factory
        self._orig_ts_registry = worker_hooks.timeseries_registry_factory
        self._orig_dataset_loader = worker_hooks.dataset_loader

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

    def test_process_optimize_job_returns_encoded_result(
        self,
        tmp_path: Path,
    ) -> None:
        """process_optimize_job returns JSON-serializable dict."""
        from platform_core.config import _test_hooks as config_hooks
        from platform_core.testing import FakeEnv

        from covenant_radar_api.worker.optimize_job import process_optimize_job

        fake_backend = _FakeBackend()
        fake_optimizer = _FakeOptimizer()
        fake_objective = _FakeObjective()
        fake_dataset = _make_fake_dataset()

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
            result = process_optimize_job(config_json)

            assert result["backend"] == "xgboost"
            assert result["status"] == "complete"
            assert result["dataset"] == "taiwan"
        finally:
            config_hooks.get_env = orig_get_env
