"""Tests for streaming worker entry point."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from covenant_ml.types import PredictorProtocol
from covenant_persistence.protocols import ConnectionProtocol
from platform_core.testing import make_fake_env

from covenant_radar_api import streaming_worker_entry_hooks as _hooks
from covenant_radar_api.streaming._test_hooks_repositories import (
    FakeCovenantRepository,
    FakeCovenantResultRepository,
    FakeDealRepository,
    FakeMeasurementRepository,
)
from covenant_radar_api.streaming.config import StreamingConfig
from covenant_radar_api.streaming_worker_entry import (
    ModelType,
    StreamingWorkerDeps,
    _create_connection,
    _create_repositories,
    _create_worker,
    _load_encoders,
    _load_metrics_config,
    _load_model,
    _parse_model_type,
    _run_worker,
    main,
)
from covenant_radar_api.streaming_worker_entry_hooks import (
    ConnectionFactoryProtocol,
    FakeConnection,
    FakeCursor,
    LoggerProtocol,
    RepositoryFactoryProtocol,
    _fake_connection_factory,
)

# =============================================================================
# Fixtures
# =============================================================================


class _RecordingLogger:
    """Logger that records calls for testing."""

    def __init__(self) -> None:
        self.info_messages: list[tuple[str, dict[str, str] | None]] = []
        self.error_messages: list[tuple[str, dict[str, str] | None]] = []

    def info(self, message: str, *, extra: dict[str, str] | None = None) -> None:
        """Record info message."""
        self.info_messages.append((message, extra))

    def error(self, message: str, *, extra: dict[str, str] | None = None) -> None:
        """Record error message."""
        self.error_messages.append((message, extra))


@pytest.fixture()
def recording_logger() -> _RecordingLogger:
    """Create a recording logger for testing."""
    return _RecordingLogger()


@pytest.fixture()
def restore_hooks() -> Generator[None, None, None]:
    """Fixture that restores _hooks after test."""
    # Save original hooks
    orig_connection_factory = _hooks.connection_factory
    orig_repository_factory = _hooks.repository_factory
    orig_xgboost_loader = _hooks.xgboost_loader
    orig_logger_factory = _hooks.logger_factory

    yield

    # Restore original hooks
    _hooks.connection_factory = orig_connection_factory
    _hooks.repository_factory = orig_repository_factory
    _hooks.xgboost_loader = orig_xgboost_loader
    _hooks.logger_factory = orig_logger_factory


def _make_test_streaming_config(enabled: bool = True) -> StreamingConfig:
    """Create a test streaming config.

    Args:
        enabled: Whether streaming is enabled.

    Returns:
        StreamingConfig for testing.
    """
    return {
        "confluent": {
            "bootstrap_servers": "test:9092",
            "api_key": "test-key",
            "api_secret": "test-secret",
            "security_protocol": "SASL_SSL",
            "sasl_mechanism": "PLAIN",
        },
        "schema_registry": None,
        "topics": {
            "measurements": "test.measurements.v1",
            "predictions": "test.predictions.v1",
            "alerts": "test.alerts.v1",
            "dlq": "test.dlq.v1",
        },
        "consumer": {
            "group_id": "test-group",
            "auto_offset_reset": "earliest",
            "enable_auto_commit": False,
            "fetch_min_bytes": 1,
            "session_timeout_ms": 45000,
            "heartbeat_interval_ms": 15000,
        },
        "producer": {
            "acks": "all",
            "retries": 3,
            "linger_ms": 5,
            "batch_size": 16384,
            "compression_type": "gzip",
        },
        "enabled": enabled,
    }


# =============================================================================
# Tests for _parse_model_type
# =============================================================================


class TestParseModelType:
    """Tests for _parse_model_type function."""

    def test_xgboost_valid(self) -> None:
        """Test xgboost is valid."""
        result = _parse_model_type("xgboost")
        assert result == "xgboost"

    def test_lightgbm_valid(self) -> None:
        """Test lightgbm is valid."""
        result = _parse_model_type("lightgbm")
        assert result == "lightgbm"

    def test_logreg_valid(self) -> None:
        """Test logreg is valid."""
        result = _parse_model_type("logreg")
        assert result == "logreg"

    def test_random_forest_valid(self) -> None:
        """Test random_forest is valid."""
        result = _parse_model_type("random_forest")
        assert result == "random_forest"

    def test_mlp_valid(self) -> None:
        """Test mlp is valid."""
        result = _parse_model_type("mlp")
        assert result == "mlp"

    def test_invalid_raises_error(self) -> None:
        """Test invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid MODEL_TYPE"):
            _parse_model_type("invalid")


# =============================================================================
# Tests for _load_metrics_config
# =============================================================================


class TestLoadMetricsConfig:
    """Tests for _load_metrics_config function."""

    def test_returns_default_values(self) -> None:
        """Test returns default Datadog config."""
        env = make_fake_env()
        _ = env  # No env vars set

        config = _load_metrics_config()

        assert config["host"] == "localhost"
        assert config["port"] == 8125
        assert config["namespace"] == "covenant"

    def test_reads_env_vars(self) -> None:
        """Test reads custom env vars."""
        env = make_fake_env()
        env.set("DD_AGENT_HOST", "custom-host")
        env.set("DD_DOGSTATSD_PORT", "9999")

        config = _load_metrics_config()

        assert config["host"] == "custom-host"
        assert config["port"] == 9999


# =============================================================================
# Tests for _load_encoders
# =============================================================================


class TestLoadEncoders:
    """Tests for _load_encoders function."""

    def test_returns_sector_encoder(self) -> None:
        """Test returns sector encoder with expected keys and values."""
        sector_encoder, _ = _load_encoders()

        # Verify all expected sectors with exact values
        assert sector_encoder["Technology"] == 0
        assert sector_encoder["Healthcare"] == 1
        assert sector_encoder["Finance"] == 2
        assert sector_encoder["Manufacturing"] == 3
        assert sector_encoder["Retail"] == 4
        assert sector_encoder["Energy"] == 5
        assert sector_encoder["Real Estate"] == 6
        assert sector_encoder["Other"] == 7
        assert len(sector_encoder) == 8

    def test_returns_region_encoder(self) -> None:
        """Test returns region encoder with expected keys and values."""
        _, region_encoder = _load_encoders()

        # Verify all expected regions with exact values
        assert region_encoder["North America"] == 0
        assert region_encoder["Europe"] == 1
        assert region_encoder["Asia Pacific"] == 2
        assert region_encoder["Latin America"] == 3
        assert region_encoder["Middle East"] == 4
        assert region_encoder["Africa"] == 5
        assert len(region_encoder) == 6


# =============================================================================
# Tests for _create_connection
# =============================================================================


class TestCreateConnection:
    """Tests for _create_connection function."""

    def test_uses_connection_factory_hook(
        self,
        restore_hooks: None,
    ) -> None:
        """Test uses connection_factory hook."""
        env = make_fake_env()
        env.set("DATABASE_URL", "postgresql://test:5432/testdb")

        received_urls: list[str] = []
        fake_conn = FakeConnection()

        class RecordingConnectionFactory:
            """Factory that records calls and returns fake connection."""

            def __call__(self, database_url: str) -> ConnectionProtocol:
                received_urls.append(database_url)
                return fake_conn

        _hooks.connection_factory = RecordingConnectionFactory()

        conn = _create_connection()

        assert len(received_urls) == 1
        assert received_urls[0] == "postgresql://test:5432/testdb"
        # Verify we get the exact instance we returned
        assert conn is fake_conn

    def test_raises_when_database_url_missing(
        self,
        restore_hooks: None,
    ) -> None:
        """Test raises when DATABASE_URL is not set."""
        env = make_fake_env()
        _ = env  # No DATABASE_URL

        _hooks.connection_factory = _fake_connection_factory

        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            _create_connection()


# =============================================================================
# Tests for _create_repositories
# =============================================================================


class TestCreateRepositories:
    """Tests for _create_repositories function."""

    def test_uses_repository_factory_hook(
        self,
        restore_hooks: None,
    ) -> None:
        """Test uses repository_factory hook."""
        received_conns: list[ConnectionProtocol] = []

        # Create typed fake repositories
        fake_deal_repo = FakeDealRepository()
        fake_covenant_repo = FakeCovenantRepository()
        fake_measurement_repo = FakeMeasurementRepository()
        fake_result_repo = FakeCovenantResultRepository()

        def _recording_factory(
            conn: ConnectionProtocol,
        ) -> tuple[
            FakeDealRepository,
            FakeCovenantRepository,
            FakeMeasurementRepository,
            FakeCovenantResultRepository,
        ]:
            received_conns.append(conn)
            return fake_deal_repo, fake_covenant_repo, fake_measurement_repo, fake_result_repo

        _hooks.repository_factory = _recording_factory

        fake_conn = FakeConnection()
        deal_repo, covenant_repo, measurement_repo, result_repo = _create_repositories(fake_conn)

        assert len(received_conns) == 1
        assert received_conns[0] is fake_conn
        # Verify we get the exact instances we returned
        assert deal_repo is fake_deal_repo
        assert covenant_repo is fake_covenant_repo
        assert measurement_repo is fake_measurement_repo
        assert result_repo is fake_result_repo


# =============================================================================
# Tests for main
# =============================================================================


class TestMain:
    """Tests for main function."""

    def test_returns_error_when_streaming_disabled(
        self,
        recording_logger: _RecordingLogger,
    ) -> None:
        """Test returns 1 when streaming is disabled."""
        config = _make_test_streaming_config(enabled=False)

        result = main(
            streaming_config=config,
            deps=None,
            logger=recording_logger,
        )

        assert result == 1
        assert len(recording_logger.error_messages) == 1
        assert "disabled" in recording_logger.error_messages[0][0].lower()

    def test_logs_error_message_when_disabled(
        self,
        recording_logger: _RecordingLogger,
    ) -> None:
        """Test logs appropriate error when streaming disabled."""
        config = _make_test_streaming_config(enabled=False)

        main(
            streaming_config=config,
            deps=None,
            logger=recording_logger,
        )

        assert len(recording_logger.error_messages) == 1
        msg, _ = recording_logger.error_messages[0]
        assert "STREAMING__ENABLED" in msg


# =============================================================================
# Tests for LoggerProtocol
# =============================================================================


class TestLoggerProtocol:
    """Tests for LoggerProtocol interface."""

    def test_recording_logger_implements_protocol(self) -> None:
        """Test _RecordingLogger implements LoggerProtocol."""
        logger: LoggerProtocol = _RecordingLogger()

        # Should not raise
        logger.info("test message")
        logger.info("test with extra", extra={"key": "value"})
        logger.error("error message")
        logger.error("error with extra", extra={"err": "details"})


# =============================================================================
# Tests for FakeConnection
# =============================================================================


class TestFakeConnection:
    """Tests for FakeConnection class."""

    def test_init_creates_unclosed_connection(self) -> None:
        """Test init creates connection in open state."""
        conn = FakeConnection()

        assert conn.closed is False
        assert conn.committed is False
        assert conn.rolled_back is False

    def test_cursor_returns_fake_cursor(self) -> None:
        """Test cursor returns FakeCursor with expected attributes."""
        conn = FakeConnection()

        cursor = conn.cursor()

        # Verify the cursor has the expected initial state
        assert cursor.executed_queries == []
        assert cursor.rowcount == 0

    def test_commit_sets_committed_flag(self) -> None:
        """Test commit sets committed flag."""
        conn = FakeConnection()

        conn.commit()

        assert conn.committed is True

    def test_rollback_sets_rolled_back_flag(self) -> None:
        """Test rollback sets rolled_back flag."""
        conn = FakeConnection()

        conn.rollback()

        assert conn.rolled_back is True

    def test_close_sets_closed_flag(self) -> None:
        """Test close sets closed flag."""
        conn = FakeConnection()

        conn.close()

        assert conn.closed is True


# =============================================================================
# Tests for Module Exports
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_streaming_worker_entry_exports_main(self) -> None:
        """Test streaming_worker_entry exports main function."""
        from covenant_radar_api import streaming_worker_entry

        assert callable(streaming_worker_entry.main)

    def test_streaming_worker_entry_exports_types(self) -> None:
        """Test streaming_worker_entry exports expected types."""

        # ModelType is a Literal type alias - verify it accepts valid values
        model_type: ModelType = "xgboost"
        assert model_type == "xgboost"

        # StreamingWorkerDeps is a TypedDict - verify it has expected keys
        assert "consumer" in StreamingWorkerDeps.__annotations__
        assert "producer" in StreamingWorkerDeps.__annotations__
        assert "metrics" in StreamingWorkerDeps.__annotations__
        assert "model" in StreamingWorkerDeps.__annotations__

    def test_hooks_module_exports_protocols(self) -> None:
        """Test hooks module exports expected protocols."""
        from covenant_radar_api.streaming_worker_entry_hooks import (
            LoggerFactoryProtocol,
            LoggerProtocol,
            ModelLoaderProtocol,
        )

        # Verify all protocols are classes (Protocol is a metaclass)
        assert callable(ConnectionFactoryProtocol)
        assert callable(LoggerProtocol)
        assert callable(LoggerFactoryProtocol)
        assert callable(ModelLoaderProtocol)
        assert callable(RepositoryFactoryProtocol)

    def test_hooks_module_exports_factories(self) -> None:
        """Test hooks module exports factory functions."""
        from covenant_radar_api.streaming_worker_entry_hooks import (
            connection_factory,
            logger_factory,
            repository_factory,
            xgboost_loader,
        )

        # Verify hooks are callable
        assert callable(connection_factory)
        assert callable(logger_factory)
        assert callable(repository_factory)
        assert callable(xgboost_loader)

    def test_hooks_module_exports_fakes(self) -> None:
        """Test hooks module exports fake implementations."""
        from covenant_radar_api.streaming_worker_entry_hooks import (
            FakeConnection,
            _fake_connection_factory,
        )

        # Verify fakes can be instantiated and have expected attributes
        conn = FakeConnection()
        assert conn.closed is False

        cursor = FakeCursor()
        assert cursor.executed_queries == []

        # Verify factory is callable
        assert callable(_fake_connection_factory)


# =============================================================================
# Tests for FakeCursor
# =============================================================================


class TestFakeCursor:
    """Tests for FakeCursor class."""

    def test_execute_records_query(self) -> None:
        """Test execute records query and params."""
        cursor = FakeCursor()

        cursor.execute("SELECT * FROM table WHERE id = %s", (1,))

        assert len(cursor.executed_queries) == 1
        assert cursor.executed_queries[0] == ("SELECT * FROM table WHERE id = %s", (1,))

    def test_execute_multiple_queries(self) -> None:
        """Test execute records multiple queries."""
        cursor = FakeCursor()

        cursor.execute("SELECT 1", ())
        cursor.execute("SELECT 2", ("param",))

        assert len(cursor.executed_queries) == 2
        assert cursor.executed_queries[0] == ("SELECT 1", ())
        assert cursor.executed_queries[1] == ("SELECT 2", ("param",))

    def test_fetchone_returns_none_when_empty(self) -> None:
        """Test fetchone returns None when no rows."""
        cursor = FakeCursor()

        result = cursor.fetchone()

        assert result is None

    def test_fetchone_returns_row(self) -> None:
        """Test fetchone returns row and removes it from list."""
        cursor = FakeCursor()
        cursor._rows = [(1, "test"), (2, "other")]

        result = cursor.fetchone()

        assert result == (1, "test")
        assert cursor._rows == [(2, "other")]

    def test_fetchall_returns_all_rows(self) -> None:
        """Test fetchall returns all rows and clears list."""
        cursor = FakeCursor()
        cursor._rows = [(1, "a"), (2, "b"), (3, "c")]

        result = cursor.fetchall()

        assert result == [(1, "a"), (2, "b"), (3, "c")]
        assert cursor._rows == []

    def test_fetchall_returns_empty_when_no_rows(self) -> None:
        """Test fetchall returns empty list when no rows."""
        cursor = FakeCursor()

        result = cursor.fetchall()

        assert result == []

    def test_rowcount_returns_set_value(self) -> None:
        """Test rowcount returns configured value."""
        cursor = FakeCursor()
        cursor._rowcount = 5

        assert cursor.rowcount == 5


# =============================================================================
# Tests for real hook implementations
# =============================================================================


class TestRealLoggerFactory:
    """Tests for _real_logger_factory function."""

    def test_returns_logger(self) -> None:
        """Test returns logger that implements LoggerProtocol."""
        from covenant_radar_api.streaming_worker_entry_hooks import _real_logger_factory

        logger = _real_logger_factory("test_module")

        # Should not raise - logger implements protocol
        logger.info("test message")
        logger.info("test with extra", extra={"key": "value"})
        logger.error("error message")


# =============================================================================
# Tests for _load_model
# =============================================================================


class TestLoadModel:
    """Tests for _load_model function."""

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Test raises FileNotFoundError for non-existent path."""
        fake_path = tmp_path / "nonexistent_model.ubj"
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            _load_model(fake_path, "xgboost")

    def test_xgboost_uses_hook(
        self,
        tmp_path: Path,
        restore_hooks: None,
    ) -> None:
        """Test xgboost model type uses xgboost_loader hook."""
        from covenant_radar_api.streaming._test_hooks_model import FakePredictor

        model_file = tmp_path / "model.json"
        model_file.write_text("{}")
        fake_model = FakePredictor()

        class _FakeLoader:
            """Fake model loader that returns fake_model."""

            def __call__(self, model_path: str) -> PredictorProtocol:
                assert model_path == str(model_file)
                return fake_model

        _hooks.xgboost_loader = _FakeLoader()

        result = _load_model(model_file, "xgboost")
        assert result is fake_model

    def test_lightgbm_loads_model(self, tmp_path: Path) -> None:
        """Test lightgbm model type loads real LightGBM model."""
        import numpy as np
        from covenant_ml.backends.lightgbm.backend import _get_lightgbm_imports
        from numpy.typing import NDArray

        lgbm_ctor, _ = _get_lightgbm_imports()

        x: NDArray[np.float64] = np.zeros((4, 2), dtype=np.float64)
        x[0, 0] = 1.0
        x[0, 1] = 2.0
        x[1, 0] = 3.0
        x[1, 1] = 4.0
        x[2, 0] = 5.0
        x[2, 1] = 6.0
        x[3, 0] = 7.0
        x[3, 1] = 8.0
        y: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
        y[2] = 1
        y[3] = 1
        clf = lgbm_ctor(
            boosting_type="gbdt",
            num_leaves=2,
            max_depth=2,
            learning_rate=0.1,
            n_estimators=5,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            min_child_samples=1,
            random_state=42,
            n_jobs=-1,
            device="cpu",
            objective="binary",
            metric="binary_logloss",
            class_weight=None,
            verbose=-1,
        )
        clf.fit(x, y, eval_set=[(x, y)])

        model_path = tmp_path / "model.txt"
        clf.booster_.save_model(str(model_path))

        result = _load_model(model_path, "lightgbm")
        proba = result.predict_proba(x)
        assert proba.shape == (4, 2)

    def test_logreg_loads_model(self, tmp_path: Path) -> None:
        """Test logreg model type loads real LogReg model."""
        import numpy as np
        from covenant_ml.backends.logreg.backend import (
            _create_logreg_model,
            _get_joblib_imports,
        )
        from numpy.typing import NDArray

        x: NDArray[np.float64] = np.zeros((4, 2), dtype=np.float64)
        x[0, 0] = 1.0
        x[0, 1] = 2.0
        x[1, 0] = 3.0
        x[1, 1] = 4.0
        x[2, 0] = 5.0
        x[2, 1] = 6.0
        x[3, 0] = 7.0
        x[3, 1] = 8.0
        y: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
        y[2] = 1
        y[3] = 1
        lr = _create_logreg_model(
            penalty="l2",
            inverse_reg_strength=1.0,
            solver="lbfgs",
            max_iter=100,
            tol=1e-4,
            random_state=42,
            class_weight=None,
            l1_ratio=None,
            n_jobs=-1,
        )
        lr.fit(x, y)
        model_path = tmp_path / "model.joblib"
        dump_fn, _ = _get_joblib_imports()
        dump_fn(lr, str(model_path))

        result = _load_model(model_path, "logreg")
        proba = result.predict_proba(x)
        assert proba.shape == (4, 2)

    def test_random_forest_loads_model(self, tmp_path: Path) -> None:
        """Test random_forest model type loads real Random Forest model."""
        import numpy as np
        from covenant_ml.backends.random_forest.backend import _get_sklearn_imports
        from numpy.typing import NDArray

        rf_ctor, dump_fn, _ = _get_sklearn_imports()

        x: NDArray[np.float64] = np.zeros((4, 2), dtype=np.float64)
        x[0, 0] = 1.0
        x[0, 1] = 2.0
        x[1, 0] = 3.0
        x[1, 1] = 4.0
        x[2, 0] = 5.0
        x[2, 1] = 6.0
        x[3, 0] = 7.0
        x[3, 1] = 8.0
        y: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
        y[2] = 1
        y[3] = 1
        rf = rf_ctor(
            n_estimators=2,
            max_depth=2,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            class_weight=None,
            n_jobs=-1,
            random_state=42,
            oob_score=False,
        )
        rf.fit(x, y)
        model_path = tmp_path / "model.joblib"
        dump_fn(rf, str(model_path))

        result = _load_model(model_path, "random_forest")
        proba = result.predict_proba(x)
        assert proba.shape == (4, 2)

    def test_mlp_loads_model(self, tmp_path: Path) -> None:
        """Test mlp model type loads MLP model from state dict + metadata."""
        import numpy as np
        from numpy.typing import NDArray
        from platform_core.json_utils import dump_json_str

        from covenant_radar_api.worker._model_loaders import _build_mlp_model

        # Build a real model and save its state dict
        n_features = 4
        hidden_size = 8
        model = _build_mlp_model(
            n_features=n_features,
            hidden_sizes=[hidden_size, hidden_size],
            dropout=0.0,
            device="cpu",
        )

        from platform_ml.torch_types import _import_torch

        torch = _import_torch()
        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), str(model_path))

        # Create metadata JSON
        meta = dump_json_str(
            {
                "backend": "mlp",
                "n_features": n_features,
                "hidden_sizes": [hidden_size, hidden_size],
                "dropout": 0.0,
            }
        )
        meta_path = tmp_path / "model.meta.json"
        meta_path.write_text(meta)

        result = _load_model(model_path, "mlp")
        x_test: NDArray[np.float64] = np.zeros((2, n_features), dtype=np.float64)
        proba = result.predict_proba(x_test)
        assert proba.shape == (2, 2)


# =============================================================================
# Tests for _create_worker
# =============================================================================


class TestCreateWorker:
    """Tests for _create_worker function."""

    def test_creates_streaming_worker(self) -> None:
        """Test _create_worker creates StreamingWorker from deps."""
        from covenant_radar_api.integrations.datadog.metrics import MetricsClient
        from covenant_radar_api.streaming._test_hooks import (
            FakeKafkaConsumer,
            FakeKafkaProducer,
        )
        from covenant_radar_api.streaming._test_hooks_model import (
            FakeMetricsSink,
            FakePredictor,
        )
        from covenant_radar_api.streaming.consumer import StreamingConsumer
        from covenant_radar_api.streaming.producer import StreamingProducer
        from covenant_radar_api.streaming.worker import make_default_worker_config

        fake_consumer = FakeKafkaConsumer()
        fake_producer = FakeKafkaProducer()

        deps: StreamingWorkerDeps = {
            "consumer": StreamingConsumer(fake_consumer, "test.measurements.v1"),
            "producer": StreamingProducer(
                fake_producer, "test.predictions.v1", "test.alerts.v1", "test.dlq.v1"
            ),
            "metrics": MetricsClient(FakeMetricsSink()),
            "model": FakePredictor(),
            "deal_repo": FakeDealRepository(),
            "covenant_repo": FakeCovenantRepository(),
            "measurement_repo": FakeMeasurementRepository(),
            "result_repo": FakeCovenantResultRepository(),
            "sector_encoder": {"Technology": 0},
            "region_encoder": {"North America": 0},
            "config": make_default_worker_config(),
            "db_conn": FakeConnection(),
        }

        worker = _create_worker(deps)
        assert worker.__class__.__name__ == "StreamingWorker"
        assert worker.is_running is False


# =============================================================================
# Tests for _run_worker
# =============================================================================


class TestRunWorker:
    """Tests for _run_worker function."""

    def test_run_worker_returns_zero(self) -> None:
        """Test _run_worker runs worker and returns exit code 0."""
        import signal as sig

        from covenant_radar_api.integrations.datadog.metrics import MetricsClient
        from covenant_radar_api.streaming._test_hooks import (
            FakeKafkaConsumer,
            FakeKafkaProducer,
        )
        from covenant_radar_api.streaming._test_hooks_model import (
            FakeMetricsSink,
            FakePredictor,
        )
        from covenant_radar_api.streaming.consumer import StreamingConsumer
        from covenant_radar_api.streaming.producer import StreamingProducer
        from covenant_radar_api.streaming.worker import StreamingWorker, make_default_worker_config

        fake_consumer = FakeKafkaConsumer()
        fake_producer = FakeKafkaProducer()

        worker = StreamingWorker(
            consumer=StreamingConsumer(fake_consumer, "test.measurements.v1"),
            producer=StreamingProducer(
                fake_producer, "test.predictions.v1", "test.alerts.v1", "test.dlq.v1"
            ),
            metrics=MetricsClient(FakeMetricsSink()),
            model=FakePredictor(),
            deal_repo=FakeDealRepository(),
            covenant_repo=FakeCovenantRepository(),
            measurement_repo=FakeMeasurementRepository(),
            result_repo=FakeCovenantResultRepository(),
            sector_encoder={"Technology": 0},
            region_encoder={"North America": 0},
            config=make_default_worker_config(),
        )

        # Shut down after first poll so _run_worker returns
        fake_consumer.set_on_poll(lambda: worker.shutdown())

        # Save and restore signal handlers
        orig_sigint = sig.getsignal(sig.SIGINT)
        orig_sigterm = sig.getsignal(sig.SIGTERM)

        logger = _RecordingLogger()
        try:
            exit_code = _run_worker(worker, logger)
        finally:
            sig.signal(sig.SIGINT, orig_sigint)
            sig.signal(sig.SIGTERM, orig_sigterm)

        assert exit_code == 0
        # Verify logging happened
        assert len(logger.info_messages) >= 2
        assert "Starting" in logger.info_messages[0][0]
        assert "stopped" in logger.info_messages[1][0].lower()


# =============================================================================
# Tests for main() happy path
# =============================================================================


class TestMainHappyPath:
    """Tests for main function happy path (enabled + deps provided)."""

    def test_main_runs_worker_and_returns_zero(
        self,
        restore_hooks: None,
    ) -> None:
        """Test main runs worker with provided deps and returns 0."""
        import signal as sig

        from covenant_radar_api.integrations.datadog.metrics import MetricsClient
        from covenant_radar_api.streaming._test_hooks import (
            FakeKafkaConsumer,
            FakeKafkaProducer,
        )
        from covenant_radar_api.streaming._test_hooks_model import (
            FakeMetricsSink,
            FakePredictor,
        )
        from covenant_radar_api.streaming.consumer import StreamingConsumer
        from covenant_radar_api.streaming.producer import StreamingProducer
        from covenant_radar_api.streaming.worker import make_default_worker_config

        fake_consumer = FakeKafkaConsumer()
        fake_producer = FakeKafkaProducer()
        fake_conn = FakeConnection()

        deps: StreamingWorkerDeps = {
            "consumer": StreamingConsumer(fake_consumer, "test.measurements.v1"),
            "producer": StreamingProducer(
                fake_producer, "test.predictions.v1", "test.alerts.v1", "test.dlq.v1"
            ),
            "metrics": MetricsClient(FakeMetricsSink()),
            "model": FakePredictor(),
            "deal_repo": FakeDealRepository(),
            "covenant_repo": FakeCovenantRepository(),
            "measurement_repo": FakeMeasurementRepository(),
            "result_repo": FakeCovenantResultRepository(),
            "sector_encoder": {"Technology": 0},
            "region_encoder": {"North America": 0},
            "config": make_default_worker_config(),
            "db_conn": fake_conn,
        }

        # Worker will be created inside main() and will poll the consumer.
        # On first poll, shut down via SIGINT which triggers the signal handler.
        poll_count: list[int] = [0]

        def on_poll() -> None:
            poll_count[0] += 1
            if poll_count[0] >= 2:
                sig.raise_signal(sig.SIGINT)

        fake_consumer.set_on_poll(on_poll)

        config = _make_test_streaming_config(enabled=True)
        logger = _RecordingLogger()

        # Save and restore signal handlers
        orig_sigint = sig.getsignal(sig.SIGINT)
        orig_sigterm = sig.getsignal(sig.SIGTERM)

        try:
            exit_code = main(
                streaming_config=config,
                deps=deps,
                logger=logger,
            )
        finally:
            sig.signal(sig.SIGINT, orig_sigint)
            sig.signal(sig.SIGTERM, orig_sigterm)

        assert exit_code == 0
        # Verify db connection was closed
        assert fake_conn.closed is True


# =============================================================================
# Tests for _build_dependencies
# =============================================================================


class TestBuildDependencies:
    """Tests for _build_dependencies function."""

    def test_builds_all_dependencies(
        self,
        tmp_path: Path,
        restore_hooks: None,
    ) -> None:
        """Test _build_dependencies assembles all deps from hooks and env."""
        from covenant_radar_api.integrations.datadog import _test_hooks as dd_hooks
        from covenant_radar_api.streaming import _test_hooks as streaming_hooks
        from covenant_radar_api.streaming._test_hooks import (
            FakeKafkaConsumer,
            FakeKafkaProducer,
        )
        from covenant_radar_api.streaming._test_hooks_model import (
            FakeMetricsSink,
            FakePredictor,
        )
        from covenant_radar_api.streaming_worker_entry import _build_dependencies

        # Create a dummy xgboost model file
        model_file = tmp_path / "model.json"
        model_file.write_text("{}")

        # Set up env vars
        env = make_fake_env()
        env.set("MODEL_PATH", str(model_file))
        env.set("MODEL_TYPE", "xgboost")
        env.set("DATABASE_URL", "postgresql://test:5432/testdb")

        # Override xgboost loader hook
        fake_model = FakePredictor()

        class _FakeXgbLoader:
            """Fake XGBoost model loader."""

            def __call__(self, model_path: str) -> PredictorProtocol:
                return fake_model

        _hooks.xgboost_loader = _FakeXgbLoader()

        # Override connection factory hook
        fake_conn = FakeConnection()
        _hooks.connection_factory = lambda url: fake_conn

        # Override repository factory hook
        fake_deal = FakeDealRepository()
        fake_cov = FakeCovenantRepository()
        fake_meas = FakeMeasurementRepository()
        fake_result = FakeCovenantResultRepository()

        def fake_repo_factory(
            conn: ConnectionProtocol,
        ) -> tuple[
            FakeDealRepository,
            FakeCovenantRepository,
            FakeMeasurementRepository,
            FakeCovenantResultRepository,
        ]:
            return fake_deal, fake_cov, fake_meas, fake_result

        _hooks.repository_factory = fake_repo_factory

        # Override streaming hooks for consumer/producer factories
        fake_kafka_consumer = FakeKafkaConsumer()
        fake_kafka_producer = FakeKafkaProducer()

        orig_consumer_factory = streaming_hooks.consumer_factory
        orig_producer_factory = streaming_hooks.producer_factory

        from covenant_radar_api.streaming._test_hooks import (
            KafkaConsumerProtocol,
            KafkaProducerProtocol,
        )
        from covenant_radar_api.streaming.config import (
            ConfluentConfig,
            ConsumerConfig,
            ProducerConfig,
        )

        def fake_consumer_factory(
            confluent_config: ConfluentConfig,
            consumer_config: ConsumerConfig,
        ) -> KafkaConsumerProtocol:
            return fake_kafka_consumer

        def fake_producer_factory(
            confluent_config: ConfluentConfig,
            producer_config: ProducerConfig,
        ) -> KafkaProducerProtocol:
            return fake_kafka_producer

        streaming_hooks.consumer_factory = fake_consumer_factory
        streaming_hooks.producer_factory = fake_producer_factory

        # Override metrics sink factory
        orig_metrics_sink_factory = dd_hooks.metrics_sink_factory

        def fake_metrics_sink_factory(
            host: str,
            port: int,
            namespace: str,
        ) -> FakeMetricsSink:
            return FakeMetricsSink()

        dd_hooks.metrics_sink_factory = fake_metrics_sink_factory

        streaming_config = _make_test_streaming_config(enabled=True)

        try:
            deps = _build_dependencies(streaming_config)

            assert deps["model"] is fake_model
            assert deps["deal_repo"] is fake_deal
            assert deps["covenant_repo"] is fake_cov
            assert deps["measurement_repo"] is fake_meas
            assert deps["result_repo"] is fake_result
            assert deps["db_conn"] is fake_conn
            assert "Technology" in deps["sector_encoder"]
            assert "North America" in deps["region_encoder"]
        finally:
            streaming_hooks.consumer_factory = orig_consumer_factory
            streaming_hooks.producer_factory = orig_producer_factory
            dd_hooks.metrics_sink_factory = orig_metrics_sink_factory


# =============================================================================
# Tests for real hook implementations in streaming_worker_entry_hooks
# =============================================================================


class TestRealRepositoryFactory:
    """Tests for _real_repository_factory function."""

    def test_creates_postgres_repositories(self) -> None:
        """Test _real_repository_factory creates PostgreSQL repositories."""
        from covenant_persistence.testing import InMemoryConnection, InMemoryStore

        from covenant_radar_api.streaming_worker_entry_hooks import (
            _real_repository_factory,
        )

        conn = InMemoryConnection(InMemoryStore())
        deal_repo, covenant_repo, measurement_repo, result_repo = _real_repository_factory(conn)

        assert deal_repo.__class__.__name__ == "PostgresDealRepository"
        assert covenant_repo.__class__.__name__ == "PostgresCovenantRepository"
        assert measurement_repo.__class__.__name__ == "PostgresMeasurementRepository"
        assert result_repo.__class__.__name__ == "PostgresCovenantResultRepository"


class TestRealXgboostLoader:
    """Tests for _real_xgboost_loader function."""

    def test_loads_xgboost_model(self, tmp_path: Path) -> None:
        """Test _real_xgboost_loader loads real XGBoost model."""
        import numpy as np
        from covenant_ml.testing import make_train_config
        from covenant_ml.trainer import save_model, train_model
        from numpy.typing import NDArray

        from covenant_radar_api.streaming_worker_entry_hooks import (
            _real_xgboost_loader,
        )

        x: NDArray[np.float64] = np.zeros((4, 2), dtype=np.float64)
        x[0, 0] = 1.0
        x[1, 0] = 2.0
        x[2, 0] = 3.0
        x[3, 0] = 4.0
        y: NDArray[np.int64] = np.zeros(4, dtype=np.int64)
        y[2] = 1
        y[3] = 1
        config = make_train_config(
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=1.0,
            reg_lambda=5.0,
        )
        model = train_model(x, y, config)
        model_path = tmp_path / "model.ubj"
        save_model(model, str(model_path))

        loaded = _real_xgboost_loader(str(model_path))
        proba = loaded.predict_proba(x)
        assert proba.shape == (4, 2)


class TestFakeConnectionFactory:
    """Tests for _fake_connection_factory function."""

    def test_returns_fake_connection(self) -> None:
        """Test _fake_connection_factory returns FakeConnection."""
        conn = _fake_connection_factory("postgresql://test:5432/db")
        assert conn.__class__.__name__ == "FakeConnection"


class TestRealConnectionFactory:
    """Tests for _real_connection_factory function."""

    def test_raises_on_invalid_url(self) -> None:
        """Test _real_connection_factory raises with invalid connection string."""
        from covenant_radar_api.streaming_worker_entry_hooks import (
            _real_connection_factory,
        )

        psycopg_mod = __import__("psycopg")
        error_cls: type = psycopg_mod.ProgrammingError

        with pytest.raises(error_cls):
            _real_connection_factory("invalid://not-a-real-database")


# =============================================================================
# Tests for __main__ guard
# =============================================================================


class TestMainGuard:
    """Tests for if __name__ == '__main__' block."""

    def test_main_guard_executes_main(self) -> None:
        """Test the if __name__ == '__main__' guard calls sys.exit(main()).

        Uses runpy.run_module to execute the module as __main__.
        STREAMING__ENABLED defaults to false so main() returns 1 immediately.
        """
        import runpy
        import sys

        # make_fake_env ensures all env vars are controlled;
        # STREAMING__ENABLED defaults to false in load_streaming_config
        _ = make_fake_env()

        module_name = "covenant_radar_api.streaming_worker_entry"
        saved_module = sys.modules.pop(module_name, None)

        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module(
                module_name,
                run_name="__main__",
                alter_sys=False,
            )

        if saved_module is not None:
            sys.modules[module_name] = saved_module

        assert exc_info.value.code == 1
