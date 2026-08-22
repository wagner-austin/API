"""Tests for streaming worker entry point."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from covenant_persistence.protocols import ConnectionProtocol
from platform_core.testing import make_fake_env

from covenant_radar_api import streaming_worker_entry_hooks as _hooks
from covenant_radar_api.streaming._test_hooks_repositories import (
    FakeCovenantRepository,
    FakeCovenantResultRepository,
    FakeDealRepository,
    FakeMeasurementRepository,
)
from covenant_radar_api.streaming_worker_entry import (
    ModelType,
    StreamingWorkerDeps,
    _create_connection,
    _create_repositories,
    _load_encoders,
    _load_metrics_config,
    _parse_model_type,
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
from tests._worker_entry_fixtures import (
    _make_test_streaming_config,
    _RecordingLogger,
)


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
