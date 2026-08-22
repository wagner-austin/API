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
from covenant_radar_api.streaming_worker_entry import (
    StreamingWorkerDeps,
    main,
)
from covenant_radar_api.streaming_worker_entry_hooks import (
    FakeConnection,
    _fake_connection_factory,
)
from tests._worker_entry_fixtures import (
    _make_test_streaming_config,
    _RecordingLogger,
)


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
        from covenant_radar_api.streaming.worker_events import make_default_worker_config

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
        from covenant_ml.trainer_fit import (
            save_model,
            train_model,
        )
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
