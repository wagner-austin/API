"""Tests for Datadog integration test hooks."""

from __future__ import annotations

from covenant_radar_api.integrations.datadog._test_hooks import (
    MetricsSinkFactory,
    MetricsSinkProtocol,
    RealMetricsSink,
    TracingSetupProtocol,
    metrics_sink_factory,
    tracing_setup,
)


class TestMetricsSinkProtocol:
    """Tests for MetricsSinkProtocol."""

    def test_protocol_defines_increment(self) -> None:
        """Test that protocol defines increment method."""

        class FakeSink:
            def increment(
                self,
                metric: str,
                value: int,
                tags: tuple[str, ...],
            ) -> None:
                pass

            def gauge(
                self,
                metric: str,
                value: float,
                tags: tuple[str, ...],
            ) -> None:
                pass

            def histogram(
                self,
                metric: str,
                value: float,
                tags: tuple[str, ...],
            ) -> None:
                pass

        sink: MetricsSinkProtocol = FakeSink()
        sink.increment("test.metric", 1, ("tag:value",))

    def test_protocol_defines_gauge(self) -> None:
        """Test that protocol defines gauge method."""

        class FakeSink:
            def increment(
                self,
                metric: str,
                value: int,
                tags: tuple[str, ...],
            ) -> None:
                pass

            def gauge(
                self,
                metric: str,
                value: float,
                tags: tuple[str, ...],
            ) -> None:
                pass

            def histogram(
                self,
                metric: str,
                value: float,
                tags: tuple[str, ...],
            ) -> None:
                pass

        sink: MetricsSinkProtocol = FakeSink()
        sink.gauge("test.metric", 1.0, ("tag:value",))

    def test_protocol_defines_histogram(self) -> None:
        """Test that protocol defines histogram method."""

        class FakeSink:
            def increment(
                self,
                metric: str,
                value: int,
                tags: tuple[str, ...],
            ) -> None:
                pass

            def gauge(
                self,
                metric: str,
                value: float,
                tags: tuple[str, ...],
            ) -> None:
                pass

            def histogram(
                self,
                metric: str,
                value: float,
                tags: tuple[str, ...],
            ) -> None:
                pass

        sink: MetricsSinkProtocol = FakeSink()
        sink.histogram("test.metric", 1.0, ("tag:value",))


class TestMetricsSinkFactory:
    """Tests for MetricsSinkFactory protocol."""

    def test_factory_protocol(self) -> None:
        """Test factory protocol signature."""

        def fake_factory(
            host: str,
            port: int,
            namespace: str,
        ) -> MetricsSinkProtocol:
            class FakeSink:
                def increment(
                    self,
                    metric: str,
                    value: int,
                    tags: tuple[str, ...],
                ) -> None:
                    pass

                def gauge(
                    self,
                    metric: str,
                    value: float,
                    tags: tuple[str, ...],
                ) -> None:
                    pass

                def histogram(
                    self,
                    metric: str,
                    value: float,
                    tags: tuple[str, ...],
                ) -> None:
                    pass

            return FakeSink()

        factory: MetricsSinkFactory = fake_factory
        sink = factory("localhost", 8125, "test")
        # Verify sink implements protocol by calling a method
        sink.increment("test.metric", 1, ("tag:value",))


class TestRealMetricsSink:
    """Tests for RealMetricsSink class."""

    def test_init_stores_config(self) -> None:
        """Test that init stores configuration."""
        sink = RealMetricsSink("my-host", 9999, "myapp")

        assert sink._host == "my-host"
        assert sink._port == 9999
        assert sink._namespace == "myapp"

    def test_increment_calls_statsd(self) -> None:
        """Test increment forwards to DogStatsd client."""
        # Use localhost which won't actually connect but exercises the code
        sink = RealMetricsSink("localhost", 8125, "test")
        # This sends a UDP packet which is fire-and-forget
        sink.increment("test.counter", 1, ("tag:value",))

    def test_gauge_calls_statsd(self) -> None:
        """Test gauge forwards to DogStatsd client."""
        sink = RealMetricsSink("localhost", 8125, "test")
        sink.gauge("test.gauge", 42.5, ("env:test",))

    def test_histogram_calls_statsd(self) -> None:
        """Test histogram forwards to DogStatsd client."""
        sink = RealMetricsSink("localhost", 8125, "test")
        sink.histogram("test.histogram", 100.0, ("service:api",))


class TestTracingSetupProtocol:
    """Tests for TracingSetupProtocol."""

    def test_protocol_signature(self) -> None:
        """Test protocol has correct signature."""

        def fake_setup(service: str, env: str, version: str) -> bool:
            return True

        setup: TracingSetupProtocol = fake_setup
        result = setup("my-service", "dev", "1.0.0")
        assert result is True


class TestDefaultHooks:
    """Tests for default hook values."""

    def test_metrics_sink_factory_is_callable(self) -> None:
        """Test that default metrics_sink_factory is callable."""
        assert callable(metrics_sink_factory)

    def test_tracing_setup_is_callable(self) -> None:
        """Test that default tracing_setup is callable."""
        assert callable(tracing_setup)

    def test_metrics_sink_factory_creates_working_sink(self) -> None:
        """Test that default factory creates a working sink."""
        sink = metrics_sink_factory("localhost", 8125, "test")

        # Verify sink works by calling methods (fire-and-forget UDP)
        sink.increment("factory.test.counter", 1, ("source:factory_test",))
        sink.gauge("factory.test.gauge", 42.0, ("source:factory_test",))
        sink.histogram("factory.test.histogram", 100.0, ("source:factory_test",))

    def test_tracing_setup_configures_ddtrace(self) -> None:
        """Test that default tracing_setup configures ddtrace."""
        result = tracing_setup("test-service", "test-env", "1.0.0")

        assert result is True
