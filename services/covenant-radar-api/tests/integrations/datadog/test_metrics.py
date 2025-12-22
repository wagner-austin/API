"""Tests for Datadog metrics module."""

from __future__ import annotations

from covenant_radar_api.integrations.datadog import _test_hooks
from covenant_radar_api.integrations.datadog.metrics import (
    METRIC_ALERT_TRIGGERED,
    METRIC_EVALUATION_LATENCY_MS,
    METRIC_GEMINI_LATENCY_MS,
    METRIC_GEMINI_TOKENS,
    METRIC_MEASUREMENT_RECEIVED,
    METRIC_PREDICTION_LATENCY_MS,
    METRIC_PREDICTION_RISK,
    METRIC_STREAM_LAG_MESSAGES,
    MetricsConfig,
    create_metrics_client,
    make_default_metrics_config,
)

from .conftest import FakeMetricsSink


class TestCreateMetricsClient:
    """Tests for create_metrics_client function."""

    def test_creates_client_with_config(
        self,
        fake_metrics_sink_factory: tuple[
            _test_hooks.MetricsSinkFactory,
            list[FakeMetricsSink],
        ],
    ) -> None:
        """Test that client is created with provided config."""
        factory, created_sinks = fake_metrics_sink_factory
        _test_hooks.metrics_sink_factory = factory

        config: MetricsConfig = {
            "host": "my-host",
            "port": 9999,
            "namespace": "myapp",
        }
        client = create_metrics_client(config)

        assert len(created_sinks) == 1
        assert created_sinks[0].host == "my-host"
        assert created_sinks[0].port == 9999
        assert created_sinks[0].namespace == "myapp"
        # Verify client can call methods (proves correct type)
        client.increment_measurement_received("test", "test")
        assert len(created_sinks[0].calls) == 1

    def test_uses_hook_factory(
        self,
        fake_metrics_sink_factory: tuple[
            _test_hooks.MetricsSinkFactory,
            list[FakeMetricsSink],
        ],
    ) -> None:
        """Test that hook factory is used for sink creation."""
        factory, created_sinks = fake_metrics_sink_factory
        _test_hooks.metrics_sink_factory = factory

        config: MetricsConfig = {
            "host": "localhost",
            "port": 8125,
            "namespace": "test",
        }
        create_metrics_client(config)

        assert len(created_sinks) == 1


class TestMetricsClientCounters:
    """Tests for MetricsClient counter methods."""

    def test_increment_measurement_received(
        self,
        fake_metrics_sink_factory: tuple[
            _test_hooks.MetricsSinkFactory,
            list[FakeMetricsSink],
        ],
    ) -> None:
        """Test increment_measurement_received records correct metric."""
        factory, created_sinks = fake_metrics_sink_factory
        _test_hooks.metrics_sink_factory = factory

        config: MetricsConfig = {"host": "localhost", "port": 8125, "namespace": "covenant"}
        client = create_metrics_client(config)

        client.increment_measurement_received("deal-123", "debt_to_equity")

        assert len(created_sinks[0].calls) == 1
        call = created_sinks[0].calls[0]
        assert call["method"] == "increment"
        assert call["metric"] == f"covenant.{METRIC_MEASUREMENT_RECEIVED}"
        assert call["value"] == 1.0
        assert call["tags"] == ("deal_id:deal-123", "metric_name:debt_to_equity")

    def test_increment_alert_triggered(
        self,
        fake_metrics_sink_factory: tuple[
            _test_hooks.MetricsSinkFactory,
            list[FakeMetricsSink],
        ],
    ) -> None:
        """Test increment_alert_triggered records correct metric."""
        factory, created_sinks = fake_metrics_sink_factory
        _test_hooks.metrics_sink_factory = factory

        config: MetricsConfig = {"host": "localhost", "port": 8125, "namespace": "covenant"}
        client = create_metrics_client(config)

        client.increment_alert_triggered("deal-456", "critical", "breach")

        assert len(created_sinks[0].calls) == 1
        call = created_sinks[0].calls[0]
        assert call["method"] == "increment"
        assert call["metric"] == f"covenant.{METRIC_ALERT_TRIGGERED}"
        assert call["value"] == 1.0
        assert call["tags"] == (
            "deal_id:deal-456",
            "severity:critical",
            "alert_type:breach",
        )

    def test_increment_alert_triggered_warning_high_risk(
        self,
        fake_metrics_sink_factory: tuple[
            _test_hooks.MetricsSinkFactory,
            list[FakeMetricsSink],
        ],
    ) -> None:
        """Test increment_alert_triggered with warning severity and high_risk type."""
        factory, created_sinks = fake_metrics_sink_factory
        _test_hooks.metrics_sink_factory = factory

        config: MetricsConfig = {"host": "localhost", "port": 8125, "namespace": "covenant"}
        client = create_metrics_client(config)

        client.increment_alert_triggered("deal-789", "warning", "high_risk")

        call = created_sinks[0].calls[0]
        assert call["tags"] == (
            "deal_id:deal-789",
            "severity:warning",
            "alert_type:high_risk",
        )

    def test_increment_gemini_tokens(
        self,
        fake_metrics_sink_factory: tuple[
            _test_hooks.MetricsSinkFactory,
            list[FakeMetricsSink],
        ],
    ) -> None:
        """Test increment_gemini_tokens records correct metric."""
        factory, created_sinks = fake_metrics_sink_factory
        _test_hooks.metrics_sink_factory = factory

        config: MetricsConfig = {"host": "localhost", "port": 8125, "namespace": "covenant"}
        client = create_metrics_client(config)

        client.increment_gemini_tokens("gemini-1.5-flash", "input", 150)

        call = created_sinks[0].calls[0]
        assert call["method"] == "increment"
        assert call["metric"] == f"covenant.{METRIC_GEMINI_TOKENS}"
        assert call["value"] == 150.0
        assert call["tags"] == ("model:gemini-1.5-flash", "direction:input")

    def test_increment_gemini_tokens_output(
        self,
        fake_metrics_sink_factory: tuple[
            _test_hooks.MetricsSinkFactory,
            list[FakeMetricsSink],
        ],
    ) -> None:
        """Test increment_gemini_tokens with output direction."""
        factory, created_sinks = fake_metrics_sink_factory
        _test_hooks.metrics_sink_factory = factory

        config: MetricsConfig = {"host": "localhost", "port": 8125, "namespace": "covenant"}
        client = create_metrics_client(config)

        client.increment_gemini_tokens("gemini-1.5-pro", "output", 300)

        call = created_sinks[0].calls[0]
        assert call["tags"] == ("model:gemini-1.5-pro", "direction:output")
        assert call["value"] == 300.0


class TestMetricsClientGauges:
    """Tests for MetricsClient gauge methods."""

    def test_set_prediction_risk(
        self,
        fake_metrics_sink_factory: tuple[
            _test_hooks.MetricsSinkFactory,
            list[FakeMetricsSink],
        ],
    ) -> None:
        """Test set_prediction_risk records correct metric."""
        factory, created_sinks = fake_metrics_sink_factory
        _test_hooks.metrics_sink_factory = factory

        config: MetricsConfig = {"host": "localhost", "port": 8125, "namespace": "covenant"}
        client = create_metrics_client(config)

        client.set_prediction_risk("deal-001", 0.85)

        call = created_sinks[0].calls[0]
        assert call["method"] == "gauge"
        assert call["metric"] == f"covenant.{METRIC_PREDICTION_RISK}"
        assert call["value"] == 0.85
        assert call["tags"] == ("deal_id:deal-001",)

    def test_set_stream_lag_messages(
        self,
        fake_metrics_sink_factory: tuple[
            _test_hooks.MetricsSinkFactory,
            list[FakeMetricsSink],
        ],
    ) -> None:
        """Test set_stream_lag_messages records correct metric."""
        factory, created_sinks = fake_metrics_sink_factory
        _test_hooks.metrics_sink_factory = factory

        config: MetricsConfig = {"host": "localhost", "port": 8125, "namespace": "covenant"}
        client = create_metrics_client(config)

        client.set_stream_lag_messages("covenant.measurements.v1", 3, 1500)

        call = created_sinks[0].calls[0]
        assert call["method"] == "gauge"
        assert call["metric"] == f"covenant.{METRIC_STREAM_LAG_MESSAGES}"
        assert call["value"] == 1500.0
        assert call["tags"] == ("topic:covenant.measurements.v1", "partition:3")


class TestMetricsClientHistograms:
    """Tests for MetricsClient histogram methods."""

    def test_record_evaluation_latency(
        self,
        fake_metrics_sink_factory: tuple[
            _test_hooks.MetricsSinkFactory,
            list[FakeMetricsSink],
        ],
    ) -> None:
        """Test record_evaluation_latency records correct metric."""
        factory, created_sinks = fake_metrics_sink_factory
        _test_hooks.metrics_sink_factory = factory

        config: MetricsConfig = {"host": "localhost", "port": 8125, "namespace": "covenant"}
        client = create_metrics_client(config)

        client.record_evaluation_latency("deal-002", "OK", 45.5)

        call = created_sinks[0].calls[0]
        assert call["method"] == "histogram"
        assert call["metric"] == f"covenant.{METRIC_EVALUATION_LATENCY_MS}"
        assert call["value"] == 45.5
        assert call["tags"] == ("deal_id:deal-002", "status:OK")

    def test_record_evaluation_latency_breach(
        self,
        fake_metrics_sink_factory: tuple[
            _test_hooks.MetricsSinkFactory,
            list[FakeMetricsSink],
        ],
    ) -> None:
        """Test record_evaluation_latency with BREACH status."""
        factory, created_sinks = fake_metrics_sink_factory
        _test_hooks.metrics_sink_factory = factory

        config: MetricsConfig = {"host": "localhost", "port": 8125, "namespace": "covenant"}
        client = create_metrics_client(config)

        client.record_evaluation_latency("deal-003", "BREACH", 120.0)

        call = created_sinks[0].calls[0]
        assert call["tags"] == ("deal_id:deal-003", "status:BREACH")

    def test_record_evaluation_latency_warning(
        self,
        fake_metrics_sink_factory: tuple[
            _test_hooks.MetricsSinkFactory,
            list[FakeMetricsSink],
        ],
    ) -> None:
        """Test record_evaluation_latency with WARNING status."""
        factory, created_sinks = fake_metrics_sink_factory
        _test_hooks.metrics_sink_factory = factory

        config: MetricsConfig = {"host": "localhost", "port": 8125, "namespace": "covenant"}
        client = create_metrics_client(config)

        client.record_evaluation_latency("deal-004", "WARNING", 80.0)

        call = created_sinks[0].calls[0]
        assert call["tags"] == ("deal_id:deal-004", "status:WARNING")

    def test_record_prediction_latency(
        self,
        fake_metrics_sink_factory: tuple[
            _test_hooks.MetricsSinkFactory,
            list[FakeMetricsSink],
        ],
    ) -> None:
        """Test record_prediction_latency records correct metric."""
        factory, created_sinks = fake_metrics_sink_factory
        _test_hooks.metrics_sink_factory = factory

        config: MetricsConfig = {"host": "localhost", "port": 8125, "namespace": "covenant"}
        client = create_metrics_client(config)

        client.record_prediction_latency("deal-005", "HIGH", 25.0)

        call = created_sinks[0].calls[0]
        assert call["method"] == "histogram"
        assert call["metric"] == f"covenant.{METRIC_PREDICTION_LATENCY_MS}"
        assert call["value"] == 25.0
        assert call["tags"] == ("deal_id:deal-005", "risk_tier:HIGH")

    def test_record_prediction_latency_all_tiers(
        self,
        fake_metrics_sink_factory: tuple[
            _test_hooks.MetricsSinkFactory,
            list[FakeMetricsSink],
        ],
    ) -> None:
        """Test record_prediction_latency with all risk tiers."""
        factory, created_sinks = fake_metrics_sink_factory
        _test_hooks.metrics_sink_factory = factory

        config: MetricsConfig = {"host": "localhost", "port": 8125, "namespace": "covenant"}
        client = create_metrics_client(config)

        client.record_prediction_latency("deal-a", "LOW", 10.0)
        client.record_prediction_latency("deal-b", "MEDIUM", 20.0)
        client.record_prediction_latency("deal-c", "HIGH", 30.0)
        client.record_prediction_latency("deal-d", "CRITICAL", 40.0)

        assert len(created_sinks[0].calls) == 4
        assert created_sinks[0].calls[0]["tags"] == ("deal_id:deal-a", "risk_tier:LOW")
        assert created_sinks[0].calls[1]["tags"] == ("deal_id:deal-b", "risk_tier:MEDIUM")
        assert created_sinks[0].calls[2]["tags"] == ("deal_id:deal-c", "risk_tier:HIGH")
        assert created_sinks[0].calls[3]["tags"] == ("deal_id:deal-d", "risk_tier:CRITICAL")

    def test_record_gemini_latency(
        self,
        fake_metrics_sink_factory: tuple[
            _test_hooks.MetricsSinkFactory,
            list[FakeMetricsSink],
        ],
    ) -> None:
        """Test record_gemini_latency records correct metric."""
        factory, created_sinks = fake_metrics_sink_factory
        _test_hooks.metrics_sink_factory = factory

        config: MetricsConfig = {"host": "localhost", "port": 8125, "namespace": "covenant"}
        client = create_metrics_client(config)

        client.record_gemini_latency("gemini-1.5-flash", 350.0)

        call = created_sinks[0].calls[0]
        assert call["method"] == "histogram"
        assert call["metric"] == f"covenant.{METRIC_GEMINI_LATENCY_MS}"
        assert call["value"] == 350.0
        assert call["tags"] == ("model:gemini-1.5-flash",)


class TestMakeDefaultMetricsConfig:
    """Tests for make_default_metrics_config function."""

    def test_returns_valid_config(self) -> None:
        """Test that default config has all required fields."""
        config = make_default_metrics_config()

        assert config["host"] == "localhost"
        assert config["port"] == 8125
        assert config["namespace"] == "covenant"


class TestMetricConstants:
    """Tests for metric name constants."""

    def test_metric_constants_exist(self) -> None:
        """Test that all metric constants are defined."""
        assert METRIC_MEASUREMENT_RECEIVED == "measurement.received"
        assert METRIC_EVALUATION_LATENCY_MS == "evaluation.latency_ms"
        assert METRIC_PREDICTION_LATENCY_MS == "prediction.latency_ms"
        assert METRIC_PREDICTION_RISK == "prediction.risk_probability"
        assert METRIC_ALERT_TRIGGERED == "alert.triggered"
        assert METRIC_STREAM_LAG_MESSAGES == "stream.lag_messages"
        assert METRIC_GEMINI_LATENCY_MS == "gemini.latency_ms"
        assert METRIC_GEMINI_TOKENS == "gemini.tokens"
