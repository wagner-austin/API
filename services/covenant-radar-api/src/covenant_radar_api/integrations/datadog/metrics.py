"""Datadog custom metrics via DogStatsD.

This module provides a typed wrapper around DogStatsD for emitting
custom application metrics to Datadog.

Metric types:
- Counter: Cumulative count (e.g., measurements received)
- Gauge: Current value (e.g., consumer lag)
- Histogram: Distribution (e.g., prediction latency)

Usage:
    from covenant_radar_api.integrations.datadog import (
        create_metrics_client,
        MetricsConfig,
    )

    config: MetricsConfig = {
        "host": "localhost",
        "port": 8125,
        "namespace": "covenant",
    }
    client = create_metrics_client(config)
    client.increment_measurement_received("deal-123", "debt_to_equity")
"""

from __future__ import annotations

from typing import Literal, TypedDict

from . import _test_hooks


class MetricsConfig(TypedDict, total=True):
    """DogStatsD configuration.

    Fields:
        host: DogStatsD agent host.
        port: DogStatsD agent port.
        namespace: Metric namespace prefix (prepended to all metrics).
    """

    host: str
    port: int
    namespace: str


# Standard metric names as constants for type safety
METRIC_MEASUREMENT_RECEIVED = "measurement.received"
METRIC_EVALUATION_LATENCY_MS = "evaluation.latency_ms"
METRIC_PREDICTION_LATENCY_MS = "prediction.latency_ms"
METRIC_PREDICTION_RISK = "prediction.risk_probability"
METRIC_ALERT_TRIGGERED = "alert.triggered"
METRIC_STREAM_LAG_MESSAGES = "stream.lag_messages"
METRIC_GEMINI_LATENCY_MS = "gemini.latency_ms"
METRIC_GEMINI_TOKENS = "gemini.tokens"


class MetricsClient:
    """Typed client for emitting Datadog custom metrics.

    This client wraps DogStatsD with typed methods for each metric
    we emit in the covenant-radar-api service.

    All methods are fire-and-forget (UDP) and do not block.
    """

    def __init__(self, sink: _test_hooks.MetricsSinkProtocol) -> None:
        """Initialize the metrics client.

        Args:
            sink: Underlying metrics sink (DogStatsD or fake).
        """
        self._sink = sink

    # =========================================================================
    # Counter Metrics
    # =========================================================================

    def increment_measurement_received(
        self,
        deal_id: str,
        metric_name: str,
    ) -> None:
        """Record a measurement received event.

        Args:
            deal_id: Deal identifier.
            metric_name: Name of the metric received (e.g., "debt_to_equity").
        """
        self._sink.increment(
            METRIC_MEASUREMENT_RECEIVED,
            1,
            (f"deal_id:{deal_id}", f"metric_name:{metric_name}"),
        )

    def increment_alert_triggered(
        self,
        deal_id: str,
        severity: Literal["warning", "critical"],
        alert_type: Literal["breach", "high_risk"],
    ) -> None:
        """Record an alert triggered event.

        Args:
            deal_id: Deal identifier.
            severity: Alert severity level.
            alert_type: Type of alert.
        """
        self._sink.increment(
            METRIC_ALERT_TRIGGERED,
            1,
            (
                f"deal_id:{deal_id}",
                f"severity:{severity}",
                f"alert_type:{alert_type}",
            ),
        )

    def increment_gemini_tokens(
        self,
        model: str,
        direction: Literal["input", "output"],
        count: int,
    ) -> None:
        """Record Gemini token usage.

        Args:
            model: Gemini model name.
            direction: Token direction (input or output).
            count: Number of tokens.
        """
        self._sink.increment(
            METRIC_GEMINI_TOKENS,
            count,
            (f"model:{model}", f"direction:{direction}"),
        )

    # =========================================================================
    # Gauge Metrics
    # =========================================================================

    def set_prediction_risk(
        self,
        deal_id: str,
        risk_probability: float,
    ) -> None:
        """Set current risk probability for a deal.

        Args:
            deal_id: Deal identifier.
            risk_probability: Risk probability (0.0-1.0).
        """
        self._sink.gauge(
            METRIC_PREDICTION_RISK,
            risk_probability,
            (f"deal_id:{deal_id}",),
        )

    def set_stream_lag_messages(
        self,
        topic: str,
        partition: int,
        lag: int,
    ) -> None:
        """Set consumer lag for a topic partition.

        Args:
            topic: Kafka topic name.
            partition: Partition number.
            lag: Number of messages behind.
        """
        self._sink.gauge(
            METRIC_STREAM_LAG_MESSAGES,
            float(lag),
            (f"topic:{topic}", f"partition:{partition}"),
        )

    # =========================================================================
    # Histogram Metrics
    # =========================================================================

    def record_evaluation_latency(
        self,
        deal_id: str,
        status: Literal["OK", "BREACH", "WARNING"],
        latency_ms: float,
    ) -> None:
        """Record covenant evaluation latency.

        Args:
            deal_id: Deal identifier.
            status: Evaluation status.
            latency_ms: Evaluation time in milliseconds.
        """
        self._sink.histogram(
            METRIC_EVALUATION_LATENCY_MS,
            latency_ms,
            (f"deal_id:{deal_id}", f"status:{status}"),
        )

    def record_prediction_latency(
        self,
        deal_id: str,
        risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        latency_ms: float,
    ) -> None:
        """Record ML prediction latency.

        Args:
            deal_id: Deal identifier.
            risk_tier: Risk tier classification.
            latency_ms: Prediction time in milliseconds.
        """
        self._sink.histogram(
            METRIC_PREDICTION_LATENCY_MS,
            latency_ms,
            (f"deal_id:{deal_id}", f"risk_tier:{risk_tier}"),
        )

    def record_gemini_latency(
        self,
        model: str,
        latency_ms: float,
    ) -> None:
        """Record Gemini API call latency.

        Args:
            model: Gemini model name.
            latency_ms: API call time in milliseconds.
        """
        self._sink.histogram(
            METRIC_GEMINI_LATENCY_MS,
            latency_ms,
            (f"model:{model}",),
        )


def create_metrics_client(config: MetricsConfig) -> MetricsClient:
    """Create a metrics client with the given configuration.

    Uses the injectable metrics_sink_factory from _test_hooks.
    Production code uses RealMetricsSink; tests inject FakeMetricsSink.

    Args:
        config: DogStatsD configuration.

    Returns:
        MetricsClient instance.
    """
    sink = _test_hooks.metrics_sink_factory(
        config["host"],
        config["port"],
        config["namespace"],
    )
    return MetricsClient(sink)


def make_default_metrics_config() -> MetricsConfig:
    """Create default metrics configuration.

    Returns:
        MetricsConfig with sensible defaults.
    """
    return {
        "host": "localhost",
        "port": 8125,
        "namespace": "covenant",
    }


__all__ = [
    "METRIC_ALERT_TRIGGERED",
    "METRIC_EVALUATION_LATENCY_MS",
    "METRIC_GEMINI_LATENCY_MS",
    "METRIC_GEMINI_TOKENS",
    "METRIC_MEASUREMENT_RECEIVED",
    "METRIC_PREDICTION_LATENCY_MS",
    "METRIC_PREDICTION_RISK",
    "METRIC_STREAM_LAG_MESSAGES",
    "MetricsClient",
    "MetricsConfig",
    "create_metrics_client",
    "make_default_metrics_config",
]
