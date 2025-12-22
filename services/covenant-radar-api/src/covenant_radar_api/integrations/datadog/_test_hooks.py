"""Test hooks for Datadog integration.

Production code uses real implementations; tests can override these module-level
symbols to inject fakes without conditionals in core logic.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from typing import Protocol

# =============================================================================
# Metrics Client Hook
# =============================================================================


class MetricsSinkProtocol(Protocol):
    """Protocol for metrics sink (DogStatsD or fake)."""

    def increment(
        self,
        metric: str,
        value: int,
        tags: tuple[str, ...],
    ) -> None:
        """Increment a counter metric.

        Args:
            metric: Metric name (e.g., "covenant.measurement.received").
            value: Amount to increment by.
            tags: Tuple of tag strings (e.g., ("deal_id:123", "status:ok")).
        """
        ...

    def gauge(
        self,
        metric: str,
        value: float,
        tags: tuple[str, ...],
    ) -> None:
        """Set a gauge metric.

        Args:
            metric: Metric name (e.g., "covenant.stream.lag_messages").
            value: Current value.
            tags: Tuple of tag strings.
        """
        ...

    def histogram(
        self,
        metric: str,
        value: float,
        tags: tuple[str, ...],
    ) -> None:
        """Record a histogram value.

        Args:
            metric: Metric name (e.g., "covenant.prediction.latency_ms").
            value: Value to record.
            tags: Tuple of tag strings.
        """
        ...


class MetricsSinkFactory(Protocol):
    """Protocol for creating metrics sinks."""

    def __call__(
        self,
        host: str,
        port: int,
        namespace: str,
    ) -> MetricsSinkProtocol:
        """Create a metrics sink.

        Args:
            host: DogStatsD agent host.
            port: DogStatsD agent port.
            namespace: Metric namespace prefix.

        Returns:
            MetricsSinkProtocol implementation.
        """
        ...


class RealMetricsSink:
    """Real metrics sink using the datadog package's DogStatsd client.

    This implementation uses the official datadog Python library's DogStatsd
    class to emit metrics via UDP to the Datadog agent.

    Reference: https://datadogpy.readthedocs.io/
    """

    def __init__(self, host: str, port: int, namespace: str) -> None:
        """Initialize the real metrics sink.

        Args:
            host: DogStatsD agent host.
            port: DogStatsD agent port.
            namespace: Metric namespace prefix.
        """
        from datadog.dogstatsd.base import DogStatsd

        self._host = host
        self._port = port
        self._namespace = namespace
        self._statsd = DogStatsd(
            host=host,
            port=port,
            namespace=namespace,
            disable_buffering=True,
        )

    def increment(
        self,
        metric: str,
        value: int,
        tags: tuple[str, ...],
    ) -> None:
        """Increment a counter metric.

        Args:
            metric: Metric name.
            value: Amount to increment by.
            tags: Tuple of tag strings.
        """
        self._statsd.increment(
            metric,
            value,
            tags=list(tags),
        )

    def gauge(
        self,
        metric: str,
        value: float,
        tags: tuple[str, ...],
    ) -> None:
        """Set a gauge metric.

        Args:
            metric: Metric name.
            value: Current value.
            tags: Tuple of tag strings.
        """
        self._statsd.gauge(
            metric,
            value,
            tags=list(tags),
        )

    def histogram(
        self,
        metric: str,
        value: float,
        tags: tuple[str, ...],
    ) -> None:
        """Record a histogram value.

        Args:
            metric: Metric name.
            value: Value to record.
            tags: Tuple of tag strings.
        """
        self._statsd.histogram(
            metric,
            value,
            tags=list(tags),
        )


def _real_metrics_sink_factory(
    host: str,
    port: int,
    namespace: str,
) -> MetricsSinkProtocol:
    """Create a real metrics sink using DogStatsD.

    Args:
        host: DogStatsD agent host.
        port: DogStatsD agent port.
        namespace: Metric namespace prefix.

    Returns:
        RealMetricsSink instance.
    """
    return RealMetricsSink(host, port, namespace)


# Module-level injectable factory for testing.
# Production code calls this; tests override before calling create_metrics_client.
metrics_sink_factory: MetricsSinkFactory = _real_metrics_sink_factory


# =============================================================================
# Tracing Hook
# =============================================================================


class TracingSetupProtocol(Protocol):
    """Protocol for tracing setup function."""

    def __call__(
        self,
        service: str,
        env: str,
        version: str,
    ) -> bool:
        """Configure ddtrace.

        Args:
            service: Service name for traces.
            env: Environment name (dev, staging, prod).
            version: Service version.

        Returns:
            True if tracing was successfully configured.
        """
        ...


def _real_tracing_setup(
    service: str,
    env: str,
    version: str,
) -> bool:
    """Real tracing setup using ddtrace.

    Args:
        service: Service name for traces.
        env: Environment name (dev, staging, prod).
        version: Service version.

    Returns:
        True if tracing was successfully configured.
    """
    from ddtrace import config as dd_config
    from ddtrace import patch

    # Configure ddtrace
    dd_config.service = service
    dd_config.env = env
    dd_config.version = version

    # Enable auto-instrumentation for common libraries
    # Using patch() instead of deprecated patch_all()
    patch(
        fastapi=True,
        httpx=True,
        redis=True,
        logging=True,
    )

    return True


# Module-level injectable setup for testing.
# Production code calls this; tests override to skip ddtrace configuration.
tracing_setup: TracingSetupProtocol = _real_tracing_setup


__all__ = [
    "MetricsSinkFactory",
    "MetricsSinkProtocol",
    "RealMetricsSink",
    "TracingSetupProtocol",
    "metrics_sink_factory",
    "tracing_setup",
]
