"""Shared fixtures for Datadog integration tests."""

from __future__ import annotations

from collections.abc import Generator
from typing import TypedDict

import pytest

from covenant_radar_api.integrations.datadog import _test_hooks
from covenant_radar_api.integrations.datadog.tracing import reset_tracing_state


class MetricCall(TypedDict):
    """Recorded metric call for testing."""

    method: str
    metric: str
    value: float
    tags: tuple[str, ...]


class FakeMetricsSink:
    """Fake metrics sink for testing.

    Records all metric calls for assertion.
    """

    def __init__(self, host: str, port: int, namespace: str) -> None:
        """Initialize fake sink."""
        self.host = host
        self.port = port
        self.namespace = namespace
        self.calls: list[MetricCall] = []

    def increment(
        self,
        metric: str,
        value: int,
        tags: tuple[str, ...],
    ) -> None:
        """Record increment call."""
        self.calls.append(
            {
                "method": "increment",
                "metric": f"{self.namespace}.{metric}",
                "value": float(value),
                "tags": tags,
            }
        )

    def gauge(
        self,
        metric: str,
        value: float,
        tags: tuple[str, ...],
    ) -> None:
        """Record gauge call."""
        self.calls.append(
            {
                "method": "gauge",
                "metric": f"{self.namespace}.{metric}",
                "value": value,
                "tags": tags,
            }
        )

    def histogram(
        self,
        metric: str,
        value: float,
        tags: tuple[str, ...],
    ) -> None:
        """Record histogram call."""
        self.calls.append(
            {
                "method": "histogram",
                "metric": f"{self.namespace}.{metric}",
                "value": value,
                "tags": tags,
            }
        )


def _make_fake_metrics_sink(
    host: str,
    port: int,
    namespace: str,
) -> FakeMetricsSink:
    """Factory for creating fake metrics sinks."""
    return FakeMetricsSink(host, port, namespace)


def _reset_datadog_hooks_impl() -> Generator[None, None, None]:
    """Reset Datadog test hooks after each test."""
    orig_metrics_factory = _test_hooks.metrics_sink_factory
    orig_tracing_setup = _test_hooks.tracing_setup

    yield

    _test_hooks.metrics_sink_factory = orig_metrics_factory
    _test_hooks.tracing_setup = orig_tracing_setup
    reset_tracing_state()


reset_datadog_hooks = pytest.fixture(autouse=True)(_reset_datadog_hooks_impl)


def _fake_metrics_sink_factory_impl() -> tuple[
    _test_hooks.MetricsSinkFactory,
    list[FakeMetricsSink],
]:
    """Provide fake metrics sink factory and list to capture created sinks."""
    created_sinks: list[FakeMetricsSink] = []

    def factory(
        host: str,
        port: int,
        namespace: str,
    ) -> FakeMetricsSink:
        sink = _make_fake_metrics_sink(host, port, namespace)
        created_sinks.append(sink)
        return sink

    return factory, created_sinks


fake_metrics_sink_factory = pytest.fixture(_fake_metrics_sink_factory_impl)


__all__ = [
    "FakeMetricsSink",
    "MetricCall",
    "fake_metrics_sink_factory",
    "reset_datadog_hooks",
]
