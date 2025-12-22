"""Datadog integration for APM tracing and custom metrics.

This module provides:
- APM tracing with ddtrace auto-instrumentation
- Custom metrics emission via DogStatsD
- Test hooks for DI in tests

Usage:
    from covenant_radar_api.integrations.datadog import (
        DatadogConfig,
        setup_datadog_tracing,
        MetricsClient,
        create_metrics_client,
    )
"""

from __future__ import annotations

from .metrics import MetricsClient, MetricsConfig, create_metrics_client
from .tracing import DatadogConfig, TracingState, setup_datadog_tracing

__all__ = [
    "DatadogConfig",
    "MetricsClient",
    "MetricsConfig",
    "TracingState",
    "create_metrics_client",
    "setup_datadog_tracing",
]
