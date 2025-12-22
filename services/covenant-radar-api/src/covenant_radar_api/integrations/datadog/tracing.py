"""Datadog APM tracing setup.

This module provides ddtrace configuration for:
- FastAPI request lifecycle tracing
- Outbound HTTP call tracing (httpx)
- Redis operation tracing
- Log correlation with trace IDs

Usage:
    from covenant_radar_api.integrations.datadog import setup_datadog_tracing

    if settings.datadog_enabled:
        setup_datadog_tracing(
            service="covenant-radar-api",
            env="production",
            version="1.0.0",
        )
"""

from __future__ import annotations

from typing import Literal, TypedDict

from . import _test_hooks


class DatadogConfig(TypedDict, total=True):
    """Datadog configuration settings.

    Fields:
        enabled: Whether Datadog tracing is enabled.
        service: Service name for traces (e.g., "covenant-radar-api").
        env: Environment name (dev, staging, production).
        version: Service version for trace filtering.
        agent_host: Datadog agent host (default: localhost).
        dogstatsd_port: DogStatsD port for metrics (default: 8125).
        trace_enabled: Whether APM tracing is enabled (default: True when enabled=True).
    """

    enabled: bool
    service: str
    env: Literal["dev", "staging", "production"]
    version: str
    agent_host: str
    dogstatsd_port: int
    trace_enabled: bool


class TracingState(TypedDict, total=True):
    """State of tracing configuration.

    Fields:
        configured: Whether tracing has been configured.
        service: Service name configured.
        env: Environment configured.
        version: Version configured.
    """

    configured: bool
    service: str
    env: str
    version: str


# Module-level state tracking
_tracing_state: TracingState = {
    "configured": False,
    "service": "",
    "env": "",
    "version": "",
}


def setup_datadog_tracing(
    service: str,
    env: str,
    version: str,
) -> TracingState:
    """Configure Datadog APM tracing.

    This function should be called once at application startup,
    before any other imports that could be instrumented.

    The tracing setup:
    1. Sets service, env, version tags for all traces
    2. Patches common libraries (FastAPI, httpx, redis, logging)
    3. Enables trace ID correlation in logs

    Args:
        service: Service name for traces (e.g., "covenant-radar-api").
        env: Environment name (dev, staging, production).
        version: Service version for trace filtering.

    Returns:
        TracingState with configuration status.

    Example:
        >>> state = setup_datadog_tracing(
        ...     service="covenant-radar-api",
        ...     env="production",
        ...     version="1.0.0",
        ... )
        >>> state["configured"]
        True
    """
    global _tracing_state

    # Skip if already configured
    if _tracing_state["configured"]:
        return _tracing_state

    # Use the injectable hook for actual setup
    success = _test_hooks.tracing_setup(service, env, version)

    _tracing_state = {
        "configured": success,
        "service": service,
        "env": env,
        "version": version,
    }

    return _tracing_state


def get_tracing_state() -> TracingState:
    """Get the current tracing configuration state.

    Returns:
        TracingState with current configuration.
    """
    return _tracing_state


def reset_tracing_state() -> None:
    """Reset tracing state to unconfigured.

    This is primarily for testing purposes.
    """
    global _tracing_state
    _tracing_state = {
        "configured": False,
        "service": "",
        "env": "",
        "version": "",
    }


def make_default_datadog_config() -> DatadogConfig:
    """Create default Datadog configuration.

    Returns:
        DatadogConfig with sensible defaults for local development.
    """
    return {
        "enabled": False,
        "service": "covenant-radar-api",
        "env": "dev",
        "version": "0.0.0",
        "agent_host": "localhost",
        "dogstatsd_port": 8125,
        "trace_enabled": True,
    }


__all__ = [
    "DatadogConfig",
    "TracingState",
    "get_tracing_state",
    "make_default_datadog_config",
    "reset_tracing_state",
    "setup_datadog_tracing",
]
