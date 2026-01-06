"""Health check utilities for transcript-api.

Uses platform_core.health for standardized liveness probe.
"""

from __future__ import annotations

from platform_core.health import HealthResponse, healthz


def healthz_endpoint() -> HealthResponse:
    """Liveness probe - always returns ok.

    Returns:
        HealthResponse with status "ok".
    """
    return healthz()


__all__ = ["healthz_endpoint"]
