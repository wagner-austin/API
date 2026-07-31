"""Health check utilities for grandma-api.

Uses platform_core.health for the standardized liveness and readiness probes.
"""

from __future__ import annotations

from platform_core.health import HealthResponse, ReadyResponse, healthz


def healthz_endpoint() -> HealthResponse:
    """Liveness probe - always returns ok.

    Returns:
        HealthResponse with status "ok".
    """
    return healthz()


def readyz_endpoint() -> ReadyResponse:
    """Readiness probe - reports ready whenever the process is serving.

    This service holds no queue and no database: every request is handled
    in-process against OpenAI. The credentials it needs (`OPENAI_API_KEY`,
    `API_TOKEN`) are required at config load, so a process that started at all
    has them. There is nothing left to check that would not simply restate
    liveness, and probing OpenAI from a readiness handler would make the probe
    fail on someone else's outage.

    Returns:
        ReadyResponse with status "ready".
    """
    return {"status": "ready", "reason": None}


__all__ = ["healthz_endpoint", "readyz_endpoint"]
