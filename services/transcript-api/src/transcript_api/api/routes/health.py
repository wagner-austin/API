"""Health check routes for transcript-api."""

from __future__ import annotations

from fastapi import APIRouter
from platform_core.health import (
    HealthResponse,
    ReadyResponse,
    build_health_router,
)

from ...dependencies import get_redis
from ...health import healthz_endpoint, readyz_endpoint


def build_router() -> APIRouter:
    """Build this service's health router.

    The router shape and the degraded-to-503 mapping live in
    ``platform_core.health``. What stays here is this service's own: driving
    the generator dependency by hand so ``close()`` runs deterministically
    whether or not the readiness check raises.

    Returns:
        APIRouter with health endpoints configured.
    """

    def _healthz() -> HealthResponse:
        return healthz_endpoint()

    def _readyz() -> ReadyResponse:
        gen = get_redis()
        client = next(gen)
        try:
            return readyz_endpoint(client)
        finally:
            _ = next(gen, None)

    return build_health_router(healthz_route=_healthz, readyz_route=_readyz)


__all__ = ["build_router"]
