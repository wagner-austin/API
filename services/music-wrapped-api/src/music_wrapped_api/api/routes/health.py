from __future__ import annotations

from collections.abc import Generator

from fastapi import APIRouter
from platform_core.health import (
    HealthResponse,
    ReadyResponse,
    build_health_router,
)
from platform_workers.redis import RedisStrProto

from music_wrapped_api import _test_hooks

from ..health import healthz_endpoint, readyz_endpoint


def _redis_client() -> Generator[RedisStrProto, None, None]:
    url = _test_hooks.require_env("REDIS_URL")
    client = _test_hooks.redis_factory(url)
    try:
        yield client
    finally:
        client.close()


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
        gen = _redis_client()
        client = next(gen)
        try:
            return readyz_endpoint(client)
        finally:
            _ = next(gen, None)

    return build_health_router(healthz_route=_healthz, readyz_route=_readyz)


__all__ = ["build_router"]
