from __future__ import annotations

from collections.abc import Generator

from fastapi import APIRouter
from platform_core.health import HealthResponse, ReadyResponse
from platform_workers.redis import RedisStrProto
from starlette.responses import Response

from ...health import healthz_endpoint, readyz_endpoint
from .. import _test_hooks


def _redis_client() -> Generator[RedisStrProto, None, None]:
    url = _test_hooks.get_env("REDIS_URL")
    if url is None:
        raise RuntimeError("REDIS_URL environment variable is required")
    client = _test_hooks.redis_factory(url)
    try:
        yield client
    finally:
        client.close()


def build_router() -> APIRouter:
    """Build health router with /healthz and /readyz endpoints."""
    router = APIRouter()

    def _healthz() -> HealthResponse:
        return healthz_endpoint()

    def _readyz(resp: Response) -> ReadyResponse:
        # Validate Redis connectivity and workers
        # Use a generator dependency inline to ensure close() is called
        gen = _redis_client()
        client = next(gen)
        try:
            res = readyz_endpoint(client)
        finally:
            # Close generator to ensure client.close() is invoked deterministically
            _ = next(gen, None)
        if res["status"] == "degraded":
            resp.status_code = 503
        return res

    router.add_api_route("/healthz", _healthz, methods=["GET"])
    router.add_api_route("/readyz", _readyz, methods=["GET"])
    return router


__all__ = ["build_router"]
