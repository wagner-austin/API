from __future__ import annotations

from collections.abc import Generator

from fastapi import APIRouter, status
from platform_core.config import _require_env_str
from platform_core.health import HealthResponse, ReadyResponse
from platform_workers.redis import RedisStrProto
from starlette.responses import Response

from ... import _test_hooks
from ...health import healthz_endpoint, readyz_endpoint


def _redis_client() -> Generator[RedisStrProto, None, None]:
    url = _require_env_str("REDIS_URL")
    client = _test_hooks.redis_factory(url)
    try:
        yield client
    finally:
        client.close()


def build_router() -> APIRouter:
    """Build health router with /healthz and /readyz endpoints."""
    router = APIRouter()

    def _healthz_handler() -> HealthResponse:
        return healthz_endpoint()

    def _readyz_handler(resp: Response) -> ReadyResponse:
        gen = _redis_client()
        client = next(gen)
        try:
            result = readyz_endpoint(client)
        finally:
            _ = next(gen, None)
        if result["status"] == "degraded":
            resp.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result

    router.add_api_route("/healthz", _healthz_handler, methods=["GET"])
    router.add_api_route("/readyz", _readyz_handler, methods=["GET"])
    return router


__all__ = ["build_router"]
