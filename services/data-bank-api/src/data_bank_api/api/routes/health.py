"""Health check routes for data-bank-api."""

from __future__ import annotations

from fastapi import APIRouter, status
from platform_core.health import HealthResponse, ReadyResponse
from platform_workers.redis import RedisStrProto, redis_for_kv
from starlette.responses import Response

from ...config import Settings
from ...health import healthz_endpoint, readyz_endpoint


def build_router(cfg: Settings) -> APIRouter:
    """Build health router with /healthz and /readyz endpoints."""
    router = APIRouter()

    def _healthz() -> HealthResponse:
        return healthz_endpoint()

    def _readyz(resp: Response) -> ReadyResponse:
        redis: RedisStrProto = redis_for_kv(cfg["redis_url"])
        result = readyz_endpoint(
            redis=redis,
            data_root=cfg["data_root"],
            min_free_gb=cfg["min_free_gb"],
        )
        redis.close()
        if result["status"] == "degraded":
            resp.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result

    router.add_api_route("/healthz", _healthz, methods=["GET"], response_model=None)
    router.add_api_route("/readyz", _readyz, methods=["GET"], response_model=None)
    return router


__all__ = ["build_router"]
