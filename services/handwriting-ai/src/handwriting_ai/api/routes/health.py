"""Health check routes for handwriting-ai."""

from __future__ import annotations

from fastapi import APIRouter, Response
from platform_core.config import _require_env_str
from platform_core.health import HealthResponse, ReadyResponse, healthz
from platform_workers.health import readyz_redis_with_workers
from platform_workers.redis import RedisStrProto, redis_for_kv

from ...inference.engine import InferenceEngine


def build_router(engine: InferenceEngine) -> APIRouter:
    """Build health router with /healthz and /readyz endpoints."""
    router = APIRouter()

    async def _healthz() -> HealthResponse:
        return healthz()

    async def _readyz(response: Response) -> ReadyResponse:
        # First verify Redis connectivity and worker presence
        redis_url = _require_env_str("REDIS_URL")
        client: RedisStrProto = redis_for_kv(redis_url)
        try:
            redis_status = readyz_redis_with_workers(client)
        finally:
            client.close()
        if redis_status["status"] == "degraded":
            response.status_code = 503
            return redis_status

        # Then verify model engine readiness
        man = engine.manifest
        if engine.ready and man is not None:
            return {"status": "ready", "reason": None}
        response.status_code = 503
        return {"status": "degraded", "reason": "model not loaded"}

    router.add_api_route("/healthz", _healthz, methods=["GET"])
    router.add_api_route("/readyz", _readyz, methods=["GET"])
    return router


__all__ = ["build_router"]
