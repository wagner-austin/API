from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from platform_core.health import HealthResponse, ReadyResponse, healthz
from platform_workers.redis import RedisStrProto

from ..config import Settings
from ..health import readyz_endpoint
from ..provider_context import (
    get_redis_from_context,
    get_settings_from_context,
)


def build_router() -> APIRouter:
    router = APIRouter()

    async def _healthz() -> HealthResponse:
        return healthz()

    async def _readyz(
        redis: Annotated[RedisStrProto, Depends(get_redis_from_context)],
        settings: Annotated[Settings, Depends(get_settings_from_context)],
    ) -> JSONResponse:
        result: ReadyResponse = readyz_endpoint(redis=redis, data_dir=settings["data_dir"])
        content: dict[str, str | None] = {"status": result["status"], "reason": result["reason"]}
        status_code = 200 if result["status"] == "ready" else 503
        return JSONResponse(content=content, status_code=status_code)

    router.add_api_route("/healthz", _healthz, methods=["GET"])
    router.add_api_route("/readyz", _readyz, methods=["GET"])

    return router
