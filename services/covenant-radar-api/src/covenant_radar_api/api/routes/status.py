"""Service status endpoint for covenant-radar-api."""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

from covenant_persistence import DealRepository
from fastapi import APIRouter
from platform_core.json_utils import JSONValue, dump_json_str
from platform_core.logging import get_logger
from platform_workers.redis import RedisStrProto
from starlette.responses import Response

from ...core.container import ModelInfo

_log = get_logger(__name__)

# OpenAPI response schema (no type annotation for FastAPI compatibility)
_STATUS_RESPONSES: dict[int | str, dict[str, JSONValue]] = {
    200: {
        "description": "Service status with dependencies, model, and data counts",
        "content": {
            "application/json": {
                "example": {
                    "service": "covenant-radar-api",
                    "version": "0.1.0",
                    "dependencies": [
                        {"name": "redis", "status": "ok", "message": None},
                        {"name": "postgres", "status": "ok", "message": None},
                    ],
                    "model": {
                        "model_id": "default",
                        "model_path": "/data/models/active.ubj",
                        "is_loaded": False,
                    },
                    "data": {"deals": 5},
                },
            },
        },
    },
}


class DependencyStatus(TypedDict, total=True):
    """Status of a service dependency."""

    name: str
    status: Literal["ok", "error"]
    message: str | None


class ServiceStatus(TypedDict, total=True):
    """Overall service status response."""

    service: str
    version: str
    dependencies: list[DependencyStatus]
    model: ModelInfo
    data: dict[str, int]


class StatusContainerProtocol(Protocol):
    """Protocol for status check container."""

    redis: RedisStrProto

    def deal_repo(self) -> DealRepository: ...

    def get_model_info(self) -> ModelInfo: ...


def _check_redis(redis: RedisStrProto) -> DependencyStatus:
    """Check Redis connectivity."""
    try:
        pong = redis.ping()
        if pong:
            return DependencyStatus(name="redis", status="ok", message=None)
        _log.warning("redis ping returned false")
        return DependencyStatus(name="redis", status="error", message="ping returned false")
    except (OSError, RuntimeError) as err:
        _log.warning("redis check failed: %s", err)
        return DependencyStatus(name="redis", status="error", message=str(err))


def _check_database(deal_repo: DealRepository) -> tuple[DependencyStatus, int]:
    """Check database connectivity and count deals."""
    try:
        deals = deal_repo.list_all()
        count = len(list(deals))
        return DependencyStatus(name="postgres", status="ok", message=None), count
    except (OSError, RuntimeError) as err:
        _log.warning("database check failed: %s", err)
        return DependencyStatus(name="postgres", status="error", message=str(err)), 0


def build_router(get_container: StatusContainerProtocol) -> APIRouter:
    """Build status router with /status endpoint.

    Args:
        get_container: Container with dependencies for status checks.
    """
    router = APIRouter()

    def _status() -> Response:
        """Comprehensive service status with dependency health, model info, and data counts.

        Returns JSON with:
        - service: Service name and version
        - dependencies: Health status of Redis and PostgreSQL
        - model: Active ML model info (model_id, path, is_loaded)
        - data: Entity counts (deals)
        """
        redis_status = _check_redis(get_container.redis)
        db_status, deal_count = _check_database(get_container.deal_repo())
        model_info = get_container.get_model_info()

        status = ServiceStatus(
            service="covenant-radar-api",
            version="0.1.0",
            dependencies=[redis_status, db_status],
            model=model_info,
            data={"deals": deal_count},
        )

        data_dict: dict[str, JSONValue] = {
            "deals": status["data"]["deals"],
        }
        body: dict[str, JSONValue] = {
            "service": status["service"],
            "version": status["version"],
            "dependencies": [
                {
                    "name": d["name"],
                    "status": d["status"],
                    "message": d["message"],
                }
                for d in status["dependencies"]
            ],
            "model": {
                "model_id": status["model"]["model_id"],
                "model_path": status["model"]["model_path"],
                "is_loaded": status["model"]["is_loaded"],
            },
            "data": data_dict,
        }
        return Response(
            content=dump_json_str(body),
            media_type="application/json",
        )

    router.add_api_route(
        "/status",
        _status,
        methods=["GET"],
        response_model=None,
        summary="Service status",
        description=(
            "Comprehensive service status with dependency health (Redis, PostgreSQL), "
            "active ML model info, and data counts."
        ),
        response_description="Service status with dependencies, model, and data",
        responses=_STATUS_RESPONSES,
        tags=["health"],
    )
    return router


__all__ = ["DependencyStatus", "ServiceStatus", "StatusContainerProtocol", "build_router"]
