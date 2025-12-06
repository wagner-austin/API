from __future__ import annotations

from fastapi import FastAPI
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_core.json_utils import (
    InvalidJsonError,
    register_json_error_handler,
)
from platform_core.logging import setup_logging
from platform_workers.redis import RedisStrProto, redis_for_kv

from .routes import health as routes_health
from .routes import wrapped as routes_wrapped


def create_app() -> FastAPI:
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="music-wrapped-api",
        instance_id=None,
        extra_fields=["request_id"],
    )
    app = FastAPI(title="music-wrapped-api", version="0.1.0")
    install_exception_handlers_fastapi(app, logger_name="music-wrapped-api")
    register_json_error_handler(app, detail="Invalid JSON body")

    app.include_router(routes_health.build_router())
    app.include_router(routes_wrapped.build_router())

    return app


__all__ = ["InvalidJsonError", "RedisStrProto", "create_app", "redis_for_kv"]
