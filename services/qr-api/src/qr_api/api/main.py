from __future__ import annotations

from fastapi import FastAPI
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_core.json_utils import (
    InvalidJsonError,
    register_json_error_handler,
)
from platform_core.logging import setup_logging
from platform_workers.redis import RedisStrProto, redis_for_kv

from ..settings import load_default_options_from_env
from ..validators import Defaults
from .routes import health as routes_health
from .routes import qr as routes_qr


def create_app(defaults: Defaults | None = None) -> FastAPI:
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="qr-api",
        instance_id=None,
        extra_fields=["request_id"],
    )
    app = FastAPI(title="qr-api", version="0.1.0")
    install_exception_handlers_fastapi(app, logger_name="qr-api")
    register_json_error_handler(app, detail="Invalid JSON body")

    # Include standardized route modules
    app.include_router(routes_health.build_router())

    d = defaults or load_default_options_from_env()
    app.include_router(routes_qr.build_router(d))

    return app


__all__ = ["InvalidJsonError", "RedisStrProto", "create_app", "redis_for_kv"]
