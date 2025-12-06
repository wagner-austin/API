from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from fastapi import FastAPI
from platform_core.errors import ErrorCode
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_core.logging import setup_logging
from platform_core.request_context import install_request_id_middleware
from platform_workers.redis import RedisStrProto

from turkic_api.api.logging_fields import LOG_EXTRA_FIELDS
from turkic_api.api.provider_context import (
    LoggerProvider,
    QueueProviderType,
    RedisProviderType,
    SettingsProvider,
)
from turkic_api.api.provider_context import (
    provider_context as _provider_context,
)
from turkic_api.api.routes import health as routes_health
from turkic_api.api.routes import jobs as routes_jobs

_SimpleValue = str | int | float | bool | None | datetime


def _to_json_simple(obj: Mapping[str, _SimpleValue]) -> dict[str, _SimpleValue]:
    """Convert a flat TypedDict with datetime values to JSON-serializable dict.

    The API models used here (JobResponse, HealthResponse, JobStatus) are flat
    mappings with primitive values or datetimes; no nested containers required.
    """
    out: dict[str, _SimpleValue] = {}
    for k, v in obj.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        else:
            # v must be a primitive given _SimpleValue type constraints
            out[k] = v
    return out


# Initialize centralized logging (idempotent).
def _init_logging() -> None:
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="turkic-api",
        instance_id=None,
        extra_fields=LOG_EXTRA_FIELDS,
    )


RedisCombinedProtocol = RedisStrProto


# re-export selected context for tests (names expected by tests)


def create_app(
    *,
    redis_provider: RedisProviderType | None = None,
    queue_provider: QueueProviderType | None = None,
    settings_provider: SettingsProvider | None = None,
    logger_provider: LoggerProvider | None = None,
) -> FastAPI:
    _init_logging()
    app = FastAPI(title="Turkic API", version="1.0.0")
    install_request_id_middleware(app)
    install_exception_handlers_fastapi(
        app,
        logger_name="turkic-api",
        internal_error_code=ErrorCode.INTERNAL_ERROR,
    )

    _provider_context.settings_provider = settings_provider
    _provider_context.redis_provider = redis_provider
    _provider_context.queue_provider = queue_provider
    _provider_context.logger_provider = logger_provider

    app.include_router(routes_health.build_router())
    app.include_router(routes_jobs.build_router())
    return app
