from __future__ import annotations

from fastapi import FastAPI
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_core.logging import setup_logging
from platform_core.request_context import install_request_id_middleware

from ..settings import Settings, load_settings
from .routes import health as routes_health
from .routes import stats as routes_stats


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create FastAPI application.

    Args:
        settings: Optional settings override (for testing).

    Returns:
        Configured FastAPI application.
    """
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="github-stats-api",
        instance_id=None,
        extra_fields=["request_id"],
    )

    app = FastAPI(
        title="github-stats-api",
        version="0.1.0",
        description="GitHub stats SVG card generation API",
    )
    install_exception_handlers_fastapi(app, logger_name="github-stats-api")
    install_request_id_middleware(app)

    # Health routes
    app.include_router(routes_health.build_router())

    # Stats routes with settings provider
    resolved_settings = settings or load_settings()

    def settings_provider() -> Settings:
        return resolved_settings

    app.include_router(routes_stats.build_router(settings_provider))

    return app


__all__ = ["create_app"]
