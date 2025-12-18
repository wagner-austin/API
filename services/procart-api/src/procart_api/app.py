from __future__ import annotations

from fastapi import FastAPI
from platform_core.fastapi import install_exception_handlers_fastapi

from .routes.health import build_router as build_health_router
from .routes.registries import build_router as build_registries_router
from .routes.render import build_router as build_render_router


def create_app() -> FastAPI:
    """Create FastAPI app with platform-standard exception handlers.

    Returns:
        FastAPI: Configured FastAPI application.
    """
    app = FastAPI(title="procart-api", version="0.1.0")
    install_exception_handlers_fastapi(app, logger_name="procart-api")

    # Include typed routers to avoid decorator Any leakage under mypy --strict
    app.include_router(build_health_router())
    app.include_router(build_registries_router())
    app.include_router(build_render_router())

    # Routes for scenes, preview, frames, and video will be added via routers.

    return app


__all__ = ["create_app"]
