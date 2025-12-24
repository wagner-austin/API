"""FastAPI application factory for grandma-api.

Provides create_app() factory for creating the FastAPI application with
ServiceContainer dependency injection and middleware configuration.
"""

from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_core.logging import get_logger, setup_logging
from platform_core.request_context import install_request_id_middleware

from grandma_api.config import GrandmaApiSettings, load_settings
from grandma_api.core.container import ServiceContainer

from .middleware import api_key_dependency
from .routes import health as routes_health
from .routes import translate as routes_translate


class _FastAPIStateProto(Protocol):
    """Protocol for FastAPI app.state to expose typed container access."""

    container: ServiceContainer


def create_app(
    settings: GrandmaApiSettings | None = None,
    container: ServiceContainer | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Application settings. Loads from environment if None.
        container: Service container. Creates from settings if None.

    Returns:
        Configured FastAPI application with container on app.state.
    """
    cfg = settings if settings is not None else load_settings()

    setup_logging(
        level=cfg["log_level"],
        format_mode=cfg["log_format"],
        service_name="grandma-api",
        instance_id=None,
        extra_fields=["request_id"],
    )

    app = FastAPI(
        title="Grandma API",
        description="Vietnamese to English audio translation API",
        version="0.1.0",
    )

    # Create or use provided service container
    svc = container if container is not None else ServiceContainer.from_settings(cfg)

    # Expose container for testability and tooling
    state: _FastAPIStateProto = app.state
    state.container = svc

    # Middleware: request correlation
    install_request_id_middleware(app)

    # FastAPI dependency for API key; attach to routers where appropriate
    app.state.api_key_dep = api_key_dependency(cfg)

    # Install exception handlers
    install_exception_handlers_fastapi(app, logger_name="grandma-api")

    # Add CORS middleware for GitHub Pages frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Include route modules
    app.include_router(routes_health.build_router())
    app.include_router(routes_translate.build_router(svc))

    get_logger(__name__).info("API application initialized")

    return app


__all__ = ["create_app"]
