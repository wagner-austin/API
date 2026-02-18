"""FastAPI application factory for Art-Trainer.

This module provides the create_app factory function for the API.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_core.json_utils import JSONTypeError
from platform_core.logging import LogFormat, get_logger, setup_logging
from platform_core.request_context import install_request_id_middleware

from art_trainer.core.config.settings import Settings, load_settings
from art_trainer.core.services.container import ServiceContainer

from .middleware import api_key_dependency
from .routes import dataset, health, lora


async def _json_type_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle JSONTypeError exceptions.

    Args:
        request: The request that caused the error.
        exc: The JSONTypeError exception.

    Returns:
        JSON response with 400 status.
    """
    del request  # Unused
    error_detail: dict[str, str] = {"detail": str(exc)}
    return JSONResponse(
        status_code=400,
        content=error_detail,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings override. If None, loads from environment.

    Returns:
        Configured FastAPI application.
    """
    cfg = settings if settings is not None else load_settings()

    # Set up logging
    format_mode: LogFormat = "json"
    setup_logging(
        level=cfg["logging"]["level"],
        format_mode=format_mode,
        service_name="art-trainer",
        instance_id=None,
        extra_fields=None,
    )

    app = FastAPI(title="Art Trainer API", version="0.1.0")

    # Create container
    container = ServiceContainer.from_settings(cfg)
    app.state.container = container

    # Middleware
    install_request_id_middleware(app)
    app.state.api_key_dep = api_key_dependency(cfg)

    # Routers
    app.include_router(health.build_router(container), prefix="")
    app.include_router(lora.build_router(container), prefix="/lora", tags=["lora"])
    app.include_router(dataset.build_router(container), prefix="/datasets", tags=["datasets"])

    # Error handlers
    app.add_exception_handler(JSONTypeError, _json_type_error_handler)
    install_exception_handlers_fastapi(app, logger_name="art-trainer")

    get_logger(__name__).info("API application initialized")
    return app


__all__ = [
    "create_app",
]
