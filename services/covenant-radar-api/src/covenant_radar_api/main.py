"""Application factory for covenant-radar-api."""

from __future__ import annotations

from fastapi import FastAPI
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_core.logging import setup_logging
from platform_core.request_context import install_request_id_middleware

from .api.routes import covenants as routes_covenants
from .api.routes import deals as routes_deals
from .api.routes import evaluate as routes_evaluate
from .api.routes import health as routes_health
from .api.routes import measurements as routes_measurements
from .api.routes import ml as routes_ml
from .core.container import ServiceContainer


def create_app(container: ServiceContainer) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        container: Service container with all dependencies.

    Returns:
        Configured FastAPI application instance.
    """
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="covenant-radar-api",
        instance_id=None,
        extra_fields=["request_id"],
    )
    app = FastAPI(title="covenant-radar-api", version="0.1.0")
    install_request_id_middleware(app)
    install_exception_handlers_fastapi(app)

    app.include_router(routes_health.build_router(container))
    app.include_router(routes_deals.build_router(container))
    app.include_router(routes_covenants.build_router(container))
    app.include_router(routes_measurements.build_router(container))
    app.include_router(routes_evaluate.build_router(container))
    app.include_router(routes_ml.build_router(container))

    return app
