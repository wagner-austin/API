"""Application factory for covenant-radar-api."""

from __future__ import annotations

from fastapi import FastAPI
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_core.logging import setup_logging
from platform_core.request_context import install_request_id_middleware

from ..core.config import Settings, settings_from_env
from ..core.container import ServiceContainer
from .routes import covenants as routes_covenants
from .routes import deals as routes_deals
from .routes import evaluate as routes_evaluate
from .routes import health as routes_health
from .routes import measurements as routes_measurements
from .routes import ml as routes_ml
from .routes import status as routes_status


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Service settings. If None, reads from environment variables.

    Returns:
        Configured FastAPI application instance.
    """
    cfg = settings or settings_from_env()
    # Setup logging first so model loading logs appear in JSON format
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="covenant-radar-api",
        instance_id=None,
        extra_fields=["request_id", "model_path"],
    )
    # Create container with eager model loading for fast first predictions
    container = ServiceContainer.from_settings(cfg, eager_load_model=True)
    app = FastAPI(title="covenant-radar-api", version="0.1.0")
    install_request_id_middleware(app)
    install_exception_handlers_fastapi(app)

    app.include_router(routes_health.build_router(container))
    app.include_router(routes_status.build_router(container))
    app.include_router(routes_deals.build_router(container))
    app.include_router(routes_covenants.build_router(container))
    app.include_router(routes_measurements.build_router(container))
    app.include_router(routes_evaluate.build_router(container))
    app.include_router(routes_ml.build_router(container))

    return app
