from __future__ import annotations

from fastapi import FastAPI
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_core.logging import LogFormat, get_logger, setup_logging
from platform_core.request_context import install_request_id_middleware

from ..core.config.settings import Settings, load_settings
from ..core.logging.types import LOGGING_EXTRA_FIELDS
from ..core.logging.utils import narrow_log_level
from ..core.services.container import ServiceContainer
from .middleware import api_key_dependency
from .routes import health, runs, tokenizers


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    level = narrow_log_level(cfg["logging"]["level"])
    format_mode: LogFormat = "json"
    setup_logging(
        level=level,
        format_mode=format_mode,
        service_name="model-trainer",
        instance_id=None,
        extra_fields=list(LOGGING_EXTRA_FIELDS),
    )
    app = FastAPI(title="Model Trainer API", version="0.1.0")

    container = ServiceContainer.from_settings(cfg)
    # Expose container for testability and tooling
    app.state.container = container

    # Middleware: request correlation and strict API key enforcement
    install_request_id_middleware(app)
    # FastAPI dependency for API key (required); attach to routers where appropriate
    app.state.api_key_dep = api_key_dependency(cfg)

    # Routers (container captured in closures)
    app.include_router(health.build_router(container), prefix="")
    app.include_router(runs.build_router(container), prefix="/runs", tags=["runs"])
    app.include_router(
        tokenizers.build_router(container), prefix="/tokenizers", tags=["tokenizers"]
    )
    # Local artifacts routes removed: artifacts are now stored in data-bank-api

    # Errors
    install_exception_handlers_fastapi(app, logger_name="model-trainer")

    get_logger(__name__).info("API application initialized")
    return app
