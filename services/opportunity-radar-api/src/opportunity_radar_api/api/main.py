"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from platform_core.fastapi import install_exception_handlers_fastapi

from opportunity_radar_api.api.container import ServiceContainer, create_production_container
from opportunity_radar_api.api.routes import codebase as routes_codebase
from opportunity_radar_api.api.routes import devpost as routes_devpost
from opportunity_radar_api.api.routes import health as routes_health
from opportunity_radar_api.api.routes import kaggle as routes_kaggle
from opportunity_radar_api.config import OpportunityRadarSettings, load_settings


def create_app(
    container: ServiceContainer | None = None,
    settings: OpportunityRadarSettings | None = None,
    monorepo_root: Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        container: Optional pre-configured container. If None, creates production.
        settings: Optional settings. Only used if container is None.
        monorepo_root: Path to monorepo root. Only used if container is None.

    Returns:
        Configured FastAPI application instance.
    """
    if container is None:
        if settings is None:
            settings = load_settings()
        container = create_production_container(settings, monorepo_root)

    app = FastAPI(
        title="opportunity-radar-api",
        version="0.1.0",
        description="Discover Kaggle and Devpost opportunities matching codebase capabilities",
    )

    install_exception_handlers_fastapi(app)

    # Include routers
    app.include_router(routes_health.build_router())
    app.include_router(routes_codebase.build_router(container))
    app.include_router(routes_kaggle.build_router(container))
    app.include_router(routes_devpost.build_router(container))

    return app
