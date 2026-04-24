"""Application startup — wires dependencies from environment."""

from __future__ import annotations

from fastapi import FastAPI
from platform_core.logging import setup_logging

from .api.ingest import build_router as build_ingest_router
from .health import build_router as build_health_router
from .ocr import configure_ocr_hook


def make_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Wires the real OCR hook (docTR GPU) and mounts all routes.

    Returns:
        A configured FastAPI instance with all routes mounted.
    """
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="doc-extract-api",
        instance_id=None,
        extra_fields=None,
    )

    configure_ocr_hook()

    app = FastAPI(title="doc-extract-api", version="0.1.0")
    app.include_router(build_health_router())
    app.include_router(build_ingest_router())
    return app


__all__ = ["make_app"]
