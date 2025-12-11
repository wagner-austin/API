from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from platform_core.fastapi import install_exception_handlers_fastapi
from platform_core.logging import setup_logging
from platform_core.request_context import install_request_id_middleware

from .. import _test_hooks
from ..config import Settings, settings_from_env
from .routes import files as routes_files
from .routes import health as routes_health


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or settings_from_env()
    setup_logging(
        level="INFO",
        format_mode="json",
        service_name="data-bank-api",
        instance_id=None,
        extra_fields=["request_id"],
    )
    app = FastAPI(title="data-bank-api", version="0.1.0")
    install_request_id_middleware(app)
    install_exception_handlers_fastapi(app)
    storage = _test_hooks.storage_factory(
        root=Path(cfg["data_root"]),
        min_free_gb=cfg["min_free_gb"],
        max_file_bytes=cfg["max_file_bytes"],
    )

    # Include standardized route modules
    app.include_router(routes_health.build_router(cfg))
    app.include_router(routes_files.build_router(storage, cfg))

    return app
