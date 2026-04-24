"""ASGI entry point for doc-extract-api."""

from __future__ import annotations

from platform_core.request_context import install_request_id_middleware

from .startup import make_app

app = make_app()
install_request_id_middleware(app)

__all__ = ["app"]
