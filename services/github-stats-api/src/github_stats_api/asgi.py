"""ASGI application entry point for github-stats-api.

This module creates the FastAPI app instance that is loaded by the ASGI server.
"""

from __future__ import annotations

from platform_core.request_context import install_request_id_middleware

from .api.main import create_app

# Create app and install request ID middleware
app = create_app()
install_request_id_middleware(app)

__all__ = ["app"]
