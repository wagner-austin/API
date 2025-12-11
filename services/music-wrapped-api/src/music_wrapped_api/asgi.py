from __future__ import annotations

from platform_core.request_context import install_request_id_middleware

from .api.main import create_app

app = create_app()
install_request_id_middleware(app)

__all__ = ["app", "create_app"]
