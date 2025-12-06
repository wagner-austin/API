from __future__ import annotations

from platform_core.request_context import install_request_id_middleware

from .startup import make_app_from_env

app = make_app_from_env()
install_request_id_middleware(app)

__all__ = ["app"]
