"""Grandma API routes.

Endpoints:
    Health:
        GET  /healthz    - Liveness probe (always returns ok)

    Translation:
        POST /translate  - Translate Vietnamese audio to English text
"""

from __future__ import annotations

from grandma_api.api.routes.health import build_router as build_health_router
from grandma_api.api.routes.translate import build_router as build_translate_router

__all__ = [
    "build_health_router",
    "build_translate_router",
]
