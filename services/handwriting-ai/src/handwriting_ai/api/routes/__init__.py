"""Handwriting AI routes.

Endpoints:
    Health:
        GET  /healthz                     - Liveness probe (always returns ok)
        GET  /readyz                      - Readiness probe (checks Redis + model)

    Models:
        GET  /v1/models/active            - Get active model info

    Read/Predict (API key required):
        POST /v1/read                     - Read handwritten digit from image
        POST /v1/predict                  - Alias for /v1/read

    Admin (API key required):
        POST /v1/admin/models/upload      - Upload model artifacts
"""

from __future__ import annotations

from .admin import build_router as build_admin_router
from .health import build_router as build_health_router
from .models import build_router as build_models_router
from .read import build_router as build_read_router

__all__ = [
    "build_admin_router",
    "build_health_router",
    "build_models_router",
    "build_read_router",
]
