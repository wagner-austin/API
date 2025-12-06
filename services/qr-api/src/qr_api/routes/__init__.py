"""QR API routes.

Endpoints:
    Health:
        GET  /healthz    - Liveness probe (always returns ok)
        GET  /readyz     - Readiness probe (checks Redis + workers)

    QR Generation:
        POST /v1/qr      - Generate QR code PNG from JSON payload
"""

from __future__ import annotations

from .health import build_router as build_health_router
from .qr import build_router as build_qr_router

__all__ = [
    "build_health_router",
    "build_qr_router",
]
