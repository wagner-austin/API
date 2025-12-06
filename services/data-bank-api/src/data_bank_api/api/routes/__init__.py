"""Data Bank API routes.

Endpoints:
    Health:
        GET  /healthz                - Liveness probe (always returns ok)
        GET  /readyz                 - Readiness probe (checks Redis + storage + disk)

    Files:
        POST   /files                - Upload a new file
        HEAD   /files/{file_id}      - Get file metadata via headers
        GET    /files/{file_id}      - Download file (supports Range header)
        GET    /files/{file_id}/info - Get file metadata as JSON
        DELETE /files/{file_id}      - Delete a file
"""

from __future__ import annotations

from .files import build_router as build_files_router
from .health import build_router as build_health_router

__all__ = [
    "build_files_router",
    "build_health_router",
]
