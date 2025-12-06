"""Turkic API routes.

Endpoints:
    Health:
        GET  /healthz                     - Liveness probe (always returns ok)
        GET  /readyz                      - Readiness probe (checks Redis + volume)

    Jobs (/api/v1/jobs):
        POST /api/v1/jobs                 - Create a new corpus processing job
        GET  /api/v1/jobs/{job_id}        - Get job status and metadata
        GET  /api/v1/jobs/{job_id}/result - Stream job result file (requires completed status)
"""

from __future__ import annotations

from .health import build_router as build_health_router
from .jobs import build_router as build_jobs_router

__all__ = [
    "build_health_router",
    "build_jobs_router",
]
