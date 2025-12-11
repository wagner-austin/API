"""Transcript API routes.

Endpoints:
    Health:
        GET  /healthz        - Liveness probe (always returns ok)
        GET  /readyz         - Readiness probe (checks Redis + workers)

    Transcripts:
        POST /v1/captions    - Get YouTube video captions
        POST /v1/stt         - Speech-to-text transcription for YouTube video
"""

from __future__ import annotations

from .health import build_router as build_health_router
from .transcripts import build_router as build_transcripts_router

__all__ = [
    "build_health_router",
    "build_transcripts_router",
]
