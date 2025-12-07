"""Music Wrapped API routes.

Endpoints:
    Health:
        GET  /healthz                              - Liveness probe (always returns ok)
        GET  /readyz                               - Readiness probe (checks Redis)

    Authentication (/v1/wrapped/auth):
        GET  /v1/wrapped/auth/spotify/start        - Start Spotify OAuth flow
        GET  /v1/wrapped/auth/spotify/callback     - Spotify OAuth callback
        GET  /v1/wrapped/auth/lastfm/start         - Start Last.fm OAuth flow
        GET  /v1/wrapped/auth/lastfm/callback      - Last.fm OAuth callback
        POST /v1/wrapped/auth/youtube/store        - Store YouTube Music credentials
        POST /v1/wrapped/auth/apple/store          - Store Apple Music token

    Wrapped Generation (/v1/wrapped):
        POST /v1/wrapped/generate                  - Start wrapped generation job
        POST /v1/wrapped/import/youtube-takeout    - Import YouTube Takeout file
        GET  /v1/wrapped/status/{job_id}           - Get job status and progress
        GET  /v1/wrapped/result/{result_id}        - Get JSON result
        GET  /v1/wrapped/download/{result_id}      - Download PNG image
        GET  /v1/wrapped/schema                    - Get WrappedResult JSON schema
"""

from __future__ import annotations

from .health import build_router as build_health_router
from .wrapped import build_router as build_wrapped_router

__all__ = [
    "build_health_router",
    "build_wrapped_router",
]
