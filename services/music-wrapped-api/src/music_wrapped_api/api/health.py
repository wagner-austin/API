"""Health module for music-wrapped-api."""

from __future__ import annotations

from music_wrapped_api.health import healthz_endpoint, readyz_endpoint

__all__ = ["healthz_endpoint", "readyz_endpoint"]
