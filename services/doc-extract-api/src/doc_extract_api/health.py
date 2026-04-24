"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter


def _readyz() -> dict[str, str]:
    """Readiness probe.

    Returns:
        JSON with status "ok".
    """
    return {"status": "ok"}


def _healthz() -> dict[str, str]:
    """Liveness probe.

    Returns:
        JSON with status "ok".
    """
    return {"status": "ok"}


def build_router() -> APIRouter:
    """Build health router with readiness and liveness endpoints.

    Returns:
        APIRouter with health endpoints configured.
    """
    router = APIRouter()
    router.add_api_route("/readyz", _readyz, methods=["GET"])
    router.add_api_route("/healthz", _healthz, methods=["GET"])
    return router


__all__ = ["build_router"]
