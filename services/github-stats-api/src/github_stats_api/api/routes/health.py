from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse


def build_router() -> APIRouter:
    """Build health check router.

    Returns:
        FastAPI router with health endpoints.
    """
    router = APIRouter(tags=["health"])

    def health() -> PlainTextResponse:
        """Health check endpoint."""
        return PlainTextResponse("ok")

    def healthz() -> PlainTextResponse:
        """Kubernetes-style health check."""
        return PlainTextResponse("ok")

    def readyz() -> PlainTextResponse:
        """Kubernetes-style readiness check."""
        return PlainTextResponse("ok")

    router.add_api_route("/health", health, methods=["GET"])
    router.add_api_route("/healthz", healthz, methods=["GET"])
    router.add_api_route("/readyz", readyz, methods=["GET"])

    return router


__all__ = ["build_router"]
