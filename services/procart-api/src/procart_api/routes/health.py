from __future__ import annotations

from fastapi import APIRouter


def build_router() -> APIRouter:
    """Build health router.

    Returns:
        APIRouter: Router exposing /health endpoint.
    """
    r = APIRouter()

    def health() -> dict[str, str]:
        return {"status": "ok"}

    # Use add_api_route to avoid decorator-induced Any types under mypy --strict.
    r.add_api_route("/health", health, methods=["GET"])
    return r


__all__ = ["build_router"]
