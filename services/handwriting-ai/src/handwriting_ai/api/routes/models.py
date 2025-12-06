"""Model info routes for handwriting-ai."""

from __future__ import annotations

from fastapi import APIRouter

from ...inference.engine import InferenceEngine

# Non-recursive JSON value type for flat API responses
_FlatJsonValue = str | int | float | bool | list[float] | None


def build_router(engine: InferenceEngine) -> APIRouter:
    """Build models router with /v1/models/active endpoint."""
    router = APIRouter()

    async def _model_active() -> dict[str, _FlatJsonValue]:
        man = engine.manifest
        if man is None:
            return {"model_loaded": False, "model_id": None}
        return {
            "model_loaded": True,
            "model_id": man["model_id"],
            "arch": man["arch"],
            "n_classes": man["n_classes"],
            "version": man["version"],
            "created_at": man["created_at"].isoformat(),
            "schema_version": man["schema_version"],
            "val_acc": man["val_acc"],
            "temperature": man["temperature"],
        }

    router.add_api_route("/v1/models/active", _model_active, methods=["GET"])
    return router


__all__ = ["build_router"]
