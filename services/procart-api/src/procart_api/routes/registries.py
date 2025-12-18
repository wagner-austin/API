from __future__ import annotations

from fastapi import APIRouter
from procart.registry import list_available_modules
from procart.registry_camera import list_available_camera_paths
from procart.registry_composite import list_available_composite_ops
from procart.registry_post import list_available_post_effects
from procart.registry_tone import list_available_tone_mappers


def build_router() -> APIRouter:
    """Build registries router exposing library registry names.

    Returns:
        APIRouter: Router exposing list endpoints for registered names.
    """
    r = APIRouter(prefix="/registries")

    def modules() -> dict[str, list[str]]:
        return {"modules": list_available_modules()}

    def cameras() -> dict[str, list[str]]:
        return {"camera_paths": list_available_camera_paths()}

    def tones() -> dict[str, list[str]]:
        return {"tone_mappers": list_available_tone_mappers()}

    def posts() -> dict[str, list[str]]:
        return {"post_effects": list_available_post_effects()}

    def composites() -> dict[str, list[str]]:
        return {"composite_ops": list_available_composite_ops()}

    r.add_api_route("/modules", modules, methods=["GET"])
    r.add_api_route("/camera-paths", cameras, methods=["GET"])
    r.add_api_route("/tone-mappers", tones, methods=["GET"])
    r.add_api_route("/post-effects", posts, methods=["GET"])
    r.add_api_route("/composite-ops", composites, methods=["GET"])
    return r


__all__ = ["build_router"]
