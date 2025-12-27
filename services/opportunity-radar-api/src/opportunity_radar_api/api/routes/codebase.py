"""Codebase profile routes."""

from __future__ import annotations

from fastapi import APIRouter
from platform_codebase import (
    encode_lib_info,
    encode_profile,
    encode_service_info,
)
from platform_core.json_utils import JSONObject

from opportunity_radar_api.api.container import ServiceContainer


def build_router(container: ServiceContainer) -> APIRouter:
    """Build codebase router.

    Args:
        container: Service container with dependencies.

    Returns:
        Configured APIRouter with codebase endpoints.
    """
    router = APIRouter(prefix="/codebase", tags=["codebase"])

    def _get_profile() -> JSONObject:
        """Get the codebase capability profile.

        Returns:
            Codebase profile as JSON.
        """
        profile = container.get_codebase_profile()
        return encode_profile(profile)

    def _list_libs() -> list[JSONObject]:
        """List all libraries in the monorepo.

        Returns:
            List of library info as JSON.
        """
        libs = container.scan_libs()
        return [encode_lib_info(lib) for lib in libs]

    def _list_services() -> list[JSONObject]:
        """List all services in the monorepo.

        Returns:
            List of service info as JSON.
        """
        services = container.scan_services()
        return [encode_service_info(svc) for svc in services]

    router.add_api_route("/profile", _get_profile, methods=["GET"], response_model=None)
    router.add_api_route("/libs", _list_libs, methods=["GET"], response_model=None)
    router.add_api_route("/services", _list_services, methods=["GET"], response_model=None)
    return router
