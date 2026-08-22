"""Dashboard route serving a simple HTML visualization UI."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse

from covenant_radar_api.api.routes.dashboard_html import (
    _DASHBOARD_HTML,
)

# Chart.js is vendored under api/static rather than pulled from a CDN; see
# _get_chart_js for why SRI could not pin the jsDelivr URL.
_CHART_JS_ROUTE = "/dashboard/chart.umd.min.js"
_CHART_JS_PATH = Path(__file__).resolve().parent.parent / "static" / "chart.umd.min.js"

# Dashboard HTML with embedded JS for a lightweight, single-page monitoring UI


class ContainerProtocol(Protocol):
    """Minimal protocol for dashboard - no dependencies needed."""


def build_router(get_container: ContainerProtocol) -> APIRouter:
    """Build FastAPI router for dashboard.

    Args:
        get_container: Service container (unused, but matches pattern).

    Returns:
        Configured API router.
    """
    router = APIRouter(tags=["dashboard"])

    def _get_dashboard() -> HTMLResponse:
        """Serve the dashboard HTML page.

        Returns:
            HTML response with the dashboard UI.
        """
        return HTMLResponse(content=_DASHBOARD_HTML)

    def _get_chart_js() -> Response:
        """Serve the vendored Chart.js bundle.

        Chart.js is vendored rather than loaded from jsDelivr. The jsDelivr
        `/npm/` path is dynamically generated, and jsDelivr explicitly document
        that Subresource Integrity must not be used with it — so the CDN script
        could not be pinned, leaving the dashboard trusting whatever that host
        returned. Serving it from here removes the third-party trust entirely
        and makes the dashboard work without external network access.

        Returns:
            The Chart.js bundle as JavaScript.
        """
        return Response(
            content=_CHART_JS_PATH.read_text(encoding="utf-8"),
            media_type="application/javascript",
        )

    router.add_api_route(
        "/dashboard",
        _get_dashboard,
        methods=["GET"],
        response_model=None,
        summary="Dashboard UI",
        description="Serves the real-time risk monitoring dashboard.",
    )
    router.add_api_route(
        _CHART_JS_ROUTE,
        _get_chart_js,
        methods=["GET"],
        response_model=None,
        summary="Vendored Chart.js bundle",
        description="Serves the Chart.js bundle used by the dashboard.",
    )

    return router


__all__ = ["build_router"]
