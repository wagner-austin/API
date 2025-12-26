from __future__ import annotations

from typing import Protocol

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response

from ..._test_hooks import get_client_hook
from ...client import GitHubClient
from ...settings import Settings
from ...svg_renderer import (
    build_language_stats,
    build_user_stats,
    render_langs_card,
    render_stats_card,
)
from ..validators.stats import decode_langs_request, decode_stats_request

_HTTP_TIMEOUT_SECONDS = 30.0


class _SettingsProvider(Protocol):
    """Protocol for settings provider."""

    def __call__(self) -> Settings:
        """Get settings.

        Returns:
            Service settings.
        """
        ...


class _StatsRoutes:
    """Stats routes handler.

    Attributes:
        _settings_provider: Callable that returns settings.
    """

    _settings_provider: _SettingsProvider

    def __init__(self, settings_provider: _SettingsProvider) -> None:
        """Initialize routes handler.

        Args:
            settings_provider: Callable that returns settings.
        """
        self._settings_provider = settings_provider

    async def get_stats(
        self,
        request: Request,
        username: str | None = Query(default=None, description="GitHub username"),
        theme: str | None = Query(default=None, description="Color theme"),
        hide_border: str | None = Query(default=None, description="Hide border"),
        show_icons: str | None = Query(default=None, description="Show icons"),
        include_all_commits: str | None = Query(default=None, description="Include all commits"),
        hide: str | None = Query(default=None, description="Stats to hide (comma-separated)"),
    ) -> Response:
        """Get user stats SVG card.

        Args:
            request: FastAPI request.
            username: GitHub username.
            theme: Color theme name.
            hide_border: Whether to hide border.
            show_icons: Whether to show icons.
            include_all_commits: Include all commits.
            hide: Comma-separated stats to hide.

        Returns:
            SVG response.
        """
        req = decode_stats_request(
            username=username,
            theme=theme,
            hide_border=hide_border,
            show_icons=show_icons,
            include_all_commits=include_all_commits,
            hide=hide,
        )

        settings = self._settings_provider()

        build_client = get_client_hook()
        client = build_client(_HTTP_TIMEOUT_SECONDS)
        gh = GitHubClient(settings["github_token"], client)
        data = await gh.fetch_user_stats(req["username"])
        await client.aclose()

        # Build stats and render SVG
        user_stats = build_user_stats(
            {
                "login": data["login"],
                "name": data["name"],
                "total_commits": data["total_commits"],
                "total_prs": data["total_prs"],
                "total_issues": data["total_issues"],
                "total_stars": data["total_stars"],
                "total_contributions": data["total_contributions"],
            }
        )

        svg = render_stats_card(
            stats=user_stats,
            theme_name=req["theme"],
            hide_border=req["hide_border"],
            show_icons=req["show_icons"],
            hide=req["hide"],
        )

        cache_ttl = settings["cache_ttl_seconds"]
        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={"Cache-Control": f"max-age={cache_ttl}, s-maxage={cache_ttl}"},
        )

    async def get_top_langs(
        self,
        request: Request,
        username: str | None = Query(default=None, description="GitHub username"),
        theme: str | None = Query(default=None, description="Color theme"),
        hide_border: str | None = Query(default=None, description="Hide border"),
        layout: str | None = Query(default=None, description="Layout style"),
        langs_count: str | None = Query(default=None, description="Number of languages"),
        hide: str | None = Query(default=None, description="Languages to hide (comma-separated)"),
    ) -> Response:
        """Get top languages SVG card.

        Args:
            request: FastAPI request.
            username: GitHub username.
            theme: Color theme name.
            hide_border: Whether to hide border.
            layout: Layout style.
            langs_count: Number of languages to show.
            hide: Comma-separated languages to hide.

        Returns:
            SVG response.
        """
        req = decode_langs_request(
            username=username,
            theme=theme,
            hide_border=hide_border,
            layout=layout,
            langs_count=langs_count,
            hide=hide,
        )

        settings = self._settings_provider()

        build_client = get_client_hook()
        client = build_client(_HTTP_TIMEOUT_SECONDS)
        gh = GitHubClient(settings["github_token"], client)
        data = await gh.fetch_languages(req["username"])
        await client.aclose()

        # Filter hidden languages
        hide_lower = {h.lower() for h in req["hide"]}
        filtered_data = [lang for lang in data if lang["name"].lower() not in hide_lower]

        # Build language stats
        lang_stats, total_size = build_language_stats(
            [
                {"name": lang["name"], "size": lang["size"], "color": lang["color"]}
                for lang in filtered_data
            ]
        )

        svg = render_langs_card(
            username=req["username"],
            languages=lang_stats,
            total_size=total_size,
            theme_name=req["theme"],
            hide_border=req["hide_border"],
            layout=req["layout"],
            langs_count=req["langs_count"],
        )

        cache_ttl = settings["cache_ttl_seconds"]
        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={"Cache-Control": f"max-age={cache_ttl}, s-maxage={cache_ttl}"},
        )


def build_router(settings_provider: _SettingsProvider) -> APIRouter:
    """Build stats router.

    Args:
        settings_provider: Callable that returns settings.

    Returns:
        FastAPI router with stats endpoints.
    """
    router = APIRouter(tags=["stats"])
    handler = _StatsRoutes(settings_provider)

    router.add_api_route(
        "/api",
        handler.get_stats,
        methods=["GET"],
        name="get_stats",
        summary="Get user stats card",
    )
    router.add_api_route(
        "/api/top-langs",
        handler.get_top_langs,
        methods=["GET"],
        name="get_top_langs",
        summary="Get top languages card",
    )

    return router


__all__ = ["build_router"]
