from __future__ import annotations

from typing import Protocol

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response
from platform_codebase import parse_github_repo, scan_libs_from_github, scan_services_from_github
from platform_kaggle import build_profile

from ..._test_hooks import get_client_hook, get_github_client_hook
from ...client import GitHubClient
from ...settings import Settings
from ...svg_renderer import (
    build_capabilities_response,
    build_language_stats,
    build_user_stats,
    render_capabilities_card,
    render_hero_card,
    render_langs_card,
    render_stats_card,
)
from ..schemas.stats import Capability
from ..validators.stats import (
    decode_capabilities_request,
    decode_hero_request,
    decode_langs_request,
    decode_stats_request,
)

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
        disable_animations: str | None = Query(default=None, description="Disable animations"),
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
            disable_animations: Whether to disable CSS animations.

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
            disable_animations=disable_animations,
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
            disable_animations=req["disable_animations"],
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
        disable_animations: str | None = Query(default=None, description="Disable animations"),
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
            disable_animations: Whether to disable CSS animations.

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
            disable_animations=disable_animations,
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
            disable_animations=req["disable_animations"],
        )

        cache_ttl = settings["cache_ttl_seconds"]
        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={"Cache-Control": f"max-age={cache_ttl}, s-maxage={cache_ttl}"},
        )

    def get_capabilities(
        self,
        request: Request,
        repo: str | None = Query(default=None, description="GitHub repo (owner/repo)"),
        theme: str | None = Query(default=None, description="Color theme"),
        hide_border: str | None = Query(default=None, description="Hide border"),
        disable_animations: str | None = Query(default=None, description="Disable animations"),
    ) -> Response:
        """Get codebase capabilities SVG card.

        Args:
            request: FastAPI request.
            repo: GitHub repository in owner/repo format.
            theme: Color theme name.
            hide_border: Whether to hide border.
            disable_animations: Whether to disable CSS animations.

        Returns:
            SVG response.
        """
        req = decode_capabilities_request(
            repo=repo,
            theme=theme,
            hide_border=hide_border,
            disable_animations=disable_animations,
        )

        settings = self._settings_provider()

        # Parse repo and scan via GitHub API
        owner, repo_name = parse_github_repo(req["repo"])

        build_github_client = get_github_client_hook()
        github_client = build_github_client(settings["github_token"])

        libs = scan_libs_from_github(github_client, owner, repo_name)
        services = scan_services_from_github(github_client, owner, repo_name)

        # Build profile using platform_kaggle
        profile = build_profile(libs, services)

        # Convert CodebaseCapability objects to our Capability TypedDict
        capabilities: list[Capability] = []
        for cap in profile.capabilities:
            capabilities.append(
                {
                    "name": cap.name,
                    "strength": cap.strength,
                    "tags": cap.tags,
                    "description": cap.description,
                }
            )

        response = build_capabilities_response(
            repo=req["repo"],
            capabilities=tuple(capabilities),
            ml_backends=profile.ml_backends,
            frameworks=profile.frameworks,
            data_formats=profile.data_formats,
            task_types=profile.task_types,
        )

        svg = render_capabilities_card(
            response=response,
            theme_name=req["theme"],
            hide_border=req["hide_border"],
            disable_animations=req["disable_animations"],
        )

        cache_ttl = settings["cache_ttl_seconds"]
        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={"Cache-Control": f"max-age={cache_ttl}, s-maxage={cache_ttl}"},
        )

    def get_hero(
        self,
        request: Request,
        name: str | None = Query(default=None, description="Display name"),
        subtitle: str | None = Query(default=None, description="Subtitle text"),
        lines: str | None = Query(default=None, description="Info lines (pipe-separated)"),
        theme: str | None = Query(default=None, description="Color theme"),
        disable_animations: str | None = Query(default=None, description="Disable animations"),
    ) -> Response:
        """Get hero card SVG with rain animation.

        Args:
            request: FastAPI request.
            name: Display name (large title).
            subtitle: Subtitle text.
            lines: Pipe-separated info lines.
            theme: Color theme name.
            disable_animations: Whether to disable CSS animations.

        Returns:
            SVG response.
        """
        req = decode_hero_request(
            name=name,
            subtitle=subtitle,
            lines=lines,
            theme=theme,
            disable_animations=disable_animations,
        )

        settings = self._settings_provider()

        svg = render_hero_card(
            name=req["name"],
            subtitle=req["subtitle"],
            lines=req["lines"],
            theme_name=req["theme"],
            disable_animations=req["disable_animations"],
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
    router.add_api_route(
        "/api/capabilities",
        handler.get_capabilities,
        methods=["GET"],
        name="get_capabilities",
        summary="Get codebase capabilities card",
    )
    router.add_api_route(
        "/api/hero",
        handler.get_hero,
        methods=["GET"],
        name="get_hero",
        summary="Get hero card with rain animation",
    )

    return router


__all__ = ["build_router"]
