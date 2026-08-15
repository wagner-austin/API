"""stats: decode_stats_request and related definitions."""

from __future__ import annotations

from platform_core.errors import AppError, ErrorCode

from github_stats_api.api.validators._stats_params import (
    _HIDEABLE_STATS,
    _narrow_layout,
    _parse_hide_list,
    _parse_langs_count,
    _parse_query_bool,
    _require_theme,
    _require_username,
)

from ..schemas.stats import (
    CapabilitiesRequest,
    HeroRequest,
    LangsRequest,
    SkillsRequest,
    StatsRequest,
)


def decode_stats_request(
    username: str | None,
    theme: str | None,
    hide_border: str | None,
    show_icons: str | None,
    include_all_commits: str | None,
    hide: str | None,
    disable_animations: str | None,
) -> StatsRequest:
    """Decode and validate stats request from query parameters.

    Args:
        username: GitHub username.
        theme: Color theme.
        hide_border: Whether to hide border.
        show_icons: Whether to show icons.
        include_all_commits: Include all commits.
        hide: Comma-separated list of stats to hide.
        disable_animations: Whether to disable CSS animations.

    Returns:
        Validated StatsRequest TypedDict.

    Raises:
        AppError: If validation fails.
    """
    return {
        "username": _require_username(username),
        "theme": _require_theme(theme),
        "hide_border": _parse_query_bool(hide_border, default=False, param_name="hide_border"),
        "show_icons": _parse_query_bool(show_icons, default=True, param_name="show_icons"),
        "include_all_commits": _parse_query_bool(
            include_all_commits, default=False, param_name="include_all_commits"
        ),
        "hide": _parse_hide_list(hide, _HIDEABLE_STATS),
        "disable_animations": _parse_query_bool(
            disable_animations, default=False, param_name="disable_animations"
        ),
    }


def decode_langs_request(
    username: str | None,
    theme: str | None,
    hide_border: str | None,
    layout: str | None,
    langs_count: str | None,
    hide: str | None,
    disable_animations: str | None,
) -> LangsRequest:
    """Decode and validate langs request from query parameters.

    Args:
        username: GitHub username.
        theme: Color theme.
        hide_border: Whether to hide border.
        layout: Layout style.
        langs_count: Number of languages to show.
        hide: Comma-separated list of languages to hide.
        disable_animations: Whether to disable CSS animations.

    Returns:
        Validated LangsRequest TypedDict.

    Raises:
        AppError: If validation fails.
    """
    hide_tuple = tuple(s.strip() for s in (hide or "").split(",") if s.strip())
    return {
        "username": _require_username(username),
        "theme": _require_theme(theme),
        "hide_border": _parse_query_bool(hide_border, default=False, param_name="hide_border"),
        "layout": _narrow_layout(layout),
        "langs_count": _parse_langs_count(langs_count),
        "hide": hide_tuple,
        "disable_animations": _parse_query_bool(
            disable_animations, default=False, param_name="disable_animations"
        ),
    }


def _require_repo(repo: str | None) -> str:
    """Require and validate GitHub repository in owner/repo format.

    Args:
        repo: Raw repo string from query params.

    Returns:
        Validated repo string in 'owner/repo' format.

    Raises:
        AppError: If repo is missing or invalid format.
    """
    if repo is None or repo.strip() == "":
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="repo is required (format: owner/repo)",
            http_status=400,
        )
    cleaned = repo.strip()
    parts = cleaned.split("/")
    if len(parts) != 2:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="repo must be in format 'owner/repo'",
            http_status=400,
        )
    owner, repo_name = parts
    if owner == "" or repo_name == "":
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="repo must be in format 'owner/repo'",
            http_status=400,
        )
    # Validate owner (same rules as username)
    if len(owner) > 39:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="repo owner must be 39 characters or less",
            http_status=400,
        )
    for i, ch in enumerate(owner):
        if not (ch.isalnum() or ch == "-"):
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                message=f"repo owner contains invalid character at position {i}",
                http_status=400,
            )
    # Validate repo name (alphanumeric, hyphens, underscores, dots)
    if len(repo_name) > 100:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="repo name must be 100 characters or less",
            http_status=400,
        )
    for i, ch in enumerate(repo_name):
        if not (ch.isalnum() or ch in "-_."):
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                message=f"repo name contains invalid character at position {i}",
                http_status=400,
            )
    return cleaned


def decode_capabilities_request(
    repo: str | None,
    theme: str | None,
    hide_border: str | None,
    disable_animations: str | None,
) -> CapabilitiesRequest:
    """Decode and validate capabilities request from query parameters.

    Args:
        repo: GitHub repository in 'owner/repo' format.
        theme: Color theme.
        hide_border: Whether to hide border.
        disable_animations: Whether to disable CSS animations.

    Returns:
        Validated CapabilitiesRequest TypedDict.

    Raises:
        AppError: If validation fails.
    """
    return {
        "repo": _require_repo(repo),
        "theme": _require_theme(theme),
        "hide_border": _parse_query_bool(hide_border, default=False, param_name="hide_border"),
        "disable_animations": _parse_query_bool(
            disable_animations, default=False, param_name="disable_animations"
        ),
    }


def _parse_hero_lines(raw: str | None) -> tuple[str, ...]:
    """Parse pipe-separated lines for hero card.

    Args:
        raw: Raw pipe-separated string of lines.

    Returns:
        Tuple of line strings (can be empty).
    """
    if raw is None or raw.strip() == "":
        return ()
    items = [s.strip() for s in raw.split("|") if s.strip()]
    if len(items) > 8:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="lines must contain at most 8 lines",
            http_status=400,
        )
    for i, line in enumerate(items):
        if len(line) > 80:
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                message=f"line {i + 1} exceeds 80 characters",
                http_status=400,
            )
    return tuple(items)


def decode_hero_request(
    name: str | None,
    subtitle: str | None,
    lines: str | None,
    theme: str | None,
    disable_animations: str | None,
) -> HeroRequest:
    """Decode and validate hero request from query parameters.

    Args:
        name: Display name (required).
        subtitle: Subtitle text below name.
        lines: Pipe-separated list of info lines.
        theme: Color theme.
        disable_animations: Whether to disable CSS animations.

    Returns:
        Validated HeroRequest TypedDict.

    Raises:
        AppError: If validation fails.
    """
    if name is None or name.strip() == "":
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="name is required",
            http_status=400,
        )
    parsed_name = name.strip()
    if len(parsed_name) > 40:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="name must be 40 characters or less",
            http_status=400,
        )
    parsed_subtitle = "" if subtitle is None else subtitle.strip()
    if len(parsed_subtitle) > 80:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="subtitle must be 80 characters or less",
            http_status=400,
        )
    return {
        "name": parsed_name,
        "subtitle": parsed_subtitle,
        "lines": _parse_hero_lines(lines),
        "theme": _require_theme(theme),
        "disable_animations": _parse_query_bool(
            disable_animations, default=False, param_name="disable_animations"
        ),
    }


def _parse_skills(raw: str | None) -> tuple[str, ...]:
    """Parse comma-separated skills list.

    Args:
        raw: Raw comma-separated string of skills.

    Returns:
        Tuple of skill names.

    Raises:
        AppError: If skills is empty or too many skills.
    """
    if raw is None or raw.strip() == "":
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="skills is required",
            http_status=400,
        )
    items = [s.strip() for s in raw.split(",") if s.strip()]
    if len(items) == 0:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="skills is required",
            http_status=400,
        )
    if len(items) > 20:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="skills must contain at most 20 items",
            http_status=400,
        )
    for i, skill in enumerate(items):
        if len(skill) > 30:
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                message=f"skill {i + 1} exceeds 30 characters",
                http_status=400,
            )
    return tuple(items)


def decode_skills_request(
    skills: str | None,
    theme: str | None,
    hide_border: str | None,
    disable_animations: str | None,
) -> SkillsRequest:
    """Decode and validate skills request from query parameters.

    Args:
        skills: Comma-separated list of skill names.
        theme: Color theme.
        hide_border: Whether to hide border.
        disable_animations: Whether to disable CSS animations.

    Returns:
        Validated SkillsRequest TypedDict.

    Raises:
        AppError: If validation fails.
    """
    return {
        "skills": _parse_skills(skills),
        "theme": _require_theme(theme),
        "hide_border": _parse_query_bool(hide_border, default=False, param_name="hide_border"),
        "disable_animations": _parse_query_bool(
            disable_animations, default=False, param_name="disable_animations"
        ),
    }
