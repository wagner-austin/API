from __future__ import annotations

from typing import Literal

from platform_core.errors import AppError, ErrorCode

from ..schemas.stats import CapabilitiesRequest, LangsRequest, StatsRequest

_THEMES: frozenset[str] = frozenset(
    {
        "default",
        "dark",
        "dracula",
        "github_dark",
        "transparent",
    }
)

_LAYOUTS: frozenset[str] = frozenset(
    {
        "default",
        "compact",
        "donut",
        "pie",
    }
)

_HIDEABLE_STATS: frozenset[str] = frozenset(
    {
        "stars",
        "commits",
        "prs",
        "issues",
        "contribs",
    }
)


def _require_username(username: str | None) -> str:
    """Require and validate GitHub username.

    Args:
        username: Raw username from query params.

    Returns:
        Validated username string.

    Raises:
        AppError: If username is missing or invalid.
    """
    if username is None or username.strip() == "":
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="username is required",
            http_status=400,
        )
    cleaned = username.strip()
    if len(cleaned) > 39:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="username must be 39 characters or less",
            http_status=400,
        )
    # GitHub usernames: alphanumeric + hyphens, no double hyphens, no leading/trailing hyphen
    for i, ch in enumerate(cleaned):
        if not (ch.isalnum() or ch == "-"):
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                message=f"username contains invalid character at position {i}",
                http_status=400,
            )
    if cleaned.startswith("-") or cleaned.endswith("-"):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="username cannot start or end with hyphen",
            http_status=400,
        )
    if "--" in cleaned:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="username cannot contain consecutive hyphens",
            http_status=400,
        )
    return cleaned


def _narrow_theme(
    raw: str | None,
) -> Literal["default", "dark", "dracula", "github_dark", "transparent"]:
    """Narrow theme string to Literal type.

    Args:
        raw: Raw theme string from query params.

    Returns:
        Validated theme literal.

    Raises:
        AppError: If theme is invalid.
    """
    if raw is None:
        return "default"
    if raw not in _THEMES:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message=f"theme must be one of: {', '.join(sorted(_THEMES))}",
            http_status=400,
        )
    if raw == "dark":
        return "dark"
    if raw == "dracula":
        return "dracula"
    if raw == "github_dark":
        return "github_dark"
    if raw == "transparent":
        return "transparent"
    return "default"


def _narrow_layout(
    raw: str | None,
) -> Literal["default", "compact", "donut", "pie"]:
    """Narrow layout string to Literal type.

    Args:
        raw: Raw layout string from query params.

    Returns:
        Validated layout literal.

    Raises:
        AppError: If layout is invalid.
    """
    if raw is None:
        return "default"
    if raw not in _LAYOUTS:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message=f"layout must be one of: {', '.join(sorted(_LAYOUTS))}",
            http_status=400,
        )
    if raw == "compact":
        return "compact"
    if raw == "donut":
        return "donut"
    if raw == "pie":
        return "pie"
    return "default"


def _parse_query_bool(raw: str | None, default: bool, param_name: str) -> bool:
    """Parse boolean query parameter.

    Args:
        raw: Raw string from query params.
        default: Default value if not provided.
        param_name: Parameter name for error messages.

    Returns:
        Parsed boolean value.

    Raises:
        AppError: If value is invalid.
    """
    if raw is None:
        return default
    lower = raw.lower()
    if lower in ("true", "1", "yes"):
        return True
    if lower in ("false", "0", "no"):
        return False
    raise AppError(
        code=ErrorCode.INVALID_INPUT,
        message=f"{param_name} must be true/false, 1/0, or yes/no",
        http_status=400,
    )


def _parse_hide_list(raw: str | None, allowed: frozenset[str]) -> tuple[str, ...]:
    """Decode comma-separated hide list.

    Args:
        raw: Raw comma-separated string.
        allowed: Set of allowed values.

    Returns:
        Tuple of validated hide values.

    Raises:
        AppError: If any value is invalid.
    """
    if raw is None or raw.strip() == "":
        return ()
    items = [s.strip().lower() for s in raw.split(",") if s.strip()]
    for item in items:
        if item not in allowed:
            allowed_str = ", ".join(sorted(allowed))
            msg = f"invalid hide value '{item}', must be one of: {allowed_str}"
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                message=msg,
                http_status=400,
            )
    return tuple(items)


def _parse_langs_count(raw: str | None) -> int:
    """Decode langs_count parameter.

    Args:
        raw: Raw string from query params.

    Returns:
        Validated integer between 1-20.

    Raises:
        AppError: If value is invalid.
    """
    if raw is None:
        return 8
    stripped = raw.strip()
    if stripped == "":
        return 8
    if not stripped.isdigit() and not (stripped.startswith("-") and stripped[1:].isdigit()):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="langs_count must be an integer",
            http_status=400,
        )
    val = int(stripped)
    if val < 1:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="langs_count must be at least 1",
            http_status=400,
        )
    if val > 20:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="langs_count must be at most 20",
            http_status=400,
        )
    return val


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
        "theme": _narrow_theme(theme),
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
        "theme": _narrow_theme(theme),
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
        "theme": _narrow_theme(theme),
        "hide_border": _parse_query_bool(hide_border, default=False, param_name="hide_border"),
        "disable_animations": _parse_query_bool(
            disable_animations, default=False, param_name="disable_animations"
        ),
    }


__all__ = [
    "decode_capabilities_request",
    "decode_langs_request",
    "decode_stats_request",
]
