"""stats: _THEMES and related definitions."""

from __future__ import annotations

from typing import Literal

from platform_core.errors import AppError, ErrorCode

from ..schemas.stats import (
    ThemeName,
)

_THEMES: frozenset[str] = frozenset(
    {
        "default",
        "dark",
        "dracula",
        "github_dark",
        "transparent",
        "cyberpunk",
        "synthwave",
        "neon",
        "aurora",
        "radical",
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


def _require_theme(raw: str | None) -> ThemeName:
    """Validate and narrow theme string to ThemeName Literal type.

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
    return _narrow_theme_unchecked(raw)


def _narrow_theme_unchecked(raw: str) -> ThemeName:
    """Narrow validated theme string to ThemeName Literal.

    This function assumes the theme has already been validated against _THEMES.

    Args:
        raw: Validated theme string.

    Returns:
        Theme literal.
    """
    if raw == "dark":
        return "dark"
    if raw == "dracula":
        return "dracula"
    if raw == "github_dark":
        return "github_dark"
    if raw == "transparent":
        return "transparent"
    if raw == "cyberpunk":
        return "cyberpunk"
    if raw == "synthwave":
        return "synthwave"
    if raw == "neon":
        return "neon"
    if raw == "aurora":
        return "aurora"
    if raw == "radical":
        return "radical"
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
