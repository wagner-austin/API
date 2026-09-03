"""Bot-launch configuration resolved from the process environment.

Two settings — the tankpit target URL and the guest-vs-account login
preference — need to be read the same way from every code path that
launches a :class:`Bot`. Centralising the resolvers here keeps
:mod:`tankpit_bot.bot.entry` (one-shot ``tankpit-bot`` CLI) and
:mod:`tankpit_bot.service.service_main` (long-running
``tankpit-bot-service``) in lockstep — an env-var-handling difference
between the two used to be a silent divergence risk.

Both resolvers read through :func:`tankpit_bot._test_hooks.get_env`, so
tests inject deterministic values without touching the real process
environment.
"""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.bot.ai.types import AIConfigDict, make_default_ai_config
from tankpit_bot.fleetshare.role import resolve_engagement_doctrine, resolve_fleet_role

DEFAULT_TARGET_URL = "https://tankpit.com/"
"""Canonical tankpit URL used when ``TANKPIT_URL`` is unset or empty."""

_TRUTHY_VALUES: tuple[str, ...] = ("true", "1", "yes")
"""Accepted spellings of "yes" for every boolean env var read here.

Shared rather than duplicated per setting: two copies drift, and an
operator who learns that ``TANKPIT_PREFER_ACCOUNT=yes`` works is
entitled to have ``TANKPIT_HEADLESS=yes`` work the same way.
"""


def resolve_target_url() -> str:
    """Return the tankpit target URL, honouring the ``TANKPIT_URL`` env var.

    Returns:
        The env override when set to a non-empty value; otherwise the
        canonical :data:`DEFAULT_TARGET_URL`.
    """
    override = _test_hooks.get_env("TANKPIT_URL")
    if override is not None and override != "":
        return override
    return DEFAULT_TARGET_URL


def resolve_env_flag(name: str) -> bool:
    """Return True when the named environment variable spells an affirmative.

    One parser for every boolean setting in the process. The named
    resolvers below bind a variable to a meaning; this binds a spelling
    to a truth value, and there is exactly one of it so an operator who
    learns that ``yes`` works for one flag can rely on it for all of
    them.

    Args:
        name: Environment variable to read.

    Returns:
        True when the value (case-insensitive) is one of ``"true"``,
        ``"1"`` or ``"yes"``; False otherwise, including when unset.
    """
    raw = _test_hooks.get_env(name)
    if raw is None:
        return False
    return raw.lower() in _TRUTHY_VALUES


def resolve_prefer_account() -> bool:
    """Return True when ``TANKPIT_PREFER_ACCOUNT`` selects account login.

    Returns:
        True when the env var value (case-insensitive) matches any of
        ``"true"``, ``"1"``, or ``"yes"``; False otherwise (including
        when the env var is unset).
    """
    return resolve_env_flag("TANKPIT_PREFER_ACCOUNT")


def resolve_headless() -> bool:
    """Return True when ``TANKPIT_HEADLESS`` asks for a windowless browser.

    **Defaults to False, deliberately.** On a desktop the operator runs
    the fleet headed in order to watch the tanks play, and that is the
    point of running it there; a default of headless would take the
    window away from the machine whose whole job is showing it.

    A container is the case that needs the other answer. There is no X
    server in one, so a headed launch dies immediately with "Missing X
    server or $DISPLAY" and the child exits 1 about five seconds after
    spawn. ``docker-compose.yml`` has set ``TANKPIT_HEADLESS: "true"``
    all along; until this resolver existed nothing on the bot path read
    it, so the flag looked like the answer while being inert, and every
    containerized fleet bot died on launch.

    Returns:
        True when the env var value (case-insensitive) matches any of
        ``"true"``, ``"1"``, or ``"yes"``; False otherwise, including
        when the env var is unset.
    """
    return resolve_env_flag("TANKPIT_HEADLESS")


def resolve_human_rank_window() -> tuple[int, int]:
    """Return the targetable human rank window from the environment.

    ``TANKPIT_BOT_HUMAN_MIN_RANK`` / ``TANKPIT_BOT_HUMAN_MAX_RANK``
    bound which human ranks the bot may engage (integers, 0 recruit ..
    8 general). Defaults ``(1, 8)``: recruits protected, no ceiling.
    A main-map bot can raise the floor (lieutenant+ = 4) or lower the
    ceiling to leave high ranks alone (user doctrine 2026-07-28).
    Practice bots are farmed at any rank regardless of this window.

    Returns:
        ``(min_rank, max_rank)``.

    Raises:
        ValueError: If either value is not an integer.
    """
    raw_min = _test_hooks.get_env("TANKPIT_BOT_HUMAN_MIN_RANK")
    raw_max = _test_hooks.get_env("TANKPIT_BOT_HUMAN_MAX_RANK")
    return (
        int(raw_min) if raw_min is not None else 1,
        int(raw_max) if raw_max is not None else 8,
    )


def resolve_video_fps() -> float:
    """Return the live-view capture rate from ``TANKPIT_BOT_VIDEO_FPS``.

    Default 12 fps — steady motion for phone monitoring at roughly
    0.5-1 MB/s of JPEG through the tunnel (2026-07-29 page-push live
    view, [[bot-service-architecture]]).

    Returns:
        Frames per second the in-page caster targets.

    Raises:
        ValueError: If the env value is set but not a number.
    """
    raw = _test_hooks.get_env("TANKPIT_BOT_VIDEO_FPS")
    return float(raw) if raw is not None else 12.0


def resolve_video_quality() -> float:
    """Return the live-view JPEG quality from ``TANKPIT_BOT_VIDEO_QUALITY``.

    Default 0.8 — visually clean on the composited game canvases
    without ballooning per-frame size.

    Returns:
        JPEG quality in (0, 1] for the page's ``toDataURL``.

    Raises:
        ValueError: If the env value is set but not a number.
    """
    raw = _test_hooks.get_env("TANKPIT_BOT_VIDEO_QUALITY")
    return float(raw) if raw is not None else 0.8


def resolve_priority_target() -> str:
    """Return the priority hunt account from ``TANKPIT_BOT_PRIORITY_TARGET``.

    The named account outranks every other target at acquisition
    (case-insensitive; humans already outrank practice bots without
    any configuration -- [[bot-behavior-contract]] §3.2).

    Returns:
        The configured account name, or ``""`` when unset/empty.
    """
    raw = _test_hooks.get_env("TANKPIT_BOT_PRIORITY_TARGET")
    return raw if raw is not None else ""


__all__ = [
    "DEFAULT_TARGET_URL",
    "env_ai_config",
    "resolve_env_flag",
    "resolve_headless",
    "resolve_human_rank_window",
    "resolve_prefer_account",
    "resolve_priority_target",
    "resolve_target_url",
    "resolve_video_fps",
    "resolve_video_quality",
]


def resolve_weapon_resume_slack() -> int:
    """Return the weapons resume slack from ``TANKPIT_BOT_WEAPON_RESUME_SLACK``.

    Default 0 preserves the 2026-07-25 contract verbatim: HUNT entry
    requires duals and homings at exactly ``inventory_capacity(rank)``
    ("the bot never hunts below full stock, no exceptions"). Setting a
    positive slack N relaxes the resume bar to ``cap - N`` -- the same
    shape as the radar rule's fixed cap-5 floor -- because equipment
    has no map atlas, so every between-kill top-off to the exact cap
    forces a hop-scan discovery loop (the 2026-07-29 session where the
    user flagged that loop nine times from the HUD). The weapon
    EMERGENCY break (<4) is unaffected; this only moves the resume
    bar.

    Returns:
        Non-negative slack subtracted from the weapons resume cap.

    Raises:
        ValueError: If the env value is set but not a non-negative int.
    """
    raw = _test_hooks.get_env("TANKPIT_BOT_WEAPON_RESUME_SLACK")
    if raw is None:
        return 0
    value = int(raw)
    if value < 0:
        raise ValueError(f"TANKPIT_BOT_WEAPON_RESUME_SLACK must be >= 0, got {value}")
    return value


def env_ai_config() -> AIConfigDict:
    """Build the session AI config with environment overrides applied.

    Overrides: ``TANKPIT_BOT_PRIORITY_TARGET`` (the priority hunt
    account) and ``TANKPIT_BOT_HUMAN_MIN_RANK`` /
    ``TANKPIT_BOT_HUMAN_MAX_RANK`` (the targetable human rank window)
    -- [[bot-behavior-contract]] §3.2.

    Lives here rather than in ``bot/base.py`` because it is env-resolved
    launch configuration like every other resolver in this module, and
    because both ``base`` and ``run_session`` need it — keeping it in
    ``base`` forced the run loop to import the class that imports it.

    Returns:
        Default AI config with env-resolved fields filled in.
    """
    min_rank, max_rank = resolve_human_rank_window()
    return AIConfigDict(
        **{
            **make_default_ai_config(),
            "priority_target_name": resolve_priority_target(),
            "human_target_min_rank": min_rank,
            "human_target_max_rank": max_rank,
            "role": resolve_fleet_role(),
            "doctrine": resolve_engagement_doctrine(),
        }
    )
