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

DEFAULT_TARGET_URL = "https://tankpit.com/"
"""Canonical tankpit URL used when ``TANKPIT_URL`` is unset or empty."""

_PREFER_ACCOUNT_TRUE_VALUES: tuple[str, ...] = ("true", "1", "yes")


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


def resolve_prefer_account() -> bool:
    """Return True when ``TANKPIT_PREFER_ACCOUNT`` selects account login.

    Returns:
        True when the env var value (case-insensitive) matches any of
        ``"true"``, ``"1"``, or ``"yes"``; False otherwise (including
        when the env var is unset).
    """
    raw = _test_hooks.get_env("TANKPIT_PREFER_ACCOUNT")
    if raw is None:
        return False
    return raw.lower() in _PREFER_ACCOUNT_TRUE_VALUES


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
    "resolve_human_rank_window",
    "resolve_prefer_account",
    "resolve_priority_target",
    "resolve_target_url",
]
