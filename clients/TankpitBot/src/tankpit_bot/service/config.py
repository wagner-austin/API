"""Environment-resolved service configuration.

The service's own knobs, resolved from the environment. This lived in
``bot/config.py`` and reached back into ``service.constants`` through a
function-level import -- the deferred import was the tell that the
function was in the wrong package.
"""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.bot.config import resolve_env_flag
from tankpit_bot.service.constants import SERVICE_IDLE_EXIT_SECONDS


def resolve_idle_exit_seconds() -> float:
    """Return the service idle self-exit threshold from the environment.

    ``TANKPIT_BOT_SERVICE_IDLE_EXIT_SECONDS`` overrides the default
    idle window (:data:`~tankpit_bot.service.constants.SERVICE_IDLE_EXIT_SECONDS`,
    1800 s). ``0`` (or any non-positive value) DISABLES the idle
    self-exit entirely — the always-on deployment mode (2026-07-29):
    with the SPA's tankpit video served by this service, the phone
    expects the URL to answer at any hour, so the startup launcher
    runs the service with the exit disabled.

    Returns:
        The idle threshold in seconds; non-positive means disabled.

    Raises:
        ValueError: If the env value is set but not a number.
    """
    raw = _test_hooks.get_env("TANKPIT_BOT_SERVICE_IDLE_EXIT_SECONDS")
    return float(raw) if raw is not None else SERVICE_IDLE_EXIT_SECONDS


def resolve_autostart() -> bool:
    """Return True when the service should run one session on boot.

    The standalone ``tankpit-bot-service`` idles until ``POST /start``,
    which is right for the phone flow: the operator decides when a tank
    goes out. A FLEET CHILD is the opposite case — the manager already
    decided by spawning it, and nobody is going to call ``/start`` on a
    process that exists solely to play one bounded session.

    So this is set only by :func:`~tankpit_bot.service.fleet_bot._child_environment`,
    and it is an explicit flag rather than an inference from the session
    bounds being present. Inferring it would silently change what an
    existing ``tankpit-bot-service`` does the moment someone exported
    ``TANKPIT_BOT_SESSION_SECONDS`` in their shell.

    Returns:
        True when ``TANKPIT_BOT_AUTOSTART`` spells an affirmative;
        False otherwise, including when unset.
    """
    return resolve_env_flag("TANKPIT_BOT_AUTOSTART")


__all__ = ["resolve_autostart", "resolve_idle_exit_seconds"]
