"""Fleet role resolution from the environment.

``TANKPIT_ROLE`` names what this bot does with its ticks — see
:data:`tankpit_bot.fleetshare.types.FleetRole`. Unset means fighter:
the full doctrine is the primary configuration, and a gatherer is an
explicit operator choice.
"""

from __future__ import annotations

from platform_core.members import as_member

from tankpit_bot import _test_hooks
from tankpit_bot.fleetshare.types import EngagementDoctrine, FleetRole


def resolve_fleet_role() -> FleetRole:
    """Resolve this bot's fleet role from ``TANKPIT_ROLE``.

    Returns:
        The configured role; ``FleetRole.FIGHTER`` when the variable is
        unset or empty.

    Raises:
        JSONTypeError: If ``TANKPIT_ROLE`` is set to an unknown role; the
            message names the variable, the word and every known role.
    """
    raw = _test_hooks.get_env("TANKPIT_ROLE")
    if raw is None or raw == "":
        return FleetRole.FIGHTER
    return as_member(raw, "TANKPIT_ROLE", FleetRole)


def resolve_engagement_doctrine() -> EngagementDoctrine:
    """Resolve this bot's engagement doctrine from ``TANKPIT_DOCTRINE``.

    Returns:
        The configured doctrine; ``EngagementDoctrine.SKIRMISH`` (today's
        behavior) when the variable is unset or empty.

    Raises:
        JSONTypeError: If ``TANKPIT_DOCTRINE`` is set to an unknown
            doctrine; the message names the variable, the word and every
            known doctrine.
    """
    raw = _test_hooks.get_env("TANKPIT_DOCTRINE")
    if raw is None or raw == "":
        return EngagementDoctrine.SKIRMISH
    return as_member(raw, "TANKPIT_DOCTRINE", EngagementDoctrine)


__all__ = [
    "resolve_engagement_doctrine",
    "resolve_fleet_role",
]
