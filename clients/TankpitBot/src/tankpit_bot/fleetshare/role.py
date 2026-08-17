"""Fleet role resolution from the environment.

``TANKPIT_ROLE`` names what this bot does with its ticks — see
:data:`tankpit_bot.fleetshare.types.FleetRole`. Unset means fighter:
the full doctrine is the primary configuration, and a gatherer is an
explicit operator choice.
"""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.fleetshare.types import FLEET_ROLES, FleetRole


def resolve_fleet_role() -> FleetRole:
    """Resolve this bot's fleet role from ``TANKPIT_ROLE``.

    Returns:
        The configured role; ``"fighter"`` when the variable is unset
        or empty.

    Raises:
        ValueError: If ``TANKPIT_ROLE`` is set to an unknown role.
    """
    raw = _test_hooks.get_env("TANKPIT_ROLE")
    if raw is None or raw == "":
        return "fighter"
    for role in FLEET_ROLES:
        if raw == role:
            return role
    raise ValueError(f"TANKPIT_ROLE must be one of {FLEET_ROLES}, got {raw!r}")


__all__ = [
    "resolve_fleet_role",
]
