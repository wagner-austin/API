"""Tank name registry.

Tracks tank_id -> name from TankRegistry messages. The historical
player_id correlation mapper was deleted 2026-06-19 -- the protocol
0x47 Movement decoder reads tank_id directly per tpclient.js Lg.h, so
the correlation step was both unnecessary and based on wrong bytes
(what was treated as `player_id` at bytes 8-11 is actually lb_score
24-bit BE at 6-8 plus rank at 9).
"""

from __future__ import annotations

# Tank name registry: tank_id -> name (populated from TankRegistry messages)
_tank_names: dict[int, str] = {}


def reset_player_id_mapper() -> None:
    """Reset the tank-name registry for a new session."""
    _tank_names.clear()


def register_tank_name(tank_id: int, name: str) -> None:
    """Register a tank name from TankRegistry message.

    Args:
        tank_id: Tank ID.
        name: Tank name.
    """
    if name:
        _tank_names[tank_id] = name


def get_tank_name(tank_id: int) -> str:
    """Get tank name by ID.

    Args:
        tank_id: Tank ID to look up.

    Returns:
        Tank name, or empty string if not found.
    """
    return _tank_names.get(tank_id, "")


__all__ = [
    "get_tank_name",
    "register_tank_name",
    "reset_player_id_mapper",
]
