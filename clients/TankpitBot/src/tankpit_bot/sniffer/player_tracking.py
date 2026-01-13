"""Player ID mapping and tank name tracking.

This module correlates Movement player_id values to tank_id values
and maintains a registry of tank names from TankRegistry messages.
"""

from __future__ import annotations

from tankpit_bot.container import PlayerIdMapper

# Module-level mapper for resolving Movement player_id -> tank_id
_player_id_mapper = PlayerIdMapper()

# Tank name registry: tank_id -> name (populated from TankRegistry messages)
_tank_names: dict[int, str] = {}


def reset_player_id_mapper() -> None:
    """Reset mapper for new session (used by tests)."""
    _player_id_mapper.clear()
    _tank_names.clear()


def resolve_movement_tank(pid: int, sx: int, sy: int) -> str:
    """Resolve tank identifier string for a Movement message.

    Tries to resolve player_id to tank_id via:
    1. Cached player_id -> tank_id mapping
    2. Position correlation with recent PositionUpdate data

    Args:
        pid: Player ID from Movement message.
        sx: Start X coordinate.
        sy: Start Y coordinate.

    Returns:
        Tank identifier string: '"TankName"', 'tank=ID', or 'pid=ID'.
    """
    resolved_tid = _player_id_mapper.get_tank_id(pid)
    if resolved_tid is None:
        # Try position correlation
        pos_tid = _player_id_mapper._position_to_tank.get((sx, sy))
        if pos_tid is not None:
            _player_id_mapper._player_to_tank[pid] = pos_tid
            resolved_tid = pos_tid
    if resolved_tid is not None:
        name = _tank_names.get(resolved_tid, "")
        return f'"{name}"' if name else f"tank={resolved_tid}"
    return f"pid={pid}"


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


def record_movement_response(tank_id: int, x: int, y: int) -> None:
    """Record position for tank ID correlation.

    Args:
        tank_id: Tank ID from MovementResponse.
        x: X coordinate.
        y: Y coordinate.
    """
    _player_id_mapper.record_movement_response(tank_id=tank_id, x=x, y=y)


__all__ = [
    "get_tank_name",
    "record_movement_response",
    "register_tank_name",
    "reset_player_id_mapper",
    "resolve_movement_tank",
]
