"""TankObservation TypedDict + factory + encode/decode.

A TankObservation is a single, immutable record describing what a wire
or map-snapshot message proved about one tank at one instant. The
``apply_tank_observation`` mutator merges it into ``WorldStateDict`` and
advances exactly the freshness timestamps the observation actually
proves -- never more.

Three timestamps live on every tank (see ``TankStateDict`` docstring for
the full freshness model):

* ``timestamp_ms`` -- advances on EVERY observation.
* ``last_wire_seen_ms`` -- advances only when ``is_wire_sourced`` is True.
* ``last_position_update_ms`` -- advances only when ``is_wire_sourced``
  is True AND ``position`` is not None.

These rules are enforced inside the mutator and locked by tests in
``tests/state/test_tank_observation.py``. They exist because wire
broadcasts decouple status-only updates (TankStatusSync, every 2 s
globally for every active tank) from position-bearing updates (0x3D
MovementResponse, only for tanks the server thinks the bot can see).
Conflating the two -- the historical bug -- meant the bot kept firing
at stale registry positions while the status broadcast lied that the
registry was fresh.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_bool,
    require_int,
)
from typing_extensions import TypedDict

from tankpit_bot.state.types.constants import EntitySource, require_entity_source


class TankObservation(TypedDict):
    """A single observation event for a tank.

    Every field is required (Required-by-default semantics). Optional
    aspects (position, team, rank, damage_state, direction, name,
    is_bot) use explicit ``None`` to signal "this observation does not
    update this aspect"; ``apply_tank_observation`` preserves the
    existing tank's value when the aspect is None.

    Attributes:
        tank_id: Subject of the observation.
        timestamp_ms: Wall-clock ms when the observation arrived.
        is_wire_sourced: True if this observation came from a wire
            message (any 0x2x/0x3x/0x4x/0x5x/etc. binary message); False
            if it came from the map snapshot (WorldState blob 0x4C).
            Drives ``last_wire_seen_ms`` advancement and gates
            ``last_position_update_ms`` advancement.
        storage_source: Which ``EntitySource`` label to record on the
            tank when this observation creates or refreshes it. The
            label is independent from ``is_wire_sourced`` because
            multiple wire sources resolve to the same storage label.
        position: Fresh ``(x, y)`` if this observation proves position,
            else ``None``. None preserves the tank's existing position.
        team: Fresh team if observed, else ``None``.
        rank: Fresh rank if observed, else ``None``.
        damage_state: Fresh damage tier if observed, else ``None``.
        direction: Fresh direction byte if observed, else ``None``.
            ``None`` preserves the existing direction.
        name: Fresh player name if observed, else ``None``.
        is_bot: Fresh bot flag if observed, else ``None``.
    """

    tank_id: int
    timestamp_ms: int
    is_wire_sourced: bool
    storage_source: EntitySource
    position: tuple[int, int] | None
    team: int | None
    rank: int | None
    damage_state: int | None
    direction: int | None
    name: str | None
    is_bot: bool | None


def make_tank_observation(
    tank_id: int,
    timestamp_ms: int,
    is_wire_sourced: bool,
    storage_source: EntitySource,
    *,
    position: tuple[int, int] | None = None,
    team: int | None = None,
    rank: int | None = None,
    damage_state: int | None = None,
    direction: int | None = None,
    name: str | None = None,
    is_bot: bool | None = None,
) -> TankObservation:
    """Build an immutable ``TankObservation``.

    Args:
        tank_id: Subject of the observation.
        timestamp_ms: Wall-clock ms when the observation arrived.
        is_wire_sourced: True if from a wire message; False for map
            snapshot.
        storage_source: Which ``EntitySource`` label to record.
        position: Fresh ``(x, y)`` if observed, else ``None``.
        team: Fresh team if observed.
        rank: Fresh rank if observed.
        damage_state: Fresh damage tier if observed.
        direction: Fresh direction byte if observed.
        name: Fresh player name if observed.
        is_bot: Fresh bot flag if observed.

    Returns:
        Constructed ``TankObservation`` with the supplied fields.
    """
    return TankObservation(
        tank_id=tank_id,
        timestamp_ms=timestamp_ms,
        is_wire_sourced=is_wire_sourced,
        storage_source=storage_source,
        position=position,
        team=team,
        rank=rank,
        damage_state=damage_state,
        direction=direction,
        name=name,
        is_bot=is_bot,
    )


def _require_position(data: JSONObject, key: str) -> tuple[int, int] | None:
    """Validate and extract an optional ``(x, y)`` tuple from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        The decoded tuple, or ``None`` if the JSON field is null.

    Raises:
        JSONTypeError: If the field is present but malformed.
    """
    raw = data.get(key)
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != 2:
        raise JSONTypeError(f"{key} must be a 2-element list, got {raw!r}")
    x_raw, y_raw = raw[0], raw[1]
    if not isinstance(x_raw, int) or isinstance(x_raw, bool):
        raise JSONTypeError(f"{key}[0] must be int, got {type(x_raw).__name__}")
    if not isinstance(y_raw, int) or isinstance(y_raw, bool):
        raise JSONTypeError(f"{key}[1] must be int, got {type(y_raw).__name__}")
    return (x_raw, y_raw)


def _require_optional_int(data: JSONObject, key: str) -> int | None:
    """Validate and extract an optional integer from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        The decoded int, or ``None`` if the JSON field is null.

    Raises:
        JSONTypeError: If the field is present but not an int.
    """
    raw = data.get(key)
    if raw is None:
        return None
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise JSONTypeError(f"{key} must be int, got {type(raw).__name__}")
    return raw


def _require_optional_str(data: JSONObject, key: str) -> str | None:
    """Validate and extract an optional string from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        The decoded string, or ``None`` if the JSON field is null.

    Raises:
        JSONTypeError: If the field is present but not a string.
    """
    raw = data.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise JSONTypeError(f"{key} must be str, got {type(raw).__name__}")
    return raw


def _require_optional_bool(data: JSONObject, key: str) -> bool | None:
    """Validate and extract an optional bool from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        The decoded bool, or ``None`` if the JSON field is null.

    Raises:
        JSONTypeError: If the field is present but not a bool.
    """
    raw = data.get(key)
    if raw is None:
        return None
    if not isinstance(raw, bool):
        raise JSONTypeError(f"{key} must be bool, got {type(raw).__name__}")
    return raw


def encode_tank_observation(obs: TankObservation) -> JSONObject:
    """Encode a ``TankObservation`` to a JSON-serializable dict.

    Args:
        obs: Observation to encode.

    Returns:
        JSON-serializable representation. ``None`` fields encode as JSON
        null; ``position`` encodes as a 2-element list.
    """
    pos = obs["position"]
    return {
        "tank_id": obs["tank_id"],
        "timestamp_ms": obs["timestamp_ms"],
        "is_wire_sourced": obs["is_wire_sourced"],
        "storage_source": obs["storage_source"],
        "position": [pos[0], pos[1]] if pos is not None else None,
        "team": obs["team"],
        "rank": obs["rank"],
        "damage_state": obs["damage_state"],
        "direction": obs["direction"],
        "name": obs["name"],
        "is_bot": obs["is_bot"],
    }


def decode_tank_observation(data: JSONObject) -> TankObservation:
    """Decode a ``TankObservation`` from JSON with full validation.

    Args:
        data: JSON object produced by ``encode_tank_observation``.

    Returns:
        Validated ``TankObservation``.

    Raises:
        JSONTypeError: If required fields are missing or any field is
            present with the wrong type.
    """
    # Validate storage_source via the shared EntitySource decoder, which
    # raises JSONTypeError for unsupported values.
    storage_source: EntitySource = require_entity_source(data, "storage_source")
    return TankObservation(
        tank_id=require_int(data, "tank_id"),
        timestamp_ms=require_int(data, "timestamp_ms"),
        is_wire_sourced=require_bool(data, "is_wire_sourced"),
        storage_source=storage_source,
        position=_require_position(data, "position"),
        team=_require_optional_int(data, "team"),
        rank=_require_optional_int(data, "rank"),
        damage_state=_require_optional_int(data, "damage_state"),
        direction=_require_optional_int(data, "direction"),
        name=_require_optional_str(data, "name"),
        is_bot=_require_optional_bool(data, "is_bot"),
    )


__all__ = [
    "TankObservation",
    "decode_tank_observation",
    "encode_tank_observation",
    "make_tank_observation",
]
