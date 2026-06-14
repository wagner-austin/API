"""Self-tank state TypedDict + factory + encode/decode."""

from __future__ import annotations

from platform_core.json_utils import JSONObject, require_int
from typing_extensions import TypedDict


class SelfStateDict(TypedDict):
    """State of the player's own tank.

    Attributes:
        tank_id: Player's tank ID.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        team: Team ID (0-3).
        rank: Military rank (0-7).
        fuel: Current fuel (also health).
        leaderboard_position: Position on leaderboard.
    """

    tank_id: int
    x: int
    y: int
    team: int
    rank: int
    fuel: int
    leaderboard_position: int


def make_self_state(
    tank_id: int,
    x: int,
    y: int,
    team: int,
    rank: int,
    fuel: int,
    leaderboard_position: int,
) -> SelfStateDict:
    """Create self state.

    Args:
        tank_id: Player's tank ID.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        team: Team ID (0-3).
        rank: Military rank (0-7).
        fuel: Current fuel amount.
        leaderboard_position: Leaderboard position.

    Returns:
        SelfStateDict with the provided values.
    """
    return SelfStateDict(
        tank_id=tank_id,
        x=x,
        y=y,
        team=team,
        rank=rank,
        fuel=fuel,
        leaderboard_position=leaderboard_position,
    )


def encode_self_state(state: SelfStateDict) -> JSONObject:
    """Encode SelfStateDict to JSON-serializable dict.

    Args:
        state: SelfStateDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "tank_id": state["tank_id"],
        "x": state["x"],
        "y": state["y"],
        "team": state["team"],
        "rank": state["rank"],
        "fuel": state["fuel"],
        "leaderboard_position": state["leaderboard_position"],
    }


def decode_self_state(data: JSONObject) -> SelfStateDict:
    """Decode SelfStateDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated SelfStateDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return SelfStateDict(
        tank_id=require_int(data, "tank_id"),
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        team=require_int(data, "team"),
        rank=require_int(data, "rank"),
        fuel=require_int(data, "fuel"),
        leaderboard_position=require_int(data, "leaderboard_position"),
    )


__all__ = [
    "SelfStateDict",
    "decode_self_state",
    "encode_self_state",
    "make_self_state",
]
