"""TypedDicts for the tick loop decision pipeline.

Provides TickDecisionDict — the output of a strategy decision that the
executor consumes to dispatch commands and manage equipment.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from tankpit_bot.bot.ai.types import (
    AIStateDict,
    BehaviorScoreDict,
    decode_ai_state,
    decode_behavior_score,
    encode_ai_state,
    encode_behavior_score,
)
from tankpit_bot.bot.types import BotCommand


def _require_int_list(data: JSONObject, key: str) -> list[int]:
    """Validate and extract a list of ints from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        Validated list of int values.

    Raises:
        ValueError: If value is not a list of ints.
    """
    raw = data.get(key)
    if not isinstance(raw, list):
        raise ValueError(f"{key} must be a list")
    result: list[int] = []
    for item in raw:
        if not isinstance(item, int):
            raise ValueError(f"{key} items must be int, got {type(item).__name__}")
        result.append(item)
    return result


def _decode_bot_command(data: JSONObject) -> BotCommand:
    """Decode a BotCommand from JSON with validation.

    Args:
        data: JSON object containing cmd_type and optional target_x, target_y.

    Returns:
        Validated BotCommand.

    Raises:
        ValueError: If cmd_type is not recognized.
    """
    cmd_type = require_str(data, "cmd_type")
    if cmd_type == "radar":
        from tankpit_bot.bot.types import RadarCommandDict

        return RadarCommandDict(cmd_type="radar")
    if cmd_type == "map_open":
        from tankpit_bot.bot.types import MapOpenCommandDict

        return MapOpenCommandDict(cmd_type="map_open")
    target_x = require_int(data, "target_x")
    target_y = require_int(data, "target_y")
    if cmd_type == "move":
        from tankpit_bot.bot.types import MoveCommandDict

        return MoveCommandDict(cmd_type="move", target_x=target_x, target_y=target_y)
    if cmd_type == "shoot":
        from tankpit_bot.bot.types import ShootCommandDict

        target_id = require_int(data, "target_id")
        return ShootCommandDict(
            cmd_type="shoot",
            target_x=target_x,
            target_y=target_y,
            target_id=target_id,
        )
    if cmd_type == "pickup_fuel":
        from tankpit_bot.bot.types import PickupFuelCommandDict

        return PickupFuelCommandDict(cmd_type="pickup_fuel", target_x=target_x, target_y=target_y)
    if cmd_type == "pickup_equipment":
        from tankpit_bot.bot.types import PickupEquipmentCommandDict

        return PickupEquipmentCommandDict(
            cmd_type="pickup_equipment",
            target_x=target_x,
            target_y=target_y,
        )
    if cmd_type == "teleport":
        from tankpit_bot.bot.types import TeleportCommandDict

        return TeleportCommandDict(cmd_type="teleport", target_x=target_x, target_y=target_y)
    raise ValueError(f"Unknown cmd_type: {cmd_type!r}")


def _encode_bot_command(command: BotCommand) -> JSONObject:
    """Encode a BotCommand to JSON-serializable dict.

    Args:
        command: BotCommand to encode.

    Returns:
        JSON-serializable dict representation.
    """
    if command["cmd_type"] == "radar":
        return {"cmd_type": "radar"}
    if command["cmd_type"] == "map_open":
        return {"cmd_type": "map_open"}
    result: JSONObject = {
        "cmd_type": command["cmd_type"],
        "target_x": command["target_x"],
        "target_y": command["target_y"],
    }
    if command["cmd_type"] == "shoot":
        result["target_id"] = command["target_id"]
    return result


# =============================================================================
# TickDecisionDict
# =============================================================================


class TickDecisionDict(TypedDict):
    """Output of a strategy decision consumed by the executor.

    Attributes:
        command: The primary bot command to send this tick.
        secondary_command: Optional secondary command dispatched after the
            primary succeeds. The server queues commands, so both arrive
            in the same tick window (~2040ms shot cooldown).
        behavior: The chosen behavior score (for logging/debugging).
        updated_ai_state: New AI state to persist after this tick.
        desired_equipment: Sorted list of equipment slot numbers (1-5) to enable.
    """

    command: BotCommand
    secondary_command: BotCommand | None
    behavior: BehaviorScoreDict
    updated_ai_state: AIStateDict
    desired_equipment: list[int]


def make_tick_decision(
    command: BotCommand,
    behavior: BehaviorScoreDict,
    updated_ai_state: AIStateDict,
    desired_equipment: list[int],
    *,
    secondary_command: BotCommand | None = None,
) -> TickDecisionDict:
    """Create a TickDecisionDict.

    Args:
        command: The primary bot command to send.
        behavior: The chosen behavior score.
        updated_ai_state: New AI state to persist.
        desired_equipment: Sorted list of equipment slot numbers (1-5) to enable.
        secondary_command: Optional secondary command for multi-command ticks.

    Returns:
        TickDecisionDict with the provided values.
    """
    return TickDecisionDict(
        command=command,
        secondary_command=secondary_command,
        behavior=behavior,
        updated_ai_state=updated_ai_state,
        desired_equipment=sorted(desired_equipment),
    )


def encode_tick_decision(decision: TickDecisionDict) -> JSONObject:
    """Encode TickDecisionDict to JSON-serializable dict.

    Args:
        decision: TickDecisionDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    equipment: list[JSONValue] = list(decision["desired_equipment"])
    secondary = decision["secondary_command"]
    result: JSONObject = {
        "command": _encode_bot_command(decision["command"]),
        "behavior": encode_behavior_score(decision["behavior"]),
        "updated_ai_state": encode_ai_state(decision["updated_ai_state"]),
        "desired_equipment": equipment,
    }
    if secondary is not None:
        result["secondary_command"] = _encode_bot_command(secondary)
    return result


def decode_tick_decision(data: JSONObject) -> TickDecisionDict:
    """Decode TickDecisionDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated TickDecisionDict.

    Raises:
        ValueError: If any field is invalid.
    """
    command_raw = data.get("command")
    if not isinstance(command_raw, dict):
        raise ValueError("command must be an object")

    behavior_raw = data.get("behavior")
    if not isinstance(behavior_raw, dict):
        raise ValueError("behavior must be an object")

    ai_state_raw = data.get("updated_ai_state")
    if not isinstance(ai_state_raw, dict):
        raise ValueError("updated_ai_state must be an object")

    secondary_raw = data.get("secondary_command")
    secondary: BotCommand | None = None
    if isinstance(secondary_raw, dict):
        secondary = _decode_bot_command(secondary_raw)

    return TickDecisionDict(
        command=_decode_bot_command(command_raw),
        secondary_command=secondary,
        behavior=decode_behavior_score(behavior_raw),
        updated_ai_state=decode_ai_state(ai_state_raw),
        desired_equipment=_require_int_list(data, "desired_equipment"),
    )


__all__ = [
    "TickDecisionDict",
    "decode_tick_decision",
    "encode_tick_decision",
    "make_tick_decision",
]
