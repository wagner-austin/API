"""Command dispatch and equipment management for the tick loop.

Pure functions that translate a TickDecisionDict into bot actions:
equipment slot toggling and command sending.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import BotProtocol
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import BotCommand
from tankpit_bot.runtime_logging import emit_ai
from tankpit_bot.state import ContainerStateDict, TankStateDict, WorldStateDict, coord_key
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

# Combat equipment slots that get toggled based on behavior mode.
# Slot 5 (radar) is handled separately — always enabled when desired + stocked.
_COMBAT_SLOTS: list[int] = [1, 2, 4]
_EQUIPMENT_LABELS: dict[int, str] = {
    1: "armor",
    2: "dual",
    3: "missile",
    4: "homing",
    5: "radar",
}


def _format_desired_equipment(desired: list[int]) -> str:
    """Return a readable equipment summary for log output.

    Args:
        desired: Sorted list of desired equipment slots.

    Returns:
        Comma-separated equipment names, or ``none`` if empty.
    """
    if not desired:
        return "none"
    return ",".join(_EQUIPMENT_LABELS.get(slot, f"slot{slot}") for slot in desired)


def apply_equipment(bot: BotProtocol, desired: list[int]) -> None:
    """Enable desired equipment slots and disable undesired ones.

    Checks inventory stock before enabling — does not enable a slot
    if the bot has zero remaining of that item.  Slot 5 (extra radar)
    is enabled first if desired.  Combat slots (1, 2, 4) are toggled
    on or off.

    Args:
        bot: Bot instance for equipment commands.
        desired: Sorted list of equipment slot numbers (1-5) to enable.
    """
    # Extra radar: always enable if desired and stocked
    if 5 in desired and bot._has_equipment_stock(5):
        bot.enable_equipment(5)

    # Combat slots (1=armor, 2=dual, 4=homing): enable if desired + stocked, else disable
    for slot in _COMBAT_SLOTS:
        if slot in desired and bot._has_equipment_stock(slot):
            bot.enable_equipment(slot)
        else:
            bot.disable_equipment(slot)


def dispatch_command(bot: BotProtocol, command: BotCommand) -> bool:
    """Dispatch a BotCommand through the appropriate bot action method.

    Args:
        bot: Bot instance for sending commands.
        command: The bot command to execute.

    Returns:
        True if the command was sent successfully.
    """
    if command["cmd_type"] == "move":
        return bot.move_to(command["target_x"], command["target_y"])
    if command["cmd_type"] == "pickup_fuel":
        return bot.pickup_fuel_to(command["target_x"], command["target_y"])
    if command["cmd_type"] == "pickup_equipment":
        return bot.pickup_equipment_to(command["target_x"], command["target_y"])
    if command["cmd_type"] == "shoot":
        return bot.shoot_at(command["target_x"], command["target_y"], command["target_id"])
    if command["cmd_type"] == "radar":
        return bot.use_radar()
    if command["cmd_type"] == "map_open":
        return bot.open_map()
    # Teleport: open map first (required), then teleport
    bot.open_map()
    return bot.teleport_to(command["target_x"], command["target_y"])


def _tracked_tank(world: WorldStateDict, tank_id: int) -> TankStateDict | None:
    """Return the tracked tank for a target id when present.

    Args:
        world: Current world-state snapshot.
        tank_id: Tank identifier from a planner command or AI state.

    Returns:
        Matching TankStateDict, or None if not tracked.
    """
    if tank_id <= 0:
        return None
    return world["tanks"].get(str(tank_id))


def _tracked_container(
    world: WorldStateDict,
    x: int,
    y: int,
) -> ContainerStateDict | None:
    """Return the tracked container at a coordinate when present.

    Args:
        world: Current world-state snapshot.
        x: Container X coordinate.
        y: Container Y coordinate.

    Returns:
        Matching ContainerStateDict, or None if absent.
    """
    return world["containers"].get(coord_key(x, y))


def _is_valid_shoot(world: WorldStateDict, command: BotCommand) -> bool:
    """Return True when a shoot command targets a current viewport tank.

    Args:
        world: Current world-state snapshot.
        command: Command selected by the planner.

    Returns:
        True when the target tank still exists, still matches the coordinates,
        and is currently viewport-confirmed.
    """
    if command["cmd_type"] != "shoot":
        return True
    tank = _tracked_tank(world, command["target_id"])
    if tank is None:
        emit_ai(
            "rejecting shoot at (%d,%d): target id=%d not tracked",
            command["target_x"],
            command["target_y"],
            command["target_id"],
        )
        return False
    if tank["x"] != command["target_x"] or tank["y"] != command["target_y"]:
        emit_ai(
            "rejecting shoot at (%d,%d): target id=%d moved to (%d,%d)",
            command["target_x"],
            command["target_y"],
            command["target_id"],
            tank["x"],
            tank["y"],
        )
        return False
    if tank["source"] == "viewport":
        return True
    if tank["source"] == "world_state" and _is_within_visible_viewport(
        world,
        command["target_x"],
        command["target_y"],
    ):
        return True
    emit_ai(
        "rejecting shoot at (%d,%d): target id=%d source=%s is not viewport-fresh",
        command["target_x"],
        command["target_y"],
        command["target_id"],
        tank["source"],
    )
    return False


def _is_within_visible_viewport(world: WorldStateDict, x: int, y: int) -> bool:
    """Return True when a coordinate is inside the current visible viewport."""
    left, top, right, bottom = viewport_visible_bounds(world["viewport"])
    return left <= x <= right and top <= y <= bottom


def _is_valid_pickup(world: WorldStateDict, command: BotCommand) -> bool:
    """Return True when a pickup command still has a matching container.

    Args:
        world: Current world-state snapshot.
        command: Command selected by the planner.

    Returns:
        True when the target container exists and matches the pickup kind.
    """
    if command["cmd_type"] == "pickup_fuel":
        target_x = command["target_x"]
        target_y = command["target_y"]
        container = _tracked_container(world, target_x, target_y)
        if container is None:
            emit_ai(
                "rejecting pickup_fuel at (%d,%d): container no longer exists",
                target_x,
                target_y,
            )
            return False
        if not container["is_fuel"]:
            emit_ai(
                "rejecting pickup_fuel at (%d,%d): tracked container is equipment",
                target_x,
                target_y,
            )
            return False
        return True
    if command["cmd_type"] == "pickup_equipment":
        target_x = command["target_x"]
        target_y = command["target_y"]
        container = _tracked_container(world, target_x, target_y)
        if container is None:
            emit_ai(
                "rejecting pickup_equipment at (%d,%d): container no longer exists",
                target_x,
                target_y,
            )
            return False
        if container["is_fuel"]:
            emit_ai(
                "rejecting pickup_equipment at (%d,%d): tracked container is fuel",
                target_x,
                target_y,
            )
            return False
        return True
    return True


def _is_valid_move_destination(world: WorldStateDict, command: BotCommand) -> bool:
    """Return True when a move or teleport destination is not a known mine.

    Args:
        world: Current world-state snapshot.
        command: Command selected by the planner.

    Returns:
        True when the destination tile is not a known mine.
    """
    if command["cmd_type"] == "move" or command["cmd_type"] == "teleport":
        target_x = command["target_x"]
        target_y = command["target_y"]
    else:
        return True
    if coord_key(target_x, target_y) in world["mines"]:
        emit_ai(
            "rejecting %s to (%d,%d): destination is a known mine",
            command["cmd_type"],
            target_x,
            target_y,
        )
        return False
    return True


def _tracked_combat_target(
    world: WorldStateDict,
    decision: TickDecisionDict,
) -> TankStateDict | None:
    """Return the combat target currently locked in AI state.

    Args:
        world: Current world-state snapshot.
        decision: Planner decision under execution.

    Returns:
        Matching tracked tank, or None if the locked combat target is absent
        or no longer matches the AI state's coordinates.
    """
    ai_state = decision["updated_ai_state"]
    target_id = ai_state["combat_target_id"]
    if target_id == -1:
        return None
    tank = _tracked_tank(world, target_id)
    if tank is None:
        return None
    if tank["x"] != ai_state["combat_target_x"] or tank["y"] != ai_state["combat_target_y"]:
        return None
    return tank


def _tracked_resource_target(
    world: WorldStateDict,
    decision: TickDecisionDict,
) -> ContainerStateDict | None:
    """Return the resource target currently locked in AI state.

    Args:
        world: Current world-state snapshot.
        decision: Planner decision under execution.

    Returns:
        Matching tracked container, or None if the locked target is absent or
        does not match the locked resource kind.
    """
    ai_state = decision["updated_ai_state"]
    resource_kind = ai_state["resource_target_kind"]
    if resource_kind == "":
        return None
    container = _tracked_container(
        world,
        ai_state["resource_target_x"],
        ai_state["resource_target_y"],
    )
    if container is None:
        return None
    if resource_kind == "fuel" and not container["is_fuel"]:
        return None
    if resource_kind == "equipment" and container["is_fuel"]:
        return None
    return container


def _is_valid_teleport(world: WorldStateDict, decision: TickDecisionDict) -> bool:
    """Return True when a teleport still has a trustworthy target anchor.

    Combat teleports require a currently tracked combat target with a source
    that is valid for teleporting. Resource teleports require the locked
    resource target to still exist and still be locally trustworthy. Search
    hops without a locked target are allowed through unchanged.

    Args:
        world: Current world-state snapshot.
        decision: Planner decision under execution.

    Returns:
        True when the teleport remains valid against current world state.
    """
    command = decision["command"]
    if command["cmd_type"] != "teleport":
        return True
    behavior_mode = decision["behavior"]["mode"]
    if behavior_mode == "HUNT":
        combat_target = _tracked_combat_target(world, decision)
        if decision["updated_ai_state"]["combat_target_id"] == -1:
            return True
        if combat_target is None:
            emit_ai(
                "rejecting teleport to (%d,%d): combat target is stale",
                command["target_x"],
                command["target_y"],
            )
            return False
        if combat_target["source"] not in ("viewport", "world_state"):
            emit_ai(
                "rejecting teleport to (%d,%d): combat target source=%s is invalid",
                command["target_x"],
                command["target_y"],
                combat_target["source"],
            )
            return False
        return True
    resource_target = _tracked_resource_target(world, decision)
    if decision["updated_ai_state"]["resource_target_kind"] == "":
        return True
    if resource_target is None:
        emit_ai(
            "rejecting teleport to (%d,%d): resource target is stale",
            command["target_x"],
            command["target_y"],
        )
        return False
    if resource_target["source"] not in ("viewport", "radar"):
        emit_ai(
            "rejecting teleport to (%d,%d): resource target source=%s is invalid",
            command["target_x"],
            command["target_y"],
            resource_target["source"],
        )
        return False
    return True


def _is_dispatchable(bot: BotProtocol, decision: TickDecisionDict) -> bool:
    """Return True when a decision survives executor-side validation.

    Args:
        bot: Bot instance providing the current world-state snapshot.
        decision: Planner decision under execution.

    Returns:
        True when the command is still valid against current world state.
    """
    world = bot.get_world_state()
    command = decision["command"]
    return (
        _is_valid_move_destination(world, command)
        and _is_valid_pickup(world, command)
        and _is_valid_shoot(world, command)
        and _is_valid_teleport(world, decision)
    )


def execute(bot: BotProtocol, decision: TickDecisionDict) -> bool:
    """Execute a tick decision: apply equipment then dispatch command.

    Args:
        bot: Bot instance for sending commands.
        decision: The strategy's tick decision.

    Returns:
        True if the selected command was actually dispatched.
    """
    apply_equipment(bot, decision["desired_equipment"])

    behavior = decision["behavior"]
    command = decision["command"]
    emit_ai(
        "%s score=%d target=(%d,%d) cmd=%s equip=%s reason=%s",
        behavior["mode"],
        behavior["score"],
        behavior["target_x"],
        behavior["target_y"],
        command["cmd_type"],
        _format_desired_equipment(decision["desired_equipment"]),
        behavior["reason"],
    )

    if not _is_dispatchable(bot, decision):
        return False
    return dispatch_command(bot, command)


__all__ = [
    "apply_equipment",
    "dispatch_command",
    "execute",
]
