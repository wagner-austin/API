"""Command dispatch and equipment management for the tick loop.

Pure functions that translate a TickDecisionDict into bot actions:
equipment slot toggling and command sending.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import BotProtocol
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import BotCommand
from tankpit_bot.runtime_logging import emit_ai

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


def execute(bot: BotProtocol, decision: TickDecisionDict) -> None:
    """Execute a tick decision: apply equipment then dispatch command.

    Args:
        bot: Bot instance for sending commands.
        decision: The strategy's tick decision.
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

    dispatch_command(bot, command)


__all__ = [
    "apply_equipment",
    "dispatch_command",
    "execute",
]
