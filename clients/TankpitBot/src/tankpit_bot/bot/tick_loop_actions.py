"""In-flight action resolution for the tick loop.

Each ``_wait_for_*`` function checks whether a previously dispatched
action is still resolving. ``_clear_*`` functions detect stalls,
rejections, and terrain blocks and clear the action so the next tick
can replan.

Split from ``tick_loop.py`` which keeps the orchestrator and decision
readiness checks.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.bot.ai.reachability import (
    is_collection_reachable_in_viewport,
    is_move_reachable_in_viewport,
)
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.states import InFlightActionDict, make_no_action, transition_to
from tankpit_bot.browser.cdp_utils import get_current_time_ms
from tankpit_bot.diagnostics.teleport_attempts import emit_teleport_attempt_outcome
from tankpit_bot.runtime_logging import emit_sync, emit_wire_complete
from tankpit_bot.sniffer.world_state import (
    get_terrain_map,
    get_world_service,
    is_move_target_failed,
    mark_move_target_failed,
    mark_scan_viewport_failed,
)
from tankpit_bot.sniffer.world_state_combat import check_and_clear_command_error
from tankpit_bot.sniffer.world_state_containers import increment_container_failed_pickups

# Supervisor (0x52) error codes from tpclient.js ``Gb[]``. The bot
# reacts to the codes that prove the in-flight action will never
# resolve. The remaining codes are server-side messages that don't
# correspond to an in-flight action the bot can clear.
_COMMAND_ERROR_CANT_DO_THIS = 0  # "You can't do this"
_COMMAND_ERROR_CANT_GO_THERE = 1  # "You can't go there!"
_COMMAND_ERROR_EMPTY_CONTAINER = 4  # "Empty container"
_COMMAND_ERROR_TANK_FULL = 5  # "Tank full"
_COMMAND_ERROR_INVENTORY_FULL = 7  # "Inventory full"
_COMMAND_ERROR_INSUFFICIENT_FUEL = 8  # "Insufficient fuel"
_ACTION_BLOCKING_COMMAND_ERRORS = frozenset(
    {
        _COMMAND_ERROR_CANT_DO_THIS,
        _COMMAND_ERROR_CANT_GO_THERE,
        _COMMAND_ERROR_EMPTY_CONTAINER,
        _COMMAND_ERROR_TANK_FULL,
        _COMMAND_ERROR_INVENTORY_FULL,
        _COMMAND_ERROR_INSUFFICIENT_FUEL,
    }
)

log = get_logger(__name__)


def has_in_flight_action(bot: Bot) -> bool:
    """Return True when a previously issued action is still resolving.

    Args:
        bot: Bot instance.

    Returns:
        True if the bot should wait for an in-flight action to resolve.
    """
    action = bot._state_data["in_flight_action"]
    if action["kind"] == "none" or action["outcome"] != "pending":
        return False

    kind = action["kind"]
    if kind in ("move", "collect", "teleport"):
        return _wait_for_movement_action(bot, action)

    if kind == "scan":
        return _wait_for_scan_action(bot, action)

    if kind == "map_open":
        return _wait_for_map_open_action(bot, action)

    return False


def _wait_for_movement_action(bot: Bot, action: InFlightActionDict) -> bool:
    """Return True while a move/collect/teleport action is still resolving."""
    kind = action["kind"]
    tx, ty = action["target_x"], action["target_y"]
    if _clear_command_error(bot, action):
        return False
    if _clear_rejected_movement(bot, action):
        return False
    if _clear_stalled_action(bot, action):
        return False
    if kind == "move" and _clear_blocked_walk(bot, action):
        return False
    if kind == "collect" and _clear_blocked_collection(bot, action):
        return False
    if kind == "move":
        emit_sync("waiting for movement to (%d,%d)", tx, ty)
    elif kind == "teleport":
        emit_sync("waiting for teleport to (%d,%d)", tx, ty)
    else:
        emit_sync("waiting for collection at (%d,%d)", tx, ty)
    return True


def _wait_for_scan_action(bot: Bot, action: InFlightActionDict) -> bool:
    """Return True while a radar scan is still pending."""
    if _clear_command_error(bot, action):
        return False
    if _clear_stalled_action(bot, action):
        return False
    emit_sync("waiting for radar results")
    return True


def _wait_for_map_open_action(bot: Bot, action: InFlightActionDict) -> bool:
    """Return True while a map-open action is waiting on fresh server sync."""
    if _clear_command_error(bot, action):
        return False
    if _clear_stalled_action(bot, action):
        return False
    if _clear_completed_map_open(bot, action):
        return False
    emit_sync("waiting for map open sync")
    return True


def _clear_command_error(bot: Bot, action: InFlightActionDict) -> bool:
    """Clear an in-flight action when the server emitted a 0x52 rejection.

    The Supervisor message carries an authoritative reject ("You can't
    do this", "You can't go there!", "Empty container", "Tank full",
    "Inventory full", "Insufficient fuel") whose presence means the
    in-flight action will never resolve. Without this hook the bot
    waits the full ``action_stall_timeout_ms`` (10 s) before
    replanning. Live run 20260620-184223 / pre-bug-bash: 4 of 7
    collects stalled the full 10 s on rejections the wire had already
    reported. Live capture 20260620-190728 / 20260620-190830 caught
    the same shape with two ``error_code=7`` ("Inventory full")
    rejects at full inventory; code 7 joined the blocking set on
    2026-06-21 (see [[bot-behavior-contract]] §3.4).

    Args:
        bot: Bot instance.
        action: The in-flight action record.

    Returns:
        True if a blocking command error was consumed and the action
        was cleared.
    """
    error_code = check_and_clear_command_error(get_world_service())
    if error_code not in _ACTION_BLOCKING_COMMAND_ERRORS:
        return False
    kind = action["kind"]
    tx, ty = action["target_x"], action["target_y"]
    started_ms = action["started_ms"]
    elapsed_ms = get_current_time_ms() - started_ms if started_ms > 0 else -1
    emit_sync(
        "%s to (%d,%d) rejected by server (error_code=%d), replanning",
        kind,
        tx,
        ty,
        error_code,
    )
    emit_wire_complete(
        action_kind=kind,
        duration_ms=elapsed_ms,
        signal="command_rejected",
        target_x=tx,
        target_y=ty,
        error_code=error_code,
    )
    if kind == "collect":
        increment_container_failed_pickups(get_world_service(), tx, ty)
        emit_sync("marked container at (%d,%d) as failed pickup", tx, ty)
    if kind in ("move", "teleport"):
        mark_move_target_failed(tx, ty, get_current_time_ms())
        emit_sync("marked (%d,%d) as failed %s target", tx, ty, kind)
    bot._transition("IDLE", in_flight_action=make_no_action())
    return True


def _clear_rejected_movement(
    bot: Bot,
    action: InFlightActionDict,
) -> bool:
    """Clear a move/collect whose target the server rejected.

    Args:
        bot: Bot instance.
        action: The in-flight action record to check.

    Returns:
        True if the rejected action was cleared.
    """
    kind = action["kind"]
    if kind not in ("move", "collect"):
        return False
    tx, ty = action["target_x"], action["target_y"]
    now = get_current_time_ms()
    if not is_move_target_failed(tx, ty, now):
        return False
    started_ms = action["started_ms"]
    elapsed_ms = now - started_ms if started_ms > 0 else -1
    emit_sync("%s to (%d,%d) rejected by server, replanning", kind, tx, ty)
    emit_wire_complete(
        action_kind=kind,
        duration_ms=elapsed_ms,
        signal="movement_rejected",
        target_x=tx,
        target_y=ty,
    )
    if kind == "collect":
        increment_container_failed_pickups(get_world_service(), tx, ty)
        emit_sync("marked container at (%d,%d) as failed pickup", tx, ty)
    bot._transition("IDLE", in_flight_action=make_no_action())
    return True


def _clear_stalled_action(
    bot: Bot,
    action: InFlightActionDict,
) -> bool:
    """Clear a stalled action using its authoritative timing.

    Args:
        bot: Bot instance.
        action: The in-flight action record to check.

    Returns:
        True if the stalled action was cleared.
    """
    started_ms = action["started_ms"]
    if started_ms <= 0:
        return False
    elapsed_ms = get_current_time_ms() - started_ms
    timeout_ms = bot._ai_state["config"]["action_stall_timeout_ms"]
    if elapsed_ms < timeout_ms:
        return False
    tx, ty = action["target_x"], action["target_y"]
    emit_sync(
        "%s to (%d,%d) stalled for %d ms, replanning",
        action["kind"],
        tx,
        ty,
        elapsed_ms,
    )
    emit_wire_complete(
        action_kind=action["kind"],
        duration_ms=elapsed_ms,
        signal="stall_timeout",
        target_x=tx,
        target_y=ty,
        timeout_ms=timeout_ms,
    )
    if action["kind"] == "collect":
        increment_container_failed_pickups(get_world_service(), tx, ty)
        emit_sync("marked container at (%d,%d) as failed pickup", tx, ty)
    if action["kind"] == "scan":
        _mark_current_viewport_scan_failed(bot, get_current_time_ms())
    if action["kind"] in ("move", "teleport"):
        now = get_current_time_ms()
        mark_move_target_failed(tx, ty, now)
        emit_sync("marked (%d,%d) as failed %s target", tx, ty, action["kind"])
    if action["kind"] == "teleport":
        emit_teleport_attempt_outcome(status="stall_timeout", messages=bot._messages)
    bot._transition("IDLE", in_flight_action=make_no_action())
    return True


def _mark_current_viewport_scan_failed(bot: Bot, timestamp_ms: int) -> None:
    """Record the current viewport as a failed radar target.

    Args:
        bot: Bot instance.
        timestamp_ms: Failure timestamp in milliseconds.
    """
    viewport = bot.get_world_state()["viewport"]
    mark_scan_viewport_failed(viewport["left"], viewport["top"], timestamp_ms)
    emit_sync(
        "marked viewport (%d,%d) as failed scan target",
        viewport["left"],
        viewport["top"],
    )


def _clear_completed_map_open(
    bot: Bot,
    action: InFlightActionDict,
) -> bool:
    """Clear a pending map_open once the authoritative MAP_DATA was processed.

    Args:
        bot: Bot instance.
        action: The pending map_open action record.

    Returns:
        True if MAP_DATA was processed and the action was cleared.
    """
    if not get_world_service().check_and_clear_map_data_processed():
        return False
    started_ms = action["started_ms"]
    duration_ms = get_current_time_ms() - started_ms if started_ms > 0 else -1
    emit_wire_complete(
        action_kind="map_open",
        duration_ms=duration_ms,
        signal="map_data_processed",
    )
    bot._state_data = transition_to(
        bot._state_data,
        bot.get_state(),
        in_flight_action=make_no_action(),
    )
    return True


def _clear_blocked_walk(
    bot: Bot,
    action: InFlightActionDict,
) -> bool:
    """Clear a walk when terrain shows the destination is unreachable.

    Args:
        bot: Bot instance.
        action: The in-flight move action record.

    Returns:
        True if the blocked walk was cleared.
    """
    world = bot.get_world_state()
    self_state = world["self_state"]
    terrain = get_terrain_map()
    if self_state is None or terrain is None:
        return False
    tx, ty = action["target_x"], action["target_y"]
    if is_move_reachable_in_viewport(
        world,
        terrain,
        self_state["x"],
        self_state["y"],
        tx,
        ty,
        world["mines"],
    ):
        return False
    emit_sync("movement to (%d,%d) is terrain-blocked, replanning", tx, ty)
    bot._transition("IDLE", in_flight_action=make_no_action())
    return True


def _clear_blocked_collection(
    bot: Bot,
    action: InFlightActionDict,
) -> bool:
    """Clear a collection when the target is terrain-blocked.

    Args:
        bot: Bot instance.
        action: The in-flight collect action record.

    Returns:
        True if the blocked collection was cleared.
    """
    world = bot.get_world_state()
    self_state = world["self_state"]
    terrain = get_terrain_map()
    if self_state is None or terrain is None:
        return False
    tx, ty = action["target_x"], action["target_y"]
    if abs(self_state["x"] - tx) <= 1 and abs(self_state["y"] - ty) <= 1:
        return False
    if is_collection_reachable_in_viewport(
        world,
        terrain,
        self_state["x"],
        self_state["y"],
        tx,
        ty,
        world["mines"],
    ):
        return False
    emit_sync("collection target (%d,%d) is terrain-blocked, replanning", tx, ty)
    bot._transition("IDLE", in_flight_action=make_no_action())
    return True


__all__ = [
    "has_in_flight_action",
]
