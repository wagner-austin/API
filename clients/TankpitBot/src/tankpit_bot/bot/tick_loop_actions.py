"""In-flight action tracking for the tick loop.

Whether an action is still outstanding, the per-kind waits, and the
clears for rejected, stalled, and blocked actions. Command-error
handling is :mod:`tankpit_bot.bot.tick_loop_command_errors`.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.ferry import compose_decision_terrain
from tankpit_bot.bot.ai.reachability import (
    is_collection_reachable_in_viewport,
    is_move_reachable_in_viewport,
)
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.states import ActionKind, InFlightActionDict, make_no_action, transition_to
from tankpit_bot.bot.tick_loop_command_errors import (
    _clear_command_error,
    _drain_orphan_command_error,
)
from tankpit_bot.browser.cdp_utils import get_current_time_ms
from tankpit_bot.ledger.outcome.collect import (
    emit_collect_movement_rejected,
    emit_collect_stall_timeout,
)
from tankpit_bot.ledger.outcome.map_open import (
    emit_map_open_data_processed,
    emit_map_open_stall_timeout,
)
from tankpit_bot.ledger.outcome.move import (
    emit_move_movement_rejected,
    emit_move_stall_timeout,
)
from tankpit_bot.ledger.outcome.scan import (
    emit_scan_stall_timeout,
)
from tankpit_bot.ledger.outcome.scope import (
    emit_scope_confirmed,
    emit_scope_stall_timeout,
)
from tankpit_bot.ledger.outcome.teleport import (
    emit_teleport_stall_timeout,
)
from tankpit_bot.runtime_logging import emit_sync
from tankpit_bot.sniffer.world_state_containers import (
    increment_container_failed_pickups,
)


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

    if kind == "scope":
        return _wait_for_scope_action(bot, action)

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
    """Return True while a radar scan is still pending.

    Radar dispatch (``CMD_RADAR`` 0x66, client ``Mb``) is not
    server-side rejectable -- the server accepts every scan and
    replies with a ``0x4F`` result. Any 0x52 landing here must
    belong to a prior action; the drain call consumes and diagnoses
    it so it can't poison the NEXT action, but never transitions
    the scan out of pending.
    """
    _drain_orphan_command_error(bot.world, action)
    if _clear_stalled_action(bot, action):
        return False
    emit_sync("waiting for radar results")
    return True


def _wait_for_map_open_action(bot: Bot, action: InFlightActionDict) -> bool:
    """Return True while a map-open action is waiting on fresh server sync.

    Map-open dispatch (``CMD_MAP_OPEN`` 0x6C, client ``Nb``) is not
    server-side rejectable. The same drain rule as :func:`_wait_for_
    scan_action` applies. Live run 2026-07-06 20:20:59 was the
    smoking gun: a late-arriving ``code=4`` from a completed collect
    was misattributed as a map_open rejection under the old
    universal blocking set.
    """
    _drain_orphan_command_error(bot.world, action)
    if _clear_stalled_action(bot, action):
        return False
    if _clear_completed_map_open(bot, action):
        return False
    emit_sync("waiting for map open sync")
    return True


def _wait_for_scope_action(bot: Bot, action: InFlightActionDict) -> bool:
    """Return True while a viewport pan awaits its 0x5A confirmation.

    Scope dispatch (``Rb``) is not server-side rejectable — the server
    either answers with the shifted 0x5A (median one server tick, 759
    archived pans) or silently drops the pan; the stall timeout is the
    only exit for a drop. Holding here is what makes the scope-pending
    radar drop unrepresentable ([[viewport-shift-protocol]]): no radar
    or map_open can dispatch until the window has settled. The same
    orphan-0x52 drain rule as :func:`_wait_for_scan_action` applies.
    """
    _drain_orphan_command_error(bot.world, action)
    if _clear_stalled_action(bot, action):
        return False
    if _clear_confirmed_scope(bot, action):
        return False
    emit_sync("waiting for scope confirmation")
    return True


def _clear_confirmed_scope(
    bot: Bot,
    action: InFlightActionDict,
) -> bool:
    """Clear a pending pan once its answering 0x5A was ingested.

    Args:
        bot: Bot instance.
        action: The pending scope action record.

    Returns:
        True if the 0x5A arrived and the action was cleared.
    """
    if not bot.world.check_and_clear_viewport_update_processed():
        return False
    started_ms = action["started_ms"]
    duration_ms = get_current_time_ms() - started_ms if started_ms > 0 else -1
    emit_scope_confirmed(bot.world.ledger, duration_ms=duration_ms)
    bot._state_data = transition_to(
        bot._state_data,
        bot.get_state(),
        in_flight_action=make_no_action(),
    )
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
    if not bot.world.is_move_target_failed(tx, ty, now):
        return False
    started_ms = action["started_ms"]
    elapsed_ms = now - started_ms if started_ms > 0 else -1
    emit_sync("%s to (%d,%d) rejected by server, replanning", kind, tx, ty)
    if kind == "move":
        emit_move_movement_rejected(
            bot.world.ledger, duration_ms=elapsed_ms, target_x=tx, target_y=ty
        )
    else:
        emit_collect_movement_rejected(
            bot.world.ledger, duration_ms=elapsed_ms, target_x=tx, target_y=ty
        )
    if kind == "collect":
        increment_container_failed_pickups(bot.world, tx, ty)
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
    _emit_stall_outcome(bot, action["kind"], tx, ty, elapsed_ms, timeout_ms)
    if action["kind"] == "collect":
        increment_container_failed_pickups(bot.world, tx, ty)
        emit_sync("marked container at (%d,%d) as failed pickup", tx, ty)
    if action["kind"] == "scan":
        _mark_current_viewport_scan_failed(bot, get_current_time_ms())
    if action["kind"] in ("move", "teleport"):
        now = get_current_time_ms()
        bot.world.mark_move_target_failed(tx, ty, now)
        emit_sync("marked (%d,%d) as failed %s target", tx, ty, action["kind"])
    bot._transition("IDLE", in_flight_action=make_no_action())
    return True


def _emit_stall_outcome(
    bot: Bot,
    kind: ActionKind,
    tx: int,
    ty: int,
    elapsed_ms: int,
    timeout_ms: int,
) -> None:
    """Route a stall timeout to its kind's typed outcome emitter.

    Args:
        bot: Bot instance (for the teleport wire window).
        kind: In-flight action kind that stalled.
        tx: Action target X.
        ty: Action target Y.
        elapsed_ms: Dispatch-to-stall wall-clock ms.
        timeout_ms: The stall threshold that fired.
    """
    if kind == "move":
        emit_move_stall_timeout(
            bot.world.ledger,
            duration_ms=elapsed_ms,
            target_x=tx,
            target_y=ty,
            timeout_ms=timeout_ms,
        )
    elif kind == "collect":
        emit_collect_stall_timeout(
            bot.world.ledger,
            duration_ms=elapsed_ms,
            target_x=tx,
            target_y=ty,
            timeout_ms=timeout_ms,
        )
    elif kind == "teleport":
        emit_teleport_stall_timeout(
            bot.world.ledger,
            duration_ms=elapsed_ms,
            target_x=tx,
            target_y=ty,
            timeout_ms=timeout_ms,
            messages=bot._messages,
        )
    elif kind == "scan":
        emit_scan_stall_timeout(
            bot.world.ledger,
            duration_ms=elapsed_ms,
            target_x=tx,
            target_y=ty,
            timeout_ms=timeout_ms,
        )
    elif kind == "map_open":
        emit_map_open_stall_timeout(bot.world.ledger, duration_ms=elapsed_ms, timeout_ms=timeout_ms)
    elif kind == "scope":
        emit_scope_stall_timeout(bot.world.ledger, duration_ms=elapsed_ms, timeout_ms=timeout_ms)


def _mark_current_viewport_scan_failed(bot: Bot, timestamp_ms: int) -> None:
    """Record the current viewport as a failed radar target.

    Args:
        bot: Bot instance.
        timestamp_ms: Failure timestamp in milliseconds.
    """
    viewport = bot.get_world_state()["viewport"]
    bot.world.mark_scan_viewport_failed(viewport["left"], viewport["top"], timestamp_ms)
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
    if not bot.world.check_and_clear_map_data_processed():
        return False
    started_ms = action["started_ms"]
    duration_ms = get_current_time_ms() - started_ms if started_ms > 0 else -1
    emit_map_open_data_processed(bot.world.ledger, duration_ms=duration_ms)
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
    now_ms = get_current_time_ms()
    terrain = compose_decision_terrain(
        world,
        bot.world.get_terrain_map(),
        now_ms,
        bot.world.hostile_landing_keys(now_ms),
    )
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
    now_ms = get_current_time_ms()
    terrain = compose_decision_terrain(
        world,
        bot.world.get_terrain_map(),
        now_ms,
        bot.world.hostile_landing_keys(now_ms),
    )
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
    ):
        return False
    emit_sync("collection target (%d,%d) is terrain-blocked, replanning", tx, ty)
    bot._transition("IDLE", in_flight_action=make_no_action())
    return True


__all__ = [
    "has_in_flight_action",
]
