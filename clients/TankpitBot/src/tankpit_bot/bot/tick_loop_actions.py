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

from tankpit_bot.bot.ai.equipment import hostile_mines
from tankpit_bot.bot.ai.reachability import (
    is_collection_reachable_in_viewport,
    is_move_reachable_in_viewport,
)
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.states import ActionKind, InFlightActionDict, make_no_action, transition_to
from tankpit_bot.browser.cdp_utils import get_current_time_ms
from tankpit_bot.ledger.outcome.collect import (
    emit_collect_command_rejected,
    emit_collect_movement_rejected,
    emit_collect_stall_timeout,
)
from tankpit_bot.ledger.outcome.map_open import (
    emit_map_open_command_rejected,
    emit_map_open_data_processed,
    emit_map_open_stall_timeout,
)
from tankpit_bot.ledger.outcome.move import (
    emit_move_command_rejected,
    emit_move_movement_rejected,
    emit_move_stall_timeout,
)
from tankpit_bot.ledger.outcome.scan import (
    emit_scan_command_rejected,
    emit_scan_stall_timeout,
)
from tankpit_bot.ledger.outcome.teleport import (
    emit_teleport_command_rejected,
    emit_teleport_stall_timeout,
)
from tankpit_bot.runtime_logging import emit_diagnostic, emit_sync
from tankpit_bot.sniffer.world_state import (
    get_terrain_map,
    get_world_service,
    is_move_target_failed,
    mark_move_target_failed,
    mark_scan_viewport_failed,
)
from tankpit_bot.sniffer.world_state_combat import check_and_clear_command_error
from tankpit_bot.sniffer.world_state_containers import (
    increment_container_failed_pickups,
    remove_container_at,
)
from tankpit_bot.sniffer.world_state_inventory import update_inventory_from_full_signal

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


# Per-``ActionKind`` whitelist of the 0x52 codes that action can legitimately
# draw. A 0x52 whose code is NOT in the current action's whitelist belongs
# to a prior action (typically one that already completed via a different
# wire signal like ``container_consumed`` or ``teleport_landed``) and would
# poison the in-flight action if attributed to it.
#
# Live-run 2026-07-06 20:20:59 was the smoking gun: a ``collect fuel at
# (189,77)`` completed via ``container_consumed``, but the wire's
# late-arriving code=4 ("Empty container") landed while the next tick's
# ``map_open`` was in flight. Under the previous universal blocking set,
# ``map_open`` was declared rejected, HUNT could not acquire, and the
# session exited ``no_viable_targets`` at fuel 531 with a fully-stocked
# tank. Radar and map_open both dispatch commands the server never
# rejects, so their whitelist is empty and any code arriving during their
# wait is treated as an orphan.
#
# Sourced from ``tpclient.js`` server-side dispatch (see
# ``wiki/pages/client-constants.md``) and the empirical set the bot has
# observed across captures. Codes ``2/3/6/9/10`` are informational or
# universally non-blocking and appear in no whitelist.
_COMMAND_ERROR_APPLICABILITY: dict[ActionKind, frozenset[int]] = {
    "move": frozenset(
        {
            _COMMAND_ERROR_CANT_DO_THIS,
            _COMMAND_ERROR_CANT_GO_THERE,
            _COMMAND_ERROR_INSUFFICIENT_FUEL,
        }
    ),
    "teleport": frozenset(
        {
            _COMMAND_ERROR_CANT_DO_THIS,
            _COMMAND_ERROR_CANT_GO_THERE,
            _COMMAND_ERROR_INSUFFICIENT_FUEL,
        }
    ),
    "collect": frozenset(
        {
            _COMMAND_ERROR_CANT_DO_THIS,
            _COMMAND_ERROR_CANT_GO_THERE,
            _COMMAND_ERROR_EMPTY_CONTAINER,
            _COMMAND_ERROR_TANK_FULL,
            _COMMAND_ERROR_INVENTORY_FULL,
            _COMMAND_ERROR_INSUFFICIENT_FUEL,
        }
    ),
    "scan": frozenset(),
    "map_open": frozenset(),
    "shoot": frozenset(),  # shot rejections use their own path in tick_loop
    "none": frozenset(),
}

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
    """Return True while a radar scan is still pending.

    Radar dispatch (``CMD_RADAR`` 0x66, client ``Mb``) is not
    server-side rejectable -- the server accepts every scan and
    replies with a ``0x4F`` result. Any 0x52 landing here must
    belong to a prior action; the drain call consumes and diagnoses
    it so it can't poison the NEXT action, but never transitions
    the scan out of pending.
    """
    _drain_orphan_command_error(action)
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
    _drain_orphan_command_error(action)
    if _clear_stalled_action(bot, action):
        return False
    if _clear_completed_map_open(bot, action):
        return False
    emit_sync("waiting for map open sync")
    return True


def _emit_orphan_command_error(kind: ActionKind, error_code: int) -> None:
    """Emit the sync log line + diagnostic for a dropped orphan 0x52 code.

    Shared by :func:`_drain_orphan_command_error` (scan/map_open wait
    paths, whose whitelists are empty) and :func:`_clear_command_error`
    (movement paths whose whitelists may still miss the incoming code).
    """
    emit_sync(
        "%s wait discarded orphan error_code=%d (not applicable to this kind)",
        kind,
        error_code,
    )
    emit_diagnostic(
        diagnostic_kind="orphan_command_error",
        action_kind=kind,
        error_code=error_code,
    )


def _drain_orphan_command_error(action: InFlightActionDict) -> None:
    """Drain a pending 0x52 code during a scan or map_open wait.

    Radar and map_open dispatch are not server-side rejectable, so
    :data:`_COMMAND_ERROR_APPLICABILITY` lists an empty whitelist for
    both. Any 0x52 arriving here therefore belongs to a prior action
    that already completed via a different wire signal
    (``container_consumed``, ``teleport_landed``, ...) but whose reject
    landed a beat later. Consuming it here prevents the code surviving
    into the NEXT action's wait -- which is exactly how live run
    2026-07-06 20:20:59 misattributed a stale code=4 to a map_open,
    ended HUNT acquisition, and exited the session at fuel 531.

    Args:
        action: The in-flight scan or map_open action.
    """
    error_code = check_and_clear_command_error(get_world_service())
    if error_code == -1:
        return
    _emit_orphan_command_error(action["kind"], error_code)


def _clear_command_error(bot: Bot, action: InFlightActionDict) -> bool:
    """Clear an in-flight movement action when the server emitted a 0x52 rejection.

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

    Error codes are scoped to the ``ActionKind`` they can legitimately
    reject: see :data:`_COMMAND_ERROR_APPLICABILITY`. A code that is
    not in the current kind's whitelist belongs to a prior action and
    is dropped as an orphan (same diagnostic path as
    :func:`_drain_orphan_command_error`) rather than being spuriously
    attributed to the in-flight action.

    Args:
        bot: Bot instance.
        action: The in-flight movement action record.

    Returns:
        True if an applicable command error was consumed and the
        action was cleared. False when there was no error, or when an
        orphan code was consumed but the action stays pending.
    """
    error_code = check_and_clear_command_error(get_world_service())
    if error_code == -1:
        return False
    kind: ActionKind = action["kind"]
    if error_code not in _COMMAND_ERROR_APPLICABILITY[kind]:
        _emit_orphan_command_error(kind, error_code)
        return False
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
    _emit_command_rejected_outcome(bot, kind, tx, ty, elapsed_ms, error_code)
    if kind == "collect":
        # Semantic split (Bug 0.3, 2026-07-06): "failed pickup" and
        # "blacklist this container forever" are not the same event.
        #
        # * ``code=5`` ("Tank full"): the container was not empty, the
        #   server refused because our tank could not accept the
        #   transfer. Bug 0.2's ``_pickup_not_worth_walk`` pre-dispatch gate (né _would_overfill)
        #   prevents this in the normal flow, so a surviving code=5 is
        #   a race between planner-time and dispatch-time fuel state.
        #   Blacklisting a still-full container is wrong -- the next
        #   tick with headroom can consume it.
        # * ``code=4`` ("Empty container"): the container is drained.
        #   Delete the belief outright -- the volume the planner acted
        #   on is contradicted by the server. (Until 2026-07-19 this
        #   removal was done by the DOM game-log consumer one or two
        #   ticks later; the wire code is the same signal, earlier.)
        # * ``code=7`` ("Inventory full"): user mechanic 2026-07-18 --
        #   containers "fill whatever is empty. you will only get a
        #   full inventory message if all your items are full." The
        #   container is fine; the TANK is full. Reconcile every slot
        #   belief up to capacity (the rejection is an authoritative
        #   absolute inventory statement) and do NOT blacklist.
        #   Pre-fix (through 2026-07-18) this blacklisted a perfectly
        #   good container per rejection.
        # * ``code=0`` ("You can't do this"): illegal geometry.
        #   Blacklist per position.
        #
        # Pre-fix (through 2026-07-06): all four codes incremented
        # failed_pickups, so the 22:37 fuel-loop's four consecutive
        # partial-transfer + code=5 events blacklisted four still-full
        # fuel containers.
        if error_code == _COMMAND_ERROR_TANK_FULL:
            emit_sync(
                "container at (%d,%d) rejected code=5 (tank full) -- "
                "not blacklisting, container is not empty",
                tx,
                ty,
            )
        elif error_code == _COMMAND_ERROR_INVENTORY_FULL:
            update_inventory_from_full_signal(get_world_service())
            emit_sync(
                "container at (%d,%d) rejected code=7 (inventory full) -- "
                "reconciled all slots to capacity, container kept",
                tx,
                ty,
            )
        elif error_code == _COMMAND_ERROR_EMPTY_CONTAINER:
            remove_container_at(get_world_service(), tx, ty)
            emit_sync(
                "container at (%d,%d) rejected code=4 (empty) -- belief removed",
                tx,
                ty,
            )
        else:
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
    if kind == "move":
        emit_move_movement_rejected(duration_ms=elapsed_ms, target_x=tx, target_y=ty)
    else:
        emit_collect_movement_rejected(duration_ms=elapsed_ms, target_x=tx, target_y=ty)
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
    _emit_stall_outcome(bot, action["kind"], tx, ty, elapsed_ms, timeout_ms)
    if action["kind"] == "collect":
        increment_container_failed_pickups(get_world_service(), tx, ty)
        emit_sync("marked container at (%d,%d) as failed pickup", tx, ty)
    if action["kind"] == "scan":
        _mark_current_viewport_scan_failed(bot, get_current_time_ms())
    if action["kind"] in ("move", "teleport"):
        now = get_current_time_ms()
        mark_move_target_failed(tx, ty, now)
        emit_sync("marked (%d,%d) as failed %s target", tx, ty, action["kind"])
    bot._transition("IDLE", in_flight_action=make_no_action())
    return True


def _emit_command_rejected_outcome(
    bot: Bot,
    kind: ActionKind,
    tx: int,
    ty: int,
    elapsed_ms: int,
    error_code: int,
) -> None:
    """Route a 0x52 rejection to its kind's typed outcome emitter.

    Args:
        bot: Bot instance (for the teleport wire window).
        kind: In-flight action kind that drew the rejection.
        tx: Action target X.
        ty: Action target Y.
        elapsed_ms: Dispatch-to-rejection wall-clock ms.
        error_code: The 0x52 error code.
    """
    if kind == "move":
        emit_move_command_rejected(
            duration_ms=elapsed_ms, target_x=tx, target_y=ty, error_code=error_code
        )
    elif kind == "collect":
        emit_collect_command_rejected(
            duration_ms=elapsed_ms, target_x=tx, target_y=ty, error_code=error_code
        )
    elif kind == "teleport":
        emit_teleport_command_rejected(
            duration_ms=elapsed_ms,
            target_x=tx,
            target_y=ty,
            error_code=error_code,
            messages=bot._messages,
        )
    elif kind == "scan":
        emit_scan_command_rejected(
            duration_ms=elapsed_ms, target_x=tx, target_y=ty, error_code=error_code
        )
    elif kind == "map_open":
        emit_map_open_command_rejected(duration_ms=elapsed_ms, error_code=error_code)


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
            duration_ms=elapsed_ms, target_x=tx, target_y=ty, timeout_ms=timeout_ms
        )
    elif kind == "collect":
        emit_collect_stall_timeout(
            duration_ms=elapsed_ms, target_x=tx, target_y=ty, timeout_ms=timeout_ms
        )
    elif kind == "teleport":
        emit_teleport_stall_timeout(
            duration_ms=elapsed_ms,
            target_x=tx,
            target_y=ty,
            timeout_ms=timeout_ms,
            messages=bot._messages,
        )
    elif kind == "scan":
        emit_scan_stall_timeout(
            duration_ms=elapsed_ms, target_x=tx, target_y=ty, timeout_ms=timeout_ms
        )
    elif kind == "map_open":
        emit_map_open_stall_timeout(duration_ms=elapsed_ms, timeout_ms=timeout_ms)


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
    emit_map_open_data_processed(duration_ms=duration_ms)
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
        hostile_mines(world),
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
        hostile_mines(world),
    ):
        return False
    emit_sync("collection target (%d,%d) is terrain-blocked, replanning", tx, ty)
    bot._transition("IDLE", in_flight_action=make_no_action())
    return True


__all__ = [
    "has_in_flight_action",
]
