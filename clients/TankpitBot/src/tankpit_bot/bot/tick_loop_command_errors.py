"""Command-error handling for the tick loop.

The server rejects a dispatched command by returning an error the loop
must clear rather than stall on. This module owns that path end to end:
the orphan drain, the sync emit, the clear itself, and the rejection
outcome it writes to the ledger.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.states import ActionKind, InFlightActionDict, make_no_action
from tankpit_bot.browser.cdp_utils import get_current_time_ms
from tankpit_bot.ledger.outcome.collect import (
    emit_collect_clamped_transfer,
    emit_collect_command_rejected,
    emit_collect_inventory_full,
    emit_collect_pickup_empty,
)
from tankpit_bot.ledger.outcome.map_open import (
    emit_map_open_command_rejected,
)
from tankpit_bot.ledger.outcome.move import (
    emit_move_command_rejected,
)
from tankpit_bot.ledger.outcome.scan import (
    emit_scan_command_rejected,
)
from tankpit_bot.ledger.outcome.teleport import (
    emit_teleport_command_rejected,
)
from tankpit_bot.protocol.constants import (
    SUPERVISOR_ERROR_CANT_DO,
    SUPERVISOR_ERROR_CANT_GO,
    SUPERVISOR_ERROR_EMPTY_CONTAINER,
    SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
    SUPERVISOR_ERROR_INVENTORY_FULL,
    SUPERVISOR_ERROR_NAMES,
    SUPERVISOR_ERROR_TANK_FULL,
)
from tankpit_bot.runtime_logging import (
    emit_diagnostic,
    emit_sync,
)
from tankpit_bot.sniffer.world_state import (
    get_world_service,
    mark_container_desync,
    mark_move_target_failed,
    record_movement_rejection,
)
from tankpit_bot.sniffer.world_state_combat import check_and_clear_command_error
from tankpit_bot.sniffer.world_state_containers import (
    increment_container_failed_pickups,
    remove_container_at,
)
from tankpit_bot.sniffer.world_state_dispatch_containers import was_recent_pickup_at
from tankpit_bot.sniffer.world_state_inventory import update_inventory_from_full_signal

# The 0x52 codes themselves are the canonical ``SUPERVISOR_ERROR_*``
# constants in ``protocol/constants.py`` (tpclient.js ``Gb[]``); the
# bot reacts to the codes that prove the in-flight action will never
# resolve.
#
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
            SUPERVISOR_ERROR_CANT_DO,
            SUPERVISOR_ERROR_CANT_GO,
            SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
        }
    ),
    "teleport": frozenset(
        {
            SUPERVISOR_ERROR_CANT_DO,
            SUPERVISOR_ERROR_CANT_GO,
            SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
        }
    ),
    "collect": frozenset(
        {
            SUPERVISOR_ERROR_CANT_DO,
            SUPERVISOR_ERROR_CANT_GO,
            SUPERVISOR_ERROR_EMPTY_CONTAINER,
            SUPERVISOR_ERROR_TANK_FULL,
            SUPERVISOR_ERROR_INVENTORY_FULL,
            SUPERVISOR_ERROR_INSUFFICIENT_FUEL,
        }
    ),
    "scan": frozenset(),
    "map_open": frozenset(),
    "shoot": frozenset(),  # shot rejections use their own path in tick_loop
    "none": frozenset(),
}

log = get_logger(__name__)


def _mark_movement_failure(kind: ActionKind, error_code: int, tx: int, ty: int) -> None:
    """Record a rejected movement's failed-target mark, when deserved.

    Code 0 on a teleport is a PRECONDITION receipt, not a verdict on
    the tile (flag s10-1: the map had closed server-side while the
    client snapshot still read open, and a perfectly good larder
    landing got a failed-target mark). The replan defers for a fresh
    map open and the same tile succeeds. Every other rejection marks
    the tile so the re-derivation avoids it for the TTL.

    Args:
        kind: The rejected action's kind.
        error_code: The 0x52 code the server answered with.
        tx: Target X.
        ty: Target Y.
    """
    if kind == "teleport" and error_code == SUPERVISOR_ERROR_CANT_DO:
        emit_sync(
            "teleport to (%d,%d) refused code=0 (map closed server-side) "
            "- tile not marked, replanning with a fresh map open",
            tx,
            ty,
        )
        return
    mark_move_target_failed(tx, ty, get_current_time_ms())
    emit_sync("marked (%d,%d) as failed %s target", tx, ty, kind)


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


def _emit_command_error_sync(kind: ActionKind, tx: int, ty: int, error_code: int) -> None:
    """Log one applicable 0x52 with truthful receipt/rejection wording.

    A collect's 0x52 close is a RECEIPT, not a rejection: code 5 is
    the clamp SUCCESS close (the transfer landed in the same batch),
    code 4 the drained close, code 7 the inventory statement — the
    measured choreography ([[fuel-system]]). Logging them all as
    "rejected" hid the 2026-08-03 nope-fight autopsy's ground truth
    (32 "rejections" that were successful drinks) from three read
    passes. Genuine rejections keep the word.

    Args:
        kind: The in-flight action's kind.
        tx: The action's target X.
        ty: The action's target Y.
        error_code: The 0x52 code the server answered.
    """
    if kind == "collect" and error_code in (
        SUPERVISOR_ERROR_TANK_FULL,
        SUPERVISOR_ERROR_EMPTY_CONTAINER,
        SUPERVISOR_ERROR_INVENTORY_FULL,
    ):
        emit_sync(
            "%s to (%d,%d) closed by server receipt %s (code=%d)",
            kind,
            tx,
            ty,
            SUPERVISOR_ERROR_NAMES[error_code],
            error_code,
        )
        return
    emit_sync(
        "%s to (%d,%d) rejected by server %s (code=%d), replanning",
        kind,
        tx,
        ty,
        SUPERVISOR_ERROR_NAMES[error_code],
        error_code,
    )


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
    error_code = check_and_clear_command_error(bot.world)
    if error_code == -1:
        return False
    kind: ActionKind = action["kind"]
    if error_code not in _COMMAND_ERROR_APPLICABILITY[kind]:
        _emit_orphan_command_error(kind, error_code)
        return False
    tx, ty = action["target_x"], action["target_y"]
    started_ms = action["started_ms"]
    elapsed_ms = get_current_time_ms() - started_ms if started_ms > 0 else -1
    _emit_command_error_sync(kind, tx, ty, error_code)
    _emit_command_rejected_outcome(bot, kind, tx, ty, elapsed_ms, error_code)
    if kind == "collect":
        # Semantic split (Bug 0.3, 2026-07-06): "failed pickup" and
        # "blacklist this container forever" are not the same event.
        #
        # * ``code=5`` ("Tank full"): the server's CLAMP RECEIPT — it
        #   arrives ALONGSIDE the successful clamped transfer of the
        #   same single click (bot-20260726-101949: one pickup_fuel,
        #   then 0x44 391->1100, 0x44 +0, code 5 in one batch; the
        #   10-kill run's 15 code-5s matched its 15 clamped_transfer
        #   outcomes 1:1). Not a refusal and not a race — the pickup
        #   SUCCEEDED and the container keeps the remainder.
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
        if error_code == SUPERVISOR_ERROR_TANK_FULL:
            emit_sync(
                "container at (%d,%d) rejected code=5 (tank full) -- "
                "not blacklisting, container is not empty",
                tx,
                ty,
            )
        elif error_code == SUPERVISOR_ERROR_INVENTORY_FULL:
            update_inventory_from_full_signal(bot.world)
            emit_sync(
                "container at (%d,%d) rejected code=7 (inventory full) -- "
                "reconciled all slots to capacity, container kept",
                tx,
                ty,
            )
        elif error_code == SUPERVISOR_ERROR_EMPTY_CONTAINER:
            remove_container_at(bot.world, tx, ty)
            if was_recent_pickup_at(bot.world, tx, ty, get_current_time_ms()):
                # Drain receipt (flag s9-4): the code=4 rode our own
                # successful pickup -- the ContainerPickup broadcast
                # for this tile fired within the click. Nothing about
                # memory is wrong; the belief removal is the whole
                # correction.
                emit_sync(
                    "container at (%d,%d) rejected code=4 (empty) -- drain "
                    "receipt of own pickup, belief removed",
                    tx,
                    ty,
                )
            else:
                # A genuinely vanished container: no pickup broadcast
                # ever fired for the tile, so the belief the planner
                # acted on was stale (user ruling 2026-07-30: "if one
                # item is stale or out of sync then its worth a radar.
                # not, 3 items") -- the collect cascade answers the
                # latch, subject to the radar-spend economics.
                mark_container_desync(get_current_time_ms())
                emit_sync(
                    "container at (%d,%d) rejected code=4 (empty) -- belief "
                    "removed, container memory marked desynced",
                    tx,
                    ty,
                )
        else:
            increment_container_failed_pickups(bot.world, tx, ty)
            emit_sync("marked container at (%d,%d) as failed pickup", tx, ty)
    if kind in ("move", "teleport"):
        _mark_movement_failure(kind, error_code, tx, ty)
    if error_code == SUPERVISOR_ERROR_CANT_GO and kind in ("move", "collect", "teleport"):
        # The shared fact behind a cant_go on ANY movement-bearing
        # command is "the tank tried to move and the server said no"
        # -- a walk-pickup's leg is a move even though its kind is
        # collect. Run bot-20260730-110x ticks 95-107: twelve
        # consecutive rejected walk-pickups under fire, invisible to
        # the per-tile marks because collect rejections only fed
        # failed_pickups.
        record_movement_rejection(get_current_time_ms())
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
            bot.world.ledger,
            duration_ms=elapsed_ms,
            target_x=tx,
            target_y=ty,
            error_code=error_code,
        )
    elif kind == "collect":
        # Codes 4/5/7 are resolutions, not refusals (2026-07-19): the
        # container was empty, the transfer was clamped at the cap (a
        # success -- the 5-min soak gained +2472 fuel across four of
        # these), or the inventory is authoritatively full. Only the
        # genuine refusals (code 0 geometry, code 1 can't-go) stay
        # ``command_rejected``.
        if error_code == SUPERVISOR_ERROR_EMPTY_CONTAINER:
            emit_collect_pickup_empty(
                bot.world.ledger, duration_ms=elapsed_ms, target_x=tx, target_y=ty
            )
        elif error_code == SUPERVISOR_ERROR_TANK_FULL:
            emit_collect_clamped_transfer(
                bot.world.ledger, duration_ms=elapsed_ms, target_x=tx, target_y=ty
            )
        elif error_code == SUPERVISOR_ERROR_INVENTORY_FULL:
            emit_collect_inventory_full(
                bot.world.ledger, duration_ms=elapsed_ms, target_x=tx, target_y=ty
            )
        else:
            emit_collect_command_rejected(
                bot.world.ledger,
                duration_ms=elapsed_ms,
                target_x=tx,
                target_y=ty,
                error_code=error_code,
            )
    elif kind == "teleport":
        emit_teleport_command_rejected(
            bot.world.ledger,
            duration_ms=elapsed_ms,
            target_x=tx,
            target_y=ty,
            error_code=error_code,
            messages=bot._messages,
        )
    elif kind == "scan":
        emit_scan_command_rejected(
            bot.world.ledger,
            duration_ms=elapsed_ms,
            target_x=tx,
            target_y=ty,
            error_code=error_code,
        )
    elif kind == "map_open":
        emit_map_open_command_rejected(
            bot.world.ledger, duration_ms=elapsed_ms, error_code=error_code
        )


__all__ = [
    "log",
]
