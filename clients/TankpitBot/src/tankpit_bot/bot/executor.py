"""Command dispatch and equipment management for the tick loop.

Pure functions that translate a TickDecisionDict into bot actions:
equipment slot toggling and command sending.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.scoring_types import render_reason
from tankpit_bot.bot.bot_protocol import BotProtocol
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import (
    BotCommand,
    PickupEquipmentCommandDict,
    PickupFuelCommandDict,
    TeleportCommandDict,
)
from tankpit_bot.browser.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.inventory import inventory_counts
from tankpit_bot.ledger.ammo_book import record_ammo_scan
from tankpit_bot.ledger.decision import record_decision
from tankpit_bot.ledger.events import ActionKind as LedgerActionKind
from tankpit_bot.ledger.fuel_book import record_fuel_entry
from tankpit_bot.ledger.outcome._emit import transfer_pending_decision
from tankpit_bot.ledger.outcome.teleport import (
    record_teleport_dispatch,
)
from tankpit_bot.physics.costs import RADAR_COST, teleport_cost
from tankpit_bot.physics.supervisor import (
    TELEPORT_RING1_COST_SLACK,
    equipment_pickup_refusal,
    fuel_pickup_refusal,
    teleport_refusal,
)
from tankpit_bot.runtime_logging import (
    emit_ai,
    emit_diagnostic,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types.coord import coord_key

# Combat equipment slots that get toggled based on behavior mode.
# Slot 5 (radar) is handled separately — always enabled when desired + stocked.
_COMBAT_SLOTS: list[int] = [1, 2, 4]

# Wire command type -> ledger action kind. ``hold`` dispatches nothing
# and is deliberately absent -- it produces no attempt to correlate.
_LEDGER_KIND_BY_CMD_TYPE: dict[str, LedgerActionKind] = {
    "move": "move",
    "teleport": "teleport",
    "pickup_fuel": "collect",
    "pickup_equipment": "collect",
    "radar": "scan",
    "map_open": "map_open",
    "shoot": "shoot",
}
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


def _predicted_pickup_refusal(
    ws: WorldService,
    command: PickupEquipmentCommandDict | PickupFuelCommandDict,
) -> int | None:
    """Predict the server's 0x52 answer to a pickup, when belief proves it.

    The ``physics/supervisor.py`` refusal laws applied to the live
    belief: a fuel pickup at rank fuel capacity, or an equipment
    pickup with every slot at the rank inventory cap, transfers
    nothing — the 20-kill soak bot-20260802-205105 sent 48 such
    pickups and the server refused every one. A target tile without a
    believed container record proves nothing (drained containers are
    REMOVED from belief, and drained-by-another-tank races are
    invisible) — those dispatch optimistically, which is correct.

    Args:
        ws: The session's world service, holding the live belief.
        command: A ``pickup_fuel`` or ``pickup_equipment`` command.

    Returns:
        The predicted 0x52 error code, or None to dispatch.
    """
    self_state = ws.world_state["self_state"]
    if self_state is None:
        return None
    if command["cmd_type"] == "pickup_fuel":
        container = ws.world_state["containers"].get(
            coord_key(command["target_x"], command["target_y"])
        )
        if container is None or not container["is_fuel"]:
            return None
        return fuel_pickup_refusal(self_state["fuel"], self_state["rank"], container["volume"])
    return equipment_pickup_refusal(inventory_counts(ws.inventory_state), self_state["rank"])


def _suppress_dispatch(
    command: PickupEquipmentCommandDict | PickupFuelCommandDict | TeleportCommandDict,
    predicted_code: int,
) -> bool:
    """Log a belief-refuted command instead of sending it.

    Args:
        command: The refuted command.
        predicted_code: The 0x52 code the server would answer.

    Returns:
        False — nothing was dispatched; the next tick replans.
    """
    emit_ai(
        "suppressed %s to (%d,%d): belief predicts 0x52 code %d",
        command["cmd_type"],
        command["target_x"],
        command["target_y"],
        predicted_code,
    )
    emit_diagnostic(
        diagnostic_kind="dispatch_suppressed",
        origin="executor.dispatch_command.refusal_prediction",
        command_name=command["cmd_type"],
        target_x=command["target_x"],
        target_y=command["target_y"],
        predicted_error_code=predicted_code,
    )
    return False


def _dispatch_tracked_target_command(bot: BotProtocol, command: BotCommand) -> bool:
    """Send a move or pickup command to its target tile.

    Server rejections (0x52 error codes) resolve against the in-flight
    action's recorded target, so no side-channel dispatch bookkeeping
    is needed here.

    Args:
        bot: Bot instance for sending commands.
        command: A ``move``, ``pickup_fuel``, or ``pickup_equipment``
            command.

    Returns:
        True if the command was dispatched.

    Raises:
        ValueError: If the command is not a tracked-target kind.
    """
    if command["cmd_type"] == "move":
        return bot.move_to(command["target_x"], command["target_y"])
    if command["cmd_type"] == "pickup_fuel":
        predicted = _predicted_pickup_refusal(bot.world, command)
        if predicted is not None:
            return _suppress_dispatch(command, predicted)
        return bot.pickup_fuel_to(command["target_x"], command["target_y"])
    if command["cmd_type"] == "pickup_equipment":
        predicted = _predicted_pickup_refusal(bot.world, command)
        if predicted is not None:
            return _suppress_dispatch(command, predicted)
        return bot.pickup_equipment_to(command["target_x"], command["target_y"])
    raise ValueError(f"Not a tracked-target command: {command['cmd_type']}")


def dispatch_command(
    bot: BotProtocol,
    command: BotCommand,
    snapshot: PageClientSnapshotDict,
) -> bool:
    """Dispatch a BotCommand through the appropriate bot action method.

    Args:
        bot: Bot instance for sending commands.
        command: The bot command to execute.
        snapshot: Live page-client state captured at the start of this tick.
            ``snapshot["map_visible"]`` short-circuits the teleport
            precondition: an already-open map lets the teleport dispatch
            directly instead of consuming a tick for CMD_MAP_OPEN.

    Returns:
        True if the desired effect was achieved -- either because a
        command was dispatched, or because the desired client state was
        already in place.
    """
    if command["cmd_type"] == "hold":
        # SPA-pinned idle tick — no wire traffic. Returning True keeps
        # the tick loop's success accounting honest (the desired
        # effect, "do nothing", was achieved).
        return True
    if command["cmd_type"] == "chat":
        # Fire-and-forget social send (the HELLO greeting). No ledger
        # entry: chat costs nothing, expects no world-state outcome,
        # and the flood-mute contract forbids retry-on-silence
        # ([[chat-messages]]) — so nothing downstream should wait on it.
        return bot.send_chat(command["message_id"], command["target_x"], command["target_y"])
    if command["cmd_type"] == "scope_shift":
        # Free viewport pan (no fuel, no queue slot). Like chat there
        # is no ledger entry: the outcome is the 0x5A confirmation the
        # sniffer ingests as the new viewport origin, and nothing
        # downstream correlates an attempt against it.
        return bot.scope_shift(command["direction"])
    if (
        command["cmd_type"] == "move"
        or command["cmd_type"] == "pickup_fuel"
        or command["cmd_type"] == "pickup_equipment"
    ):
        return _dispatch_tracked_target_command(bot, command)
    if command["cmd_type"] == "shoot":
        # Record the combat target so the 0x53 dispatcher can attribute
        # the seeker's resolved tile to the right tank when refreshing
        # off-viewport positions from homing/missile tracking. Snapshot
        # the inventory so combat_feedback can confirm hits via ammo
        # delta when the wire's victim_id lookup misses (off-viewport
        # target).
        ws = bot.world
        ws.last_shot_combat_target_id = command["target_id"]
        ws.pending_shot_inventory_snapshot = ws.inventory_state
        return bot.shoot_at(command["target_x"], command["target_y"], command["target_id"])
    if command["cmd_type"] == "radar":
        dispatched_radar = bot.use_radar()
        if dispatched_radar:
            record_fuel_entry(
                book=bot.world.fuel_book,
                kind="radar",
                lo=-RADAR_COST,
                hi=-RADAR_COST,
            )
            record_ammo_scan(book=bot.world.ammo_book)
        return dispatched_radar
    if command["cmd_type"] == "map_open":
        # CMD_MAP_OPEN is idempotent on the server: every dispatch
        # produces a fresh MAP_DATA payload regardless of whether the
        # client believes the overlay is open. Empirically every wire
        # CMD_MAP_OPEN in capture 20260620-183916 completed via
        # ``map_data_processed`` in ~2 s including those issued while
        # ``map_visible`` was ``True``. There is nothing to guard.
        return bot.open_map()
    return _dispatch_teleport_command(bot, command, snapshot)


def _dispatch_teleport_command(
    bot: BotProtocol,
    command: TeleportCommandDict,
    snapshot: PageClientSnapshotDict,
) -> bool:
    """Dispatch a teleport against a confirmed-open map, else open it.

    An OPEN map is a server-side precondition. A teleport dispatched
    in the same tick as the wire map_open races the server's open
    processing and is silently dropped: run 20260610-024x lost 4 of
    15 same-tick attempts to 10s stall timeouts while all 21 attempts
    against an already-open map landed. So the open and the teleport
    never share a tick: this tick opens the map, and the next tick's
    decision re-dispatches the teleport against a confirmed-open map.

    Args:
        bot: Bot instance for sending commands.
        command: The teleport command.
        snapshot: Live page-client state captured at tick start.

    Returns:
        True if the map open or the teleport was dispatched.
    """
    if snapshot["map_visible"] is not True:
        emit_ai(
            "deferring teleport to (%d,%d): opening map first",
            command["target_x"],
            command["target_y"],
        )
        emit_diagnostic(
            diagnostic_kind="teleport_deferred_for_map_open",
            origin="executor.dispatch_command.teleport_precondition",
            command_name="map_open",
            teleport_target_x=command["target_x"],
            teleport_target_y=command["target_y"],
        )
        # This tick's product is the map open, not the teleport: the
        # recorded teleport decision resolves via the map_open outcome.
        transfer_pending_decision(bot.world.ledger, "teleport", "map_open")
        return bot.open_map()
    emit_diagnostic(
        diagnostic_kind="map_open_skipped_already_open",
        origin="executor.dispatch_command.teleport_precondition",
        command_name="map_open",
        teleport_target_x=command["target_x"],
        teleport_target_y=command["target_y"],
    )
    self_state = bot.world.world_state["self_state"]
    if self_state is not None:
        floor_cost = (
            teleport_cost(
                self_state["x"], self_state["y"], command["target_x"], command["target_y"]
            )
            - TELEPORT_RING1_COST_SLACK
        )
        predicted = teleport_refusal(self_state["fuel"], floor_cost)
        if predicted is not None:
            return _suppress_dispatch(command, predicted)
    message_index = bot.captured_message_count()
    dispatched = bot.teleport_to(command["target_x"], command["target_y"])
    if dispatched:
        _record_teleport_fuel_entry(bot.world, command["target_x"], command["target_y"])
        record_teleport_dispatch(
            bot.world.ledger,
            target_x=command["target_x"],
            target_y=command["target_y"],
            message_index=message_index,
            sent_window=(
                f"map_visible={snapshot['map_visible']} "
                f"pending_actions={snapshot['pending_actions']} "
                f"ws_ready_state={snapshot['ws_ready_state']} "
                f"heartbeat_age_ms={snapshot['heartbeat_age_ms']} "
                f"page_send_age_ms={snapshot['last_page_client_send_age_ms']} "
                f"bot_send_age_ms={snapshot['last_bot_send_age_ms']}"
            ),
        )
    return dispatched


_TELEPORT_DRIFT_FUEL = 36
"""Displacement drift bound for the live fuel book: the server may
displace a landing several tiles off the requested target (mines,
terrain), changing the charge accordingly. Soak 3 (2026-07-21)
measured drift-priced misses up to ~16 fuel beyond the old 3-tile
bound, so the book prices a dispatched teleport as target cost +/-
6 tiles * 6 fuel."""


def _record_teleport_fuel_entry(ws: WorldService, target_x: int, target_y: int) -> None:
    """Record a dispatched teleport's fuel effect into the live book.

    Args:
        ws: The session's world service, holding the fuel book.
        target_x: Requested landing X.
        target_y: Requested landing Y.
    """
    self_state = ws.world_state["self_state"]
    if self_state is None:
        return
    cost = teleport_cost(self_state["x"], self_state["y"], target_x, target_y)
    record_fuel_entry(
        book=ws.fuel_book,
        kind="teleport",
        lo=-(cost + _TELEPORT_DRIFT_FUEL),
        hi=-max(cost - _TELEPORT_DRIFT_FUEL, 0),
    )


def execute(
    bot: BotProtocol,
    decision: TickDecisionDict,
    snapshot: PageClientSnapshotDict,
) -> bool:
    """Execute a tick decision: apply equipment then dispatch command.

    Args:
        bot: Bot instance for sending commands.
        decision: The strategy's tick decision.
        snapshot: Live page-client state captured at the start of this tick;
            forwarded to :func:`dispatch_command` so dispatch decisions read
            from authoritative live state rather than guessing.

    Returns:
        True if the selected command achieved its desired effect.
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
        render_reason(behavior),
        behavior_mode=behavior["mode"],
        behavior_score=behavior["score"],
        combat_target_x=behavior["target_x"],
        combat_target_y=behavior["target_y"],
        combat_target_id=behavior["target_id"],
        command_type=command["cmd_type"],
        behavior_reason=render_reason(behavior),
    )

    ledger_kind = _LEDGER_KIND_BY_CMD_TYPE.get(command["cmd_type"])
    if ledger_kind is not None:
        record_decision(
            bot.world.ledger,
            action_kind=ledger_kind,
            cmd_type=command["cmd_type"],
            mode=behavior["mode"],
            score=behavior["score"],
            reason_kind=behavior["reason_kind"],
            reason_context=behavior["reason_context"],
            target_x=behavior["target_x"],
            target_y=behavior["target_y"],
            target_id=behavior["target_id"],
        )
    primary_sent = dispatch_command(bot, command, snapshot)
    if primary_sent and decision["secondary_command"] is not None:
        secondary = decision["secondary_command"]
        emit_ai("secondary cmd=%s", secondary["cmd_type"])
        dispatch_command(bot, secondary, snapshot)
    return primary_sent


__all__ = [
    "apply_equipment",
    "dispatch_command",
    "execute",
]
