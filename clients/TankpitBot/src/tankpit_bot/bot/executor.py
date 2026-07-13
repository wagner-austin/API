"""Command dispatch and equipment management for the tick loop.

Pure functions that translate a TickDecisionDict into bot actions:
equipment slot toggling and command sending.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import BotProtocol
from tankpit_bot.action_lab.page_client_snapshot import PageClientSnapshotDict
from tankpit_bot.bot.ai.equipment import hostile_mines
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import BotCommand
from tankpit_bot.diagnostics.game_log_feedback import (
    record_move_dispatch,
    record_pickup_dispatch,
)
from tankpit_bot.diagnostics.teleport_attempts import record_teleport_dispatch
from tankpit_bot.runtime_logging import emit_ai, emit_diagnostic
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.state import ContainerStateDict, TankStateDict, WorldStateDict, coord_key

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


def _dispatch_tracked_target_command(bot: BotProtocol, command: BotCommand) -> bool:
    """Send a move/pickup command and record its target for log feedback.

    The game log reports failed pickups ("Empty container", "Tank
    full") and rejected moves ("You can't go there!") without naming a
    tile, so the dispatch target is recorded here for the feedback
    consumer to resolve against.

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
        dispatched = bot.move_to(command["target_x"], command["target_y"])
        if dispatched:
            record_move_dispatch(command["target_x"], command["target_y"])
        return dispatched
    if command["cmd_type"] == "pickup_fuel":
        dispatched = bot.pickup_fuel_to(command["target_x"], command["target_y"])
        if dispatched:
            # A pickup IS a move plus a grab: the tank drives to the
            # container, so a "You can't go there!" rejection belongs to
            # THIS target. Without the move record the rejection was
            # attributed to a stale, long-finished move dispatch (live
            # run 20260611-000x marked (105,154) for a pickup at
            # (129,152)).
            record_pickup_dispatch(command["target_x"], command["target_y"])
            record_move_dispatch(command["target_x"], command["target_y"])
        return dispatched
    if command["cmd_type"] == "pickup_equipment":
        dispatched = bot.pickup_equipment_to(command["target_x"], command["target_y"])
        if dispatched:
            record_pickup_dispatch(command["target_x"], command["target_y"])
            record_move_dispatch(command["target_x"], command["target_y"])
        return dispatched
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
    if command["cmd_type"] in ("move", "pickup_fuel", "pickup_equipment"):
        return _dispatch_tracked_target_command(bot, command)
    if command["cmd_type"] == "shoot":
        # Record the combat target so the 0x53 dispatcher can attribute
        # the seeker's resolved tile to the right tank when refreshing
        # off-viewport positions from homing/missile tracking. Snapshot
        # the inventory so combat_feedback can confirm hits via ammo
        # delta when the wire's victim_id lookup misses (off-viewport
        # target).
        ws = get_world_service()
        ws.last_shot_combat_target_id = command["target_id"]
        ws.pending_shot_inventory_snapshot = ws.inventory_state
        return bot.shoot_at(command["target_x"], command["target_y"], command["target_id"])
    if command["cmd_type"] == "radar":
        return bot.use_radar()
    if command["cmd_type"] == "map_open":
        # CMD_MAP_OPEN is idempotent on the server: every dispatch
        # produces a fresh MAP_DATA payload regardless of whether the
        # client believes the overlay is open. Empirically every wire
        # CMD_MAP_OPEN in capture 20260620-183916 completed via
        # ``map_data_processed`` in ~2 s including those issued while
        # ``map_visible`` was ``True``. There is nothing to guard.
        return bot.open_map()
    # Teleport: an OPEN map is a server-side precondition. A teleport
    # dispatched in the same tick as the wire map_open races the
    # server's open processing and is silently dropped: run
    # 20260610-024x lost 4 of 15 same-tick attempts to 10s stall
    # timeouts while all 21 attempts against an already-open map
    # landed. So the open and the teleport never share a tick: this
    # tick opens the map, and the next tick's decision re-dispatches
    # the teleport against a confirmed-open map.
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
        return bot.open_map()
    emit_diagnostic(
        diagnostic_kind="map_open_skipped_already_open",
        origin="executor.dispatch_command.teleport_precondition",
        command_name="map_open",
        teleport_target_x=command["target_x"],
        teleport_target_y=command["target_y"],
    )
    message_index = bot.captured_message_count()
    dispatched = bot.teleport_to(command["target_x"], command["target_y"])
    if dispatched:
        record_teleport_dispatch(
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
    """Return True when a shoot command still has a tracked target tank.

    Combat presence -- whether the target is a live tank rather than a
    map-only afterimage -- is decided once, upstream, by the HUNT owner's
    viewport-presence acquisition gate in :func:`analyze_threats`. The
    executor's remaining job is a race guard against the tank vanishing
    from the registry between planner-decide and dispatch: without a
    tracked tank there is no ``target_id`` for the server to route to
    and the shot would crash the wire.

    ``target_id`` is the truth channel. The server picks homing from the
    id and the seeker tracks the true target wherever it is.
    ``target_x``/``target_y`` are a viewport-legal aim hint used by the
    server to route to homing; under
    :func:`~tankpit_bot.bot.ai.combat_strategy._clamp_aim_into_viewport`
    the aim tile is deliberately clamped inside the viewport and drift
    between the aim and the tank's current position is intentional. The
    executor does not reject on that drift -- rejecting it silently
    blocked every clamped homing shot in the 2026-07-06 20:47:31
    live-run deadlock, where 26 s of client-side self-rejections
    accumulated before the loop broke.

    Args:
        world: Current world-state snapshot.
        command: Command selected by the planner.

    Returns:
        True when the target tank still exists in the tank registry.
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
    return True


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
    """Return True when a move or teleport destination is not a hostile mine.

    Friendly (same-team) mines are passable per tankpit's damage rules;
    only hostile mines block movement.

    Args:
        world: Current world-state snapshot.
        command: Command selected by the planner.

    Returns:
        True when the destination tile is not a hostile mine.
    """
    if command["cmd_type"] == "move" or command["cmd_type"] == "teleport":
        target_x = command["target_x"]
        target_y = command["target_y"]
    else:
        return True
    if coord_key(target_x, target_y) in hostile_mines(world):
        emit_ai(
            "rejecting %s to (%d,%d): destination is a hostile mine",
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
        behavior["reason"],
        behavior_mode=behavior["mode"],
        behavior_score=behavior["score"],
        combat_target_x=behavior["target_x"],
        combat_target_y=behavior["target_y"],
        combat_target_id=behavior["target_id"],
        command_type=command["cmd_type"],
        behavior_reason=behavior["reason"],
    )

    if not _is_dispatchable(bot, decision):
        return False
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
