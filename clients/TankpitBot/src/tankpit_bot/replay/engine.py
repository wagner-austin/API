"""Replay engine for offline bot decision analysis.

Feeds a captured session through the decode/world-state pipeline and
runs the planner tick-by-tick, recording a structured decision trace
for each tick where the planner has enough state to make a decision.

The engine does not simulate the execution state machine (in-flight
actions, command completion). It runs the planner on every tick where
self_state is available, showing the planner's full intent without
execution masking. This is intentional — it reveals what the planner
*would* decide given each world-state snapshot, which is more useful
for debugging bad decisions than replaying the exact execution sequence.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.bot import ai_strategy
from tankpit_bot.bot.ai.threats import analyze_threats
from tankpit_bot.bot.ai.types import AIStateDict, make_initial_ai_state, render_reason
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import (
    BotCommand,
    ChatCommandDict,
    MoveCommandDict,
    PickupEquipmentCommandDict,
    PickupFuelCommandDict,
    ShootCommandDict,
    TeleportCommandDict,
)
from tankpit_bot.capture.xor import build_session_xor_table
from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.replay.types import ReplaySessionResultDict, ReplayTickTraceDict
from tankpit_bot.sniffer.viewport import reset_viewport_tracking
from tankpit_bot.sniffer.world_state import (
    get_terrain_map,
    get_world_service,
    get_world_state,
    reset_world_state,
)
from tankpit_bot.sniffer.world_state_combat import drain_killed_tank_ids
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state
from tankpit_bot.state.types import SelfStateDict, WorldStateDict
from tankpit_bot.types import CaptureSession

log = get_logger(__name__)


def _sort_by_timestamp(pair: tuple[int, str]) -> int:
    """Return the timestamp from a (timestamp_ms, payload) pair.

    Args:
        pair: Tuple of (timestamp_ms, payload).

    Returns:
        The timestamp_ms value for sorting.
    """
    return pair[0]


def replay_session(session: CaptureSession) -> ReplaySessionResultDict:
    """Replay a captured session and return per-tick decision traces.

    Resets the global world/viewport state, builds this session's XOR
    table as a LOCAL, then processes received messages in tick-sized
    batches. After each batch the planner runs and its decision is
    recorded.

    Args:
        session: Loaded and validated capture session.

    Returns:
        ReplaySessionResultDict with per-tick decision traces.

    Raises:
        ValueError: If session has no magic key (cannot XOR-decode).
        XorStaticKeyUnavailableError: If the static key cannot be read.
    """
    magic = session["magic"]
    if magic is None:
        raise ValueError("Cannot replay session without magic key")

    # Reset the remaining global state for a clean replay
    reset_world_state()
    reset_viewport_tracking()
    xor_table = build_session_xor_table(magic)

    # Filter to received messages only, sorted by timestamp
    received_payloads: list[tuple[int, str]] = [
        (msg["timestamp_ms"], msg["payload"])
        for msg in session["messages"]
        if msg["direction"] == "received"
    ]
    received_payloads.sort(key=_sort_by_timestamp)

    if not received_payloads:
        return ReplaySessionResultDict(
            session_id=session["session_id"],
            total_ticks=0,
            total_messages=0,
            traces=[],
        )

    traces: list[ReplayTickTraceDict] = []
    ai_state = make_initial_ai_state()
    tick_index = 0
    total_messages = 0

    # Group messages into ticks by timestamp proximity.
    # Messages within TICK_RATE_MS of the batch start are in the same tick.
    batch_start_ms = received_payloads[0][0]
    batch: list[str] = []

    for timestamp_ms, payload in received_payloads:
        if timestamp_ms - batch_start_ms >= TICK_RATE_MS and batch:
            # Process this batch and run planner
            total_messages += len(batch)
            result = _process_tick_batch(
                batch,
                xor_table,
                ai_state,
                tick_index,
                batch_start_ms,
            )
            ai_state = result[0]
            if result[1] is not None:
                traces.append(result[1])
                tick_index += 1
            batch = []
            batch_start_ms = timestamp_ms
        batch.append(payload)

    # Process final batch — always non-empty because the for-loop above
    # appends at least one payload per iteration, and empty received_payloads
    # returns early before reaching this point.
    total_messages += len(batch)
    result = _process_tick_batch(batch, xor_table, ai_state, tick_index, batch_start_ms)
    if result[1] is not None:
        traces.append(result[1])

    return ReplaySessionResultDict(
        session_id=session["session_id"],
        total_ticks=len(traces),
        total_messages=total_messages,
        traces=traces,
    )


def _process_tick_batch(
    payloads: list[str],
    xor_table: bytes,
    ai_state: AIStateDict,
    tick_index: int,
    timestamp_ms: int,
) -> tuple[AIStateDict, ReplayTickTraceDict | None]:
    """Decode a batch of payloads and run the planner if ready.

    Args:
        payloads: Base64-encoded received message payloads.
        xor_table: The replayed session's XOR table.
        ai_state: Current AI state carried forward from the previous tick.
        tick_index: Current tick counter.
        timestamp_ms: Timestamp for this tick batch.

    Returns:
        Tuple of (updated AI state, trace or None if self_state unavailable).
    """
    for payload in payloads:
        _test_hooks.process_received_message_hook(payload, xor_table)

    # Merge kills from protocol into AI state
    new_kills = drain_killed_tank_ids(get_world_service())
    if new_kills:
        merged_kills = dict(ai_state["killed_tank_ids"])
        for tank_id in new_kills:
            merged_kills[str(tank_id)] = timestamp_ms
        ai_state = AIStateDict(**{**ai_state, "killed_tank_ids": merged_kills})

    world = get_world_state()
    self_state = world["self_state"]
    if self_state is None:
        return (ai_state, None)

    inventory = get_inventory_state(get_world_service())
    terrain = get_terrain_map()

    decision = ai_strategy.decide(
        world,
        self_state,
        ai_state,
        inventory,
        timestamp_ms,
        terrain,
        map_fuel_dots=get_world_service().map_fuel_dots,
    )

    updated_ai_state = decision["updated_ai_state"]
    trace = _build_trace(
        tick_index,
        timestamp_ms,
        decision,
        world,
        self_state,
        updated_ai_state,
    )

    return (updated_ai_state, trace)


def _extract_command_target(command: BotCommand) -> tuple[int, int]:
    """Extract target coordinates from a bot command.

    Args:
        command: Bot command (may or may not have target coordinates).

    Returns:
        Tuple of (target_x, target_y). Returns (0, 0) for commands
        without coordinates (radar, map_open).
    """
    if command["cmd_type"] == "radar":
        return (0, 0)
    if command["cmd_type"] == "map_open":
        return (0, 0)
    if command["cmd_type"] == "hold":
        return (0, 0)
    if command["cmd_type"] == "scope_shift":
        return (0, 0)
    # All remaining command types have target_x and target_y.
    # Narrow to a concrete type to satisfy mypy's strict union checking.
    targeted: (
        MoveCommandDict
        | ShootCommandDict
        | PickupFuelCommandDict
        | PickupEquipmentCommandDict
        | TeleportCommandDict
        | ChatCommandDict
    ) = command
    return (targeted["target_x"], targeted["target_y"])


def _build_trace(
    tick_index: int,
    timestamp_ms: int,
    decision: TickDecisionDict,
    world: WorldStateDict,
    self_state: SelfStateDict,
    ai_state: AIStateDict,
) -> ReplayTickTraceDict:
    """Build a ReplayTickTraceDict from planner output.

    Args:
        tick_index: Current tick counter.
        timestamp_ms: Timestamp for this tick.
        decision: Planner decision output.
        world: Current world state snapshot.
        self_state: Player state at decision time.
        ai_state: AI state after this tick's decision (carries durable
            mode ownership, combat target, and resource lock).

    Returns:
        Populated ReplayTickTraceDict.
    """
    behavior = decision["behavior"]
    command = decision["command"]
    threats = analyze_threats(world, self_state, timestamp_ms)
    target_x, target_y = _extract_command_target(command)

    return ReplayTickTraceDict(
        tick_index=tick_index,
        timestamp_ms=timestamp_ms,
        self_x=self_state["x"],
        self_y=self_state["y"],
        fuel=self_state["fuel"],
        behavior_mode=behavior["mode"],
        behavior_score=behavior["score"],
        behavior_reason=render_reason(behavior),
        ai_mode=ai_state["mode"],
        ai_mode_state=ai_state["mode_state"],
        command_type=command["cmd_type"],
        target_x=target_x,
        target_y=target_y,
        combat_target_id=ai_state["combat_target_id"],
        resource_target_kind=ai_state["resource_target_kind"],
        visible_threats=threats,
        container_count=len(world["containers"]),
    )


__all__ = [
    "replay_session",
]
