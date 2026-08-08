"""Enemy selection and result shaping for the enemy-teleport probe.

Picking a fresh enemy threat, resolving one by id, and rendering the
terminal result and run summary. The probe that drives them is
:mod:`tankpit_bot.action_lab.enemy_teleport`.
"""

from __future__ import annotations

from typing import Literal

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.enemy_teleport_types import (
    EnemyTeleportAttemptResultDict,
    EnemyTeleportProbeSessionDict,
)
from tankpit_bot.action_lab.probe_base import ProbeBase
from tankpit_bot.action_lab.types import TeleportTargetDict
from tankpit_bot.bot.ai.threat_primitives import find_closest_threat
from tankpit_bot.bot.ai.threats import analyze_threats
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.browser.page_client_snapshot import (
    PageClientSnapshotDict,
)


def _require_fresh_enemy_threat(
    probe: ProbeBase,
    started_ms: int,
    excluded_tank_ids: frozenset[int],
) -> EnemyThreatDict | None:
    """Return the closest enemy threat confirmed after a probe action."""
    self_state = probe.get_self_state()
    if self_state is None:
        return None
    threats = analyze_threats(
        probe.world, probe.get_world_state(), self_state, action_hooks.get_current_time_ms()
    )
    fresh = [
        threat
        for threat in threats
        if threat["timestamp_ms"] > started_ms and threat["tank_id"] not in excluded_tank_ids
    ]
    return find_closest_threat(fresh)


def _enemy_by_id(probe: ProbeBase, tank_id: int) -> EnemyThreatDict | None:
    """Return the current threat snapshot for a specific tank id."""
    self_state = probe.get_self_state()
    if self_state is None:
        return None
    for threat in analyze_threats(
        probe.world, probe.get_world_state(), self_state, action_hooks.get_current_time_ms()
    ):
        if threat["tank_id"] == tank_id:
            return threat
    return None


def _format_enemy_label(enemy: EnemyThreatDict) -> str:
    """Return a deterministic teleport target label for an enemy landing."""
    return f"enemy_{enemy['tank_id']}_{enemy['x']}_{enemy['y']}"


def _make_terminal_result(
    *,
    acquisition_strategy: Literal["map_open", "nearest_enemy"],
    status: Literal["no_enemy", "no_landing_tile", "acquisition_timeout"],
    acquisition_started_ms: int,
    acquisition_sync_timestamp_ms: int | None,
    fuel_before: int,
    world_timestamp_before: int,
    completion_timestamp_ms: int,
    fuel_after: int,
    world_timestamp_after: int,
    enemy: EnemyThreatDict | None,
    landing_target: TeleportTargetDict | None,
    landed_x: int,
    landed_y: int,
    message_start_index: int,
    message_end_index: int,
    snapshot_before: PageClientSnapshotDict,
    snapshot_after: PageClientSnapshotDict,
) -> EnemyTeleportAttemptResultDict:
    """Build a non-teleport terminal enemy-teleport result."""
    return EnemyTeleportAttemptResultDict(
        acquisition_strategy=acquisition_strategy,
        status=status,
        acquisition_started_ms=acquisition_started_ms,
        acquisition_sync_timestamp_ms=acquisition_sync_timestamp_ms,
        teleport_started_ms=None,
        completion_timestamp_ms=completion_timestamp_ms,
        acquisition_elapsed_ms=(
            None
            if acquisition_sync_timestamp_ms is None
            else acquisition_sync_timestamp_ms - acquisition_started_ms
        ),
        teleport_elapsed_ms=None,
        fuel_before=fuel_before,
        fuel_after=fuel_after,
        world_timestamp_before=world_timestamp_before,
        world_timestamp_after=world_timestamp_after,
        enemy=enemy,
        landing_target=landing_target,
        landed_signal_received=False,
        landed_x=landed_x,
        landed_y=landed_y,
        enemy_still_visible=False,
        enemy_distance_after=None,
        enemy_x_after=None,
        enemy_y_after=None,
        message_start_index=message_start_index,
        message_end_index=message_end_index,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


def format_enemy_teleport_probe_summary(session: EnemyTeleportProbeSessionDict) -> str:
    """Format a compact human-readable summary line for the session."""
    landed_adjacent = 0
    landed_not_adjacent = 0
    no_enemy = 0
    no_landing_tile = 0
    acquisition_timeout = 0
    teleport_timeout = 0
    for attempt in session["attempts"]:
        if attempt["status"] == "landed_adjacent":
            landed_adjacent += 1
        elif attempt["status"] == "landed_not_adjacent":
            landed_not_adjacent += 1
        elif attempt["status"] == "no_enemy":
            no_enemy += 1
        elif attempt["status"] == "no_landing_tile":
            no_landing_tile += 1
        elif attempt["status"] == "acquisition_timeout":
            acquisition_timeout += 1
        else:
            teleport_timeout += 1
    startup_timing = session["startup_timing"]
    bootstrap_ms = (
        startup_timing["command_ready_timestamp_ms"] - startup_timing["initial_sync_started_ms"]
    )
    return (
        "Enemy teleport probe complete: "
        f"strategy={session['acquisition_strategy']} "
        f"attempts={len(session['attempts'])} "
        f"landed_adjacent={landed_adjacent} "
        f"landed_not_adjacent={landed_not_adjacent} "
        f"no_enemy={no_enemy} "
        f"no_landing_tile={no_landing_tile} "
        f"acquisition_timeout={acquisition_timeout} "
        f"teleport_timeout={teleport_timeout} "
        "session_to_initial_sync_ms="
        f"{startup_timing['initial_sync_started_ms'] - session['start_timestamp_ms']} "
        f"initial_sync_to_command_ready_ms={bootstrap_ms}"
    )


__all__ = [
    "format_enemy_teleport_probe_summary",
]
