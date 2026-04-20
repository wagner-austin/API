"""Shared diagnostic helpers for the live fuel probe."""

from __future__ import annotations

from collections.abc import Callable

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.action_lab.action_trace import (
    build_fuel_decision_basis,
    format_fuel_decision_basis,
    format_fuel_decision_candidates,
)
from tankpit_bot.action_lab.fuel_probe_types import FuelProbeSessionDict
from tankpit_bot.action_lab.fuel_target_phase import FuelTargetPhaseProbeProtocol
from tankpit_bot.bot.ai.equipment import describe_container_search
from tankpit_bot.state.types import ContainerStateDict

log = get_logger(__name__)


def log_fuel_target_diagnostic(
    probe: FuelTargetPhaseProbeProtocol,
    *,
    radar_cycle_id: int,
    fuel_target: ContainerStateDict | None,
    terrain_provider: Callable[[], TerrainMapProtocol | None],
) -> None:
    """Emit one structured diagnostic line after radar target resolution.

    Args:
        probe: Probe exposing current world and self state.
        radar_cycle_id: Radar cycle used for the decision.
        fuel_target: Selected fuel target, if any.
        terrain_provider: Terrain lookup for decision-basis construction.

    Returns:
        None.
    """
    terrain = terrain_provider()
    self_state = probe.get_self_state()
    if terrain is None or self_state is None:
        log.info(
            "FUEL_TARGET_DIAGNOSTIC selected=%s summary=unavailable decision_basis=%s",
            "none" if fuel_target is None else f"({fuel_target['x']},{fuel_target['y']})",
            "unavailable",
        )
        return
    world = probe.get_world_state()
    selected = "none" if fuel_target is None else f"({fuel_target['x']},{fuel_target['y']})"
    summary = describe_container_search(
        world,
        self_state,
        terrain,
        want_fuel=True,
        allow_unreachable=True,
        minimum_volume=1,
    )
    decision_basis = build_fuel_decision_basis(
        world,
        self_x=self_state["x"],
        self_y=self_state["y"],
        radar_cycle_id=radar_cycle_id,
        terrain=terrain,
        fuel_target=fuel_target,
    )
    log.info(
        "FUEL_TARGET_DIAGNOSTIC selected=%s summary=%s decision_basis=%s",
        selected,
        summary,
        format_fuel_decision_basis(decision_basis),
    )


def format_visible_fuel_entries(
    probe: FuelTargetPhaseProbeProtocol,
    *,
    fuel_target: ContainerStateDict | None,
    terrain_provider: Callable[[], TerrainMapProtocol | None],
) -> str:
    """Format visible fuel candidates for diagnostics.

    Args:
        probe: Probe exposing current world and self state.
        fuel_target: Selected target for the current decision, if any.
        terrain_provider: Terrain lookup for decision-basis construction.

    Returns:
        ``"unavailable"`` when terrain or self state is missing, ``"none"``
        when no visible fuel is tracked, or a compact candidate summary.
    """
    terrain = terrain_provider()
    if terrain is None:
        return "unavailable"
    self_state = probe.get_self_state()
    if self_state is None:
        return "unavailable"
    basis = build_fuel_decision_basis(
        probe.get_world_state(),
        self_x=self_state["x"],
        self_y=self_state["y"],
        radar_cycle_id=0,
        terrain=terrain,
        fuel_target=fuel_target,
    )
    return format_fuel_decision_candidates(basis)


def format_fuel_probe_summary(session: FuelProbeSessionDict) -> str:
    """Format a compact summary for a fuel probe session.

    Args:
        session: Completed fuel-probe session payload.

    Returns:
        Human-readable one-line summary.
    """
    picked_up_fuel = 0
    no_fuel_visible = 0
    radar_timeout = 0
    map_sync_timeout = 0
    reposition_map_sync_timeout = 0
    teleport_timeout = 0
    reposition_teleport_timeout = 0
    pickup_timeout = 0
    for attempt in session["attempts"]:
        if attempt["status"] == "picked_up_fuel":
            picked_up_fuel += 1
        elif attempt["status"] == "no_fuel_visible":
            no_fuel_visible += 1
        elif attempt["status"] == "radar_timeout":
            radar_timeout += 1
        elif attempt["status"] == "map_sync_timeout":
            map_sync_timeout += 1
        elif attempt["status"] == "reposition_map_sync_timeout":
            reposition_map_sync_timeout += 1
        elif attempt["status"] == "teleport_timeout":
            teleport_timeout += 1
        elif attempt["status"] == "reposition_teleport_timeout":
            reposition_teleport_timeout += 1
        else:
            pickup_timeout += 1
    startup_timing = session["startup_timing"]
    bootstrap_ms = (
        startup_timing["command_ready_timestamp_ms"] - startup_timing["initial_sync_started_ms"]
    )
    return (
        "Fuel probe complete: "
        f"attempts={len(session['attempts'])} "
        f"target_pickups={session['target_pickups']} "
        f"picked_up_fuel={picked_up_fuel} "
        f"no_fuel_visible={no_fuel_visible} "
        f"radar_timeout={radar_timeout} "
        f"map_sync_timeout={map_sync_timeout} "
        f"reposition_map_sync_timeout={reposition_map_sync_timeout} "
        f"teleport_timeout={teleport_timeout} "
        f"reposition_teleport_timeout={reposition_teleport_timeout} "
        f"pickup_timeout={pickup_timeout} "
        "session_to_initial_sync_ms="
        f"{startup_timing['initial_sync_started_ms'] - session['start_timestamp_ms']} "
        f"initial_sync_to_command_ready_ms={bootstrap_ms}"
    )


__all__ = [
    "format_fuel_probe_summary",
    "format_visible_fuel_entries",
    "log_fuel_target_diagnostic",
]
