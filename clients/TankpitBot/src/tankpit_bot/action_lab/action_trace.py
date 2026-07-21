"""Shared phase tracing and fuel-decision diagnostics for action-lab probes."""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.action_lab.action_trace_types import (
    ActionPhaseCycleDict,
    ActionPhaseName,
    ActionPhaseOverlapDict,
    FuelDecisionBasisDict,
    FuelDecisionCandidateDict,
)
from tankpit_bot.bot.ai.equipment import _viewport_bounds
from tankpit_bot.bot.ai.equipment_search import _describe_candidate_reason
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.state.types import ContainerStateDict, WorldStateDict

log = get_logger(__name__)


class ActionCycleTracker:
    """Strict phase-cycle tracker for action-lab command phases."""

    def __init__(self) -> None:
        """Initialize an empty cycle tracker."""
        self._next_cycle_id_by_phase: dict[ActionPhaseName, int] = {
            "teleport": 0,
            "radar": 0,
            "move": 0,
            "pickup": 0,
        }
        self._active_cycles: dict[ActionPhaseName, ActionPhaseCycleDict] = {}

    def reset(self) -> None:
        """Reset all cycle counters and active phases."""
        self._next_cycle_id_by_phase = {
            "teleport": 0,
            "radar": 0,
            "move": 0,
            "pickup": 0,
        }
        self._active_cycles = {}

    def begin_phase(
        self,
        phase: ActionPhaseName,
        *,
        started_ms: int,
    ) -> tuple[ActionPhaseCycleDict, list[ActionPhaseOverlapDict]]:
        """Start a new action phase cycle.

        Args:
            phase: Phase being started.
            started_ms: Local start timestamp.

        Returns:
            The started cycle and any overlap violations against currently
            active phases.
        """
        next_cycle_id = self._next_cycle_id_by_phase[phase] + 1
        self._next_cycle_id_by_phase[phase] = next_cycle_id
        cycle = ActionPhaseCycleDict(
            phase=phase,
            cycle_id=next_cycle_id,
            started_ms=started_ms,
        )
        overlaps: list[ActionPhaseOverlapDict] = []
        for active_cycle in self._active_cycles.values():
            overlaps.append(
                ActionPhaseOverlapDict(
                    active_phase=active_cycle["phase"],
                    active_cycle_id=active_cycle["cycle_id"],
                    active_started_ms=active_cycle["started_ms"],
                    next_phase=phase,
                    next_cycle_id=next_cycle_id,
                    next_started_ms=started_ms,
                )
            )
        self._active_cycles[phase] = cycle
        return cycle, overlaps

    def end_phase(self, cycle: ActionPhaseCycleDict) -> None:
        """End an active phase cycle.

        Args:
            cycle: Cycle to close.

        Raises:
            ValueError: If the requested cycle is not the current active cycle
                for its phase.
        """
        active_cycle = self._active_cycles.get(cycle["phase"])
        if active_cycle is None:
            raise ValueError(f"phase {cycle['phase']} is not active")
        if active_cycle["cycle_id"] != cycle["cycle_id"]:
            raise ValueError(
                f"phase {cycle['phase']} active cycle mismatch: "
                f"expected {active_cycle['cycle_id']}, got {cycle['cycle_id']}"
            )
        del self._active_cycles[cycle["phase"]]


def log_phase_overlaps(
    overlaps: list[ActionPhaseOverlapDict],
    *,
    attempt_label: str,
) -> None:
    """Emit one ``action_phase_overlap`` diagnostic per detected overlap.

    Args:
        overlaps: Overlap violations to emit.
        attempt_label: Human-readable attempt label for log correlation.
    """
    for overlap in overlaps:
        emit_diagnostic(
            diagnostic_kind="action_phase_overlap",
            attempt=attempt_label,
            active_phase=overlap["active_phase"],
            active_cycle_id=overlap["active_cycle_id"],
            active_started_ms=overlap["active_started_ms"],
            next_phase=overlap["next_phase"],
            next_cycle_id=overlap["next_cycle_id"],
            next_started_ms=overlap["next_started_ms"],
        )


def build_fuel_decision_basis(
    world: WorldStateDict,
    *,
    self_x: int,
    self_y: int,
    radar_cycle_id: int,
    terrain: TerrainMapProtocol | None,
    fuel_target: ContainerStateDict | None,
) -> FuelDecisionBasisDict:
    """Build a full fuel-selection decision basis for the current viewport.

    Args:
        world: Current world state.
        self_x: Self X coordinate.
        self_y: Self Y coordinate.
        radar_cycle_id: Radar cycle associated with this decision.
        terrain: Active terrain map, if available.
        fuel_target: Selected target, if any.

    Returns:
        Typed decision basis covering all visible fuel candidates.
    """
    left, top, right, bottom = _viewport_bounds(world)
    now_ms = world["timestamp_ms"]
    candidates: list[FuelDecisionCandidateDict] = []
    for container in world["containers"].values():
        if not container["is_fuel"]:
            continue
        x = container["x"]
        y = container["y"]
        if not (left <= x <= right and top <= y <= bottom):
            continue
        age_ms = now_ms - container["timestamp_ms"]
        reason, actionable, _, _, _ = _describe_candidate_reason(
            world,
            container,
            self_x,
            self_y,
            terrain,
            want_fuel=True,
            minimum_volume=1,
        )
        selected = fuel_target is not None and fuel_target["x"] == x and fuel_target["y"] == y
        candidates.append(
            FuelDecisionCandidateDict(
                x=x,
                y=y,
                volume=container["volume"],
                failed_pickups=container["failed_pickups"],
                selected=selected,
                actionable=actionable,
                reason=reason,
                source=container["source"],
                refresh_kind=container["refresh_kind"],
                refresh_timestamp_ms=container["timestamp_ms"],
                age_ms=age_ms,
            )
        )
    candidates.sort(key=_fuel_candidate_sort_key)
    viewport = world["viewport"]
    return FuelDecisionBasisDict(
        world_timestamp_ms=world["timestamp_ms"],
        radar_cycle_id=radar_cycle_id,
        viewport_left=viewport["left"],
        viewport_top=viewport["top"],
        self_x=self_x,
        self_y=self_y,
        selected_target_x=None if fuel_target is None else fuel_target["x"],
        selected_target_y=None if fuel_target is None else fuel_target["y"],
        candidates=candidates,
    )


def _fuel_candidate_sort_key(candidate: FuelDecisionCandidateDict) -> tuple[int, int, int]:
    """Return a deterministic sort key for logged fuel candidates."""
    return (candidate["y"], candidate["x"], -candidate["volume"])


def format_fuel_decision_basis(basis: FuelDecisionBasisDict) -> str:
    """Format a compact fuel-selection decision basis for logs.

    Args:
        basis: Decision basis to format.

    Returns:
        One-line compact representation.
    """
    formatted_candidates = format_fuel_decision_candidates(basis)
    if formatted_candidates == "none":
        return (
            f"world_ts={basis['world_timestamp_ms']} radar_cycle={basis['radar_cycle_id']} "
            f"viewport=({basis['viewport_left']},{basis['viewport_top']}) "
            f"self=({basis['self_x']},{basis['self_y']}) candidates=none"
        )
    return (
        f"world_ts={basis['world_timestamp_ms']} radar_cycle={basis['radar_cycle_id']} "
        f"viewport=({basis['viewport_left']},{basis['viewport_top']}) "
        f"self=({basis['self_x']},{basis['self_y']}) "
        f"selected=({basis['selected_target_x']},{basis['selected_target_y']}) "
        f"candidates={formatted_candidates}"
    )


def format_fuel_decision_candidates(
    basis: FuelDecisionBasisDict,
    *,
    limit: int = 8,
) -> str:
    """Format only the visible fuel candidates from one decision basis.

    Args:
        basis: Decision basis to summarize.
        limit: Maximum number of candidates to include before truncation.

    Returns:
        Compact candidate-only summary, or ``"none"`` when no candidates exist.

    Raises:
        ValueError: If ``limit`` is not positive.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not basis["candidates"]:
        return "none"
    formatted_candidates: list[str] = []
    for candidate in basis["candidates"][:limit]:
        formatted_candidates.append(
            f"({candidate['x']},{candidate['y']})"
            f" v={candidate['volume']}"
            f" failed={candidate['failed_pickups']}"
            f" reason={candidate['reason']}"
            f" actionable={candidate['actionable']}"
            f" selected={candidate['selected']}"
            f" refresh={candidate['refresh_kind']}@{candidate['refresh_timestamp_ms']}"
            f" age={candidate['age_ms']}"
        )
    if len(basis["candidates"]) > limit:
        formatted_candidates.append(f"...+{len(basis['candidates']) - limit} more")
    return " | ".join(formatted_candidates)


__all__ = [
    "ActionCycleTracker",
    "build_fuel_decision_basis",
    "format_fuel_decision_basis",
    "format_fuel_decision_candidates",
    "log_phase_overlaps",
]
