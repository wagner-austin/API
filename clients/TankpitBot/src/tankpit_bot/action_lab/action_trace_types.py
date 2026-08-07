"""TypedDict models for action-lab phase tracing and fuel decision diagnostics."""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_bool,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

from tankpit_bot.types.constants import (
    ContainerRefreshKind,
    decode_container_refresh_kind,
    encode_container_refresh_kind,
)

ActionPhaseName = Literal["teleport", "radar", "move", "pickup"]


def decode_action_phase_name(data: JSONObject, field: str) -> ActionPhaseName:
    """Decode an action phase name from a JSON object.

    Args:
        data: Source JSON object.
        field: Field name to validate.

    Returns:
        Validated action phase name.

    Raises:
        JSONTypeError: If the field is not a supported action phase.
    """
    raw = require_str(data, field)
    if raw == "teleport":
        return "teleport"
    if raw == "radar":
        return "radar"
    if raw == "move":
        return "move"
    if raw == "pickup":
        return "pickup"
    raise JSONTypeError(f"Field '{field}' has invalid action phase: {raw}")


class ActionPhaseCycleDict(TypedDict):
    """One started action phase cycle."""

    phase: ActionPhaseName
    cycle_id: int
    started_ms: int


class ActionPhaseOverlapDict(TypedDict):
    """Invariant violation recorded when phases overlap."""

    active_phase: ActionPhaseName
    active_cycle_id: int
    active_started_ms: int
    next_phase: ActionPhaseName
    next_cycle_id: int
    next_started_ms: int


class FuelDecisionCandidateDict(TypedDict):
    """Fuel-selection diagnostic for one visible candidate."""

    x: int
    y: int
    volume: int
    failed_pickups: int
    selected: bool
    actionable: bool
    reason: str
    source: Literal["viewport", "radar", "world_state"]
    refresh_kind: ContainerRefreshKind
    refresh_timestamp_ms: int
    age_ms: int


class FuelDecisionBasisDict(TypedDict):
    """Decision-basis snapshot for one radar-driven fuel selection."""

    world_timestamp_ms: int
    radar_cycle_id: int
    viewport_left: int
    viewport_top: int
    self_x: int
    self_y: int
    selected_target_x: int | None
    selected_target_y: int | None
    candidates: list[FuelDecisionCandidateDict]


def _encode_optional_int(value: int | None) -> JSONValue:
    """Encode an optional integer."""
    return value


def encode_action_phase_cycle(cycle: ActionPhaseCycleDict) -> JSONObject:
    """Encode an action phase cycle."""
    return {
        "phase": cycle["phase"],
        "cycle_id": cycle["cycle_id"],
        "started_ms": cycle["started_ms"],
    }


def decode_action_phase_cycle(data: JSONObject) -> ActionPhaseCycleDict:
    """Decode an action phase cycle."""
    return ActionPhaseCycleDict(
        phase=decode_action_phase_name(data, "phase"),
        cycle_id=require_int(data, "cycle_id"),
        started_ms=require_int(data, "started_ms"),
    )


def encode_action_phase_overlap(overlap: ActionPhaseOverlapDict) -> JSONObject:
    """Encode an action phase overlap event."""
    return {
        "active_phase": overlap["active_phase"],
        "active_cycle_id": overlap["active_cycle_id"],
        "active_started_ms": overlap["active_started_ms"],
        "next_phase": overlap["next_phase"],
        "next_cycle_id": overlap["next_cycle_id"],
        "next_started_ms": overlap["next_started_ms"],
    }


def decode_action_phase_overlap(data: JSONObject) -> ActionPhaseOverlapDict:
    """Decode an action phase overlap event."""
    return ActionPhaseOverlapDict(
        active_phase=decode_action_phase_name(data, "active_phase"),
        active_cycle_id=require_int(data, "active_cycle_id"),
        active_started_ms=require_int(data, "active_started_ms"),
        next_phase=decode_action_phase_name(data, "next_phase"),
        next_cycle_id=require_int(data, "next_cycle_id"),
        next_started_ms=require_int(data, "next_started_ms"),
    )


def encode_fuel_decision_candidate(candidate: FuelDecisionCandidateDict) -> JSONObject:
    """Encode one fuel decision candidate."""
    return {
        "x": candidate["x"],
        "y": candidate["y"],
        "volume": candidate["volume"],
        "failed_pickups": candidate["failed_pickups"],
        "selected": candidate["selected"],
        "actionable": candidate["actionable"],
        "reason": candidate["reason"],
        "source": candidate["source"],
        "refresh_kind": encode_container_refresh_kind(candidate["refresh_kind"]),
        "refresh_timestamp_ms": candidate["refresh_timestamp_ms"],
        "age_ms": candidate["age_ms"],
    }


def decode_fuel_decision_candidate(data: JSONObject) -> FuelDecisionCandidateDict:
    """Decode one fuel decision candidate."""
    return FuelDecisionCandidateDict(
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        volume=require_int(data, "volume"),
        failed_pickups=require_int(data, "failed_pickups"),
        selected=require_bool(data, "selected"),
        actionable=require_bool(data, "actionable"),
        reason=require_str(data, "reason"),
        source=_decode_entity_source(data, "source"),
        refresh_kind=decode_container_refresh_kind(data, "refresh_kind"),
        refresh_timestamp_ms=require_int(data, "refresh_timestamp_ms"),
        age_ms=require_int(data, "age_ms"),
    )


def _decode_entity_source(
    data: JSONObject,
    field: str,
) -> Literal["viewport", "radar", "world_state"]:
    """Decode a coarse entity source for a fuel candidate."""
    raw = require_str(data, field)
    if raw == "viewport":
        return "viewport"
    if raw == "radar":
        return "radar"
    if raw == "world_state":
        return "world_state"
    raise JSONTypeError(f"Field '{field}' has invalid entity source: {raw}")


def encode_fuel_decision_basis(basis: FuelDecisionBasisDict) -> JSONObject:
    """Encode a full fuel decision basis."""
    encoded_candidates: list[JSONValue] = [
        encode_fuel_decision_candidate(candidate) for candidate in basis["candidates"]
    ]
    return {
        "world_timestamp_ms": basis["world_timestamp_ms"],
        "radar_cycle_id": basis["radar_cycle_id"],
        "viewport_left": basis["viewport_left"],
        "viewport_top": basis["viewport_top"],
        "self_x": basis["self_x"],
        "self_y": basis["self_y"],
        "selected_target_x": _encode_optional_int(basis["selected_target_x"]),
        "selected_target_y": _encode_optional_int(basis["selected_target_y"]),
        "candidates": encoded_candidates,
    }


def _decode_fuel_decision_candidates(raw: JSONValue) -> list[FuelDecisionCandidateDict]:
    """Decode a list of fuel decision candidates."""
    result: list[FuelDecisionCandidateDict] = []
    for item in require_list({"items": raw}, "items"):
        if not isinstance(item, dict):
            raise JSONTypeError("Field 'candidates' must contain only objects")
        result.append(decode_fuel_decision_candidate(item))
    return result


def _require_optional_int(data: JSONObject, field: str) -> int | None:
    """Decode an optional integer field."""
    raw = data.get(field)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise JSONTypeError(f"Field '{field}' must be an integer or null")
    return raw


def decode_fuel_decision_basis(data: JSONObject) -> FuelDecisionBasisDict:
    """Decode a full fuel decision basis."""
    return FuelDecisionBasisDict(
        world_timestamp_ms=require_int(data, "world_timestamp_ms"),
        radar_cycle_id=require_int(data, "radar_cycle_id"),
        viewport_left=require_int(data, "viewport_left"),
        viewport_top=require_int(data, "viewport_top"),
        self_x=require_int(data, "self_x"),
        self_y=require_int(data, "self_y"),
        selected_target_x=_require_optional_int(data, "selected_target_x"),
        selected_target_y=_require_optional_int(data, "selected_target_y"),
        candidates=_decode_fuel_decision_candidates(data.get("candidates")),
    )


__all__ = [
    "ActionPhaseCycleDict",
    "ActionPhaseName",
    "ActionPhaseOverlapDict",
    "FuelDecisionBasisDict",
    "FuelDecisionCandidateDict",
    "decode_action_phase_cycle",
    "decode_action_phase_name",
    "decode_action_phase_overlap",
    "decode_fuel_decision_basis",
    "decode_fuel_decision_candidate",
    "encode_action_phase_cycle",
    "encode_action_phase_overlap",
    "encode_fuel_decision_basis",
    "encode_fuel_decision_candidate",
]
