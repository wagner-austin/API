"""Tests for action-lab phase tracing and decision-basis TypedDict codecs."""

from __future__ import annotations

from typing import Literal

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.action_lab.action_trace_types import (
    ActionPhaseCycleDict,
    ActionPhaseName,
    ActionPhaseOverlapDict,
    FuelDecisionBasisDict,
    FuelDecisionCandidateDict,
    decode_action_phase_cycle,
    decode_action_phase_name,
    decode_action_phase_overlap,
    decode_fuel_decision_basis,
    decode_fuel_decision_candidate,
    encode_action_phase_cycle,
    encode_action_phase_overlap,
    encode_fuel_decision_basis,
    encode_fuel_decision_candidate,
)


def _sample_cycle(phase: ActionPhaseName = "teleport") -> ActionPhaseCycleDict:
    """Build a sample action phase cycle."""
    return ActionPhaseCycleDict(phase=phase, cycle_id=3, started_ms=1200)


def _sample_overlap() -> ActionPhaseOverlapDict:
    """Build a sample action phase overlap."""
    return ActionPhaseOverlapDict(
        active_phase="radar",
        active_cycle_id=2,
        active_started_ms=1400,
        next_phase="move",
        next_cycle_id=3,
        next_started_ms=1500,
    )


def _sample_candidate(
    source: ActionPhaseName | None = None,
) -> FuelDecisionCandidateDict:
    """Build a sample fuel decision candidate."""
    resolved_source = "viewport" if source is None else _candidate_source_from_phase(source)
    return FuelDecisionCandidateDict(
        x=140,
        y=109,
        volume=380,
        failed_pickups=1,
        selected=True,
        actionable=False,
        reason="stale",
        source=resolved_source,
        refresh_kind="radar_cache_refresh",
        refresh_timestamp_ms=4200,
        stale_age_ms=9500,
        stale_ttl_ms=3000,
    )


def _candidate_source_from_phase(
    phase: ActionPhaseName,
) -> Literal["viewport", "radar", "world_state"]:
    """Map a phase test variant to one supported entity source."""
    if phase == "teleport":
        return "viewport"
    if phase == "radar":
        return "radar"
    if phase == "move":
        return "world_state"
    return "viewport"


def _sample_basis() -> FuelDecisionBasisDict:
    """Build a sample decision basis."""
    return FuelDecisionBasisDict(
        world_timestamp_ms=5000,
        radar_cycle_id=7,
        viewport_left=131,
        viewport_top=102,
        self_x=147,
        self_y=110,
        selected_target_x=140,
        selected_target_y=109,
        candidates=[_sample_candidate()],
    )


@pytest.mark.parametrize("phase", ["teleport", "radar", "move", "pickup"])
def test_decode_action_phase_name_accepts_all_supported_values(phase: ActionPhaseName) -> None:
    """Action phase decode accepts every supported phase."""
    assert decode_action_phase_name({"phase": phase}, "phase") == phase


def test_decode_action_phase_name_rejects_invalid_value() -> None:
    """Action phase decode rejects unsupported values."""
    with pytest.raises(JSONTypeError, match="invalid action phase"):
        decode_action_phase_name({"phase": "bad"}, "phase")


def test_action_phase_cycle_round_trip() -> None:
    """Action phase cycles encode and decode cleanly."""
    cycle = _sample_cycle("pickup")
    assert decode_action_phase_cycle(encode_action_phase_cycle(cycle)) == cycle


def test_action_phase_overlap_round_trip() -> None:
    """Action phase overlaps encode and decode cleanly."""
    overlap = _sample_overlap()
    assert decode_action_phase_overlap(encode_action_phase_overlap(overlap)) == overlap


@pytest.mark.parametrize("phase", ["teleport", "radar", "move"])
def test_fuel_decision_candidate_round_trip_for_all_sources(phase: ActionPhaseName) -> None:
    """Fuel decision candidates preserve all supported entity sources."""
    candidate = _sample_candidate(phase)
    assert decode_fuel_decision_candidate(encode_fuel_decision_candidate(candidate)) == candidate


def test_decode_fuel_decision_candidate_rejects_invalid_source() -> None:
    """Fuel decision candidates reject unsupported entity sources."""
    encoded = encode_fuel_decision_candidate(_sample_candidate())
    encoded["source"] = "bad"

    with pytest.raises(JSONTypeError, match="invalid entity source"):
        decode_fuel_decision_candidate(encoded)


def test_fuel_decision_basis_round_trip_with_null_selected_target() -> None:
    """Decision basis codecs preserve null selected target coordinates."""
    basis = FuelDecisionBasisDict(
        world_timestamp_ms=5000,
        radar_cycle_id=7,
        viewport_left=131,
        viewport_top=102,
        self_x=147,
        self_y=110,
        selected_target_x=None,
        selected_target_y=None,
        candidates=[_sample_candidate()],
    )

    assert decode_fuel_decision_basis(encode_fuel_decision_basis(basis)) == basis


def test_decode_fuel_decision_basis_rejects_non_object_candidate() -> None:
    """Decision basis decode rejects non-object candidate entries."""
    encoded = encode_fuel_decision_basis(_sample_basis())
    encoded["candidates"] = ["bad"]

    with pytest.raises(JSONTypeError, match="candidates"):
        decode_fuel_decision_basis(encoded)


def test_decode_fuel_decision_basis_rejects_invalid_optional_selected_target() -> None:
    """Decision basis decode rejects non-integer selected target coordinates."""
    encoded = encode_fuel_decision_basis(_sample_basis())
    encoded["selected_target_x"] = True

    with pytest.raises(JSONTypeError, match="selected_target_x"):
        decode_fuel_decision_basis(encoded)
