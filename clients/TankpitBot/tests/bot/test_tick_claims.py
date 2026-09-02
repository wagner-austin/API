"""Tests for the tick's collect-claim arbitration.

The gate between decide and execute ([[fleet-forage-allocation]]):
whatever plan a decision wants to persist must own its container's
authoritative claim file before any command dispatches. Claims ride
the fake filesystem; the AI state rides the real intent layer.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import dump_json_str, load_json_str

from tankpit_bot.bot.ai.intent import set_resource_target
from tankpit_bot.bot.ai.scoring_types import make_behavior_score
from tankpit_bot.bot.ai.types import AIStateDict, make_initial_ai_state
from tankpit_bot.bot.tick_claims import _arbitrate_collect_claim, _drop_held_claim
from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision
from tankpit_bot.bot.types import make_hold_command
from tankpit_bot.fleetshare.claims import (
    ContainerClaimDict,
    claim_path,
    decode_container_claim,
    encode_container_claim,
)
from tankpit_bot.sniffer.world_service import WorldService
from tests.conftest import FakeEnv, FakeFileSystem

_NOW = 500_000
_TANK_ID = 2731


def _entered_ws() -> WorldService:
    """Build a world service inside room 6 with nothing claimed."""
    ws = WorldService()
    ws.set_selected_room("6")
    return ws


def _decision(ai_state: AIStateDict) -> TickDecisionDict:
    """Build a tick decision carrying the given AI state.

    The command is a hold — the arbitration judges the PLAN in the
    state, never the command, so the simplest command keeps the
    fixtures honest.

    Args:
        ai_state: The AI state the decision wants to persist.

    Returns:
        A minimal tick decision.
    """
    return make_tick_decision(
        command=make_hold_command(),
        behavior=make_behavior_score("COLLECT", 100, 10, 20, "fuel_locked"),
        updated_ai_state=ai_state,
        desired_equipment=[1, 2],
    )


def _planned_state(tx: int = 10, ty: int = 20) -> AIStateDict:
    """AI state holding a fuel collect plan for (tx, ty)."""
    return set_resource_target(make_initial_ai_state(), "fuel", tx, ty)


def _plant_claim(fs: FakeFileSystem, instance: str, tx: int = 10, ty: int = 20) -> None:
    """Write an existing fresh claim into the fake filesystem."""
    claim = ContainerClaimDict(instance=instance, tank_id=99, claimed_ms=_NOW - 1_000)
    fs.write_text(claim_path("6", tx, ty), dump_json_str(encode_container_claim(claim)))


def _claim_stamp(fs: FakeFileSystem, tx: int = 10, ty: int = 20) -> int:
    """Read the claim file's stamp back through the codec."""
    parsed = load_json_str(fs.read_text(claim_path("6", tx, ty)))
    assert isinstance(parsed, dict)
    return decode_container_claim(parsed)["claimed_ms"]


def test_no_plan_and_nothing_held_passes_through(
    fake_env: FakeEnv, fake_fs: FakeFileSystem
) -> None:
    """A planless tick neither claims nor releases anything."""
    ws = _entered_ws()
    decision = _decision(make_initial_ai_state())

    result = _arbitrate_collect_claim(ws, decision, _TANK_ID, _NOW)

    assert result is decision
    assert not fake_fs.path_exists(claim_path("6", 10, 20))


def test_a_dropped_plan_releases_the_held_claim(fake_env: FakeEnv, fake_fs: FakeFileSystem) -> None:
    """When the plan is gone the claim file goes with it."""
    ws = _entered_ws()
    _plant_claim(fake_fs, "")
    ws.held_claim_x = 10
    ws.held_claim_y = 20
    decision = _decision(make_initial_ai_state())

    result = _arbitrate_collect_claim(ws, decision, _TANK_ID, _NOW)

    assert result is decision
    assert not fake_fs.path_exists(claim_path("6", 10, 20))
    assert (ws.held_claim_x, ws.held_claim_y) == (-1, -1)


def test_a_new_plan_wins_its_claim(fake_env: FakeEnv, fake_fs: FakeFileSystem) -> None:
    """An unclaimed container is granted and remembered as held."""
    ws = _entered_ws()
    decision = _decision(_planned_state())

    result = _arbitrate_collect_claim(ws, decision, _TANK_ID, _NOW)

    assert result is decision
    assert (ws.held_claim_x, ws.held_claim_y) == (10, 20)
    assert _claim_stamp(fake_fs) == _NOW


def test_a_held_plan_refreshes_its_claim_each_tick(
    fake_env: FakeEnv, fake_fs: FakeFileSystem
) -> None:
    """The holder's stamp advances so the claim never reads stale."""
    ws = _entered_ws()
    _plant_claim(fake_fs, "")
    ws.held_claim_x = 10
    ws.held_claim_y = 20
    decision = _decision(_planned_state())

    result = _arbitrate_collect_claim(ws, decision, _TANK_ID, _NOW)

    assert result is decision
    assert _claim_stamp(fake_fs) == _NOW
    assert (ws.held_claim_x, ws.held_claim_y) == (10, 20)


def test_a_switched_plan_releases_the_old_claim_and_wins_the_new(
    fake_env: FakeEnv, fake_fs: FakeFileSystem
) -> None:
    """Replanning moves the claim with the plan, atomically per tick."""
    ws = _entered_ws()
    _plant_claim(fake_fs, "", tx=90, ty=90)
    ws.held_claim_x = 90
    ws.held_claim_y = 90
    decision = _decision(_planned_state())

    result = _arbitrate_collect_claim(ws, decision, _TANK_ID, _NOW)

    assert result is decision
    assert not fake_fs.path_exists(claim_path("6", 90, 90))
    assert _claim_stamp(fake_fs) == _NOW
    assert (ws.held_claim_x, ws.held_claim_y) == (10, 20)


def test_a_sibling_owned_container_kills_the_plan_before_dispatch(
    fake_env: FakeEnv, fake_fs: FakeFileSystem
) -> None:
    """The denial arm: hold command, plan released, tile bridged.

    The whole point of the mutex — the loser pays one held beat here
    instead of the journey the contention measurement priced.
    """
    ws = _entered_ws()
    _plant_claim(fake_fs, "yuppler")
    decision = _decision(_planned_state())

    result = _arbitrate_collect_claim(ws, decision, _TANK_ID, _NOW)

    assert result["command"]["cmd_type"] == "hold"
    assert result["behavior"]["reason_kind"] == "claim_denied"
    assert result["updated_ai_state"]["resource_target_kind"] == ""
    assert result["desired_equipment"] == [1, 2]
    assert "10,20" in ws.fleet_claimed_containers
    assert (ws.held_claim_x, ws.held_claim_y) == (-1, -1)


def test_a_roomless_session_passes_through_unclaimed(
    fake_env: FakeEnv, fake_fs: FakeFileSystem
) -> None:
    """No selected room means no fleet channel and nobody to contend.

    The same scope law as ``build_fleet_report``'s pre-join return —
    the sim seam's direct-entry harness runs whole sessions this way,
    and its plans must dispatch exactly as before the mutex existed.
    """
    ws = WorldService()
    decision = _decision(_planned_state())

    result = _arbitrate_collect_claim(ws, decision, _TANK_ID, _NOW)

    assert result is decision
    assert not fake_fs.path_exists(claim_path("6", 10, 20))


def test_a_held_claim_with_no_room_is_a_broken_invariant(
    fake_env: FakeEnv, fake_fs: FakeFileSystem
) -> None:
    """Claims are acquired in-room only; a roomless hold must raise."""
    ws = WorldService()
    ws.held_claim_x = 10
    ws.held_claim_y = 20

    with pytest.raises(ValueError, match="claims are acquired in-room only"):
        _drop_held_claim(ws, "")
