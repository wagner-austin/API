"""Universal-invariant assertions on bot decisions.

Each invariant has a positive case (clean decision passes) and a
negative case (constructible bad decision fails) here. The pair is
the proof of correctness: the check passes when satisfied, raises
when not.

Invariants the production type system already proves unreachable
(unknown ``cmd_type`` / ``behavior.mode``) are deliberately absent
from :mod:`tests.scenarios._invariants`, so this file has no tests
for them either -- dead checks would mean dead tests.
"""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.types import BehaviorScoreDict, make_initial_ai_state
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import (
    BotCommand,
    MapOpenCommandDict,
    RadarCommandDict,
    ShootCommandDict,
    TeleportCommandDict,
)
from tankpit_bot.inventory import InventoryItem, InventoryState
from tests.scenarios._harness import BotScenario
from tests.scenarios._invariants import (
    assert_no_violations,
    check_all_universal_invariants,
    check_does_not_target_self,
    check_does_not_teleport_to_origin_sentinel,
    check_secondary_does_not_duplicate_primary,
    check_target_on_map,
)


def _full_inventory() -> InventoryState:
    """Construct a typical full inventory for negative-test fixtures.

    Returns:
        An :class:`InventoryState` with non-zero counts for every slot.
    """
    return InventoryState(
        armor_shields=InventoryItem(count=25, enabled=False),
        dual_shots=InventoryItem(count=25, enabled=True),
        missile_shots=InventoryItem(count=25, enabled=False),
        homing_shots=InventoryItem(count=25, enabled=True),
        extra_radars=InventoryItem(count=25, enabled=True),
    )


def _decision_with_command(
    primary: BotCommand,
    secondary: BotCommand | None = None,
) -> TickDecisionDict:
    """Construct a minimal :class:`TickDecisionDict` for invariant tests.

    Args:
        primary: Primary command for the decision.
        secondary: Optional secondary command.

    Returns:
        A typed decision dict suitable for invariant tests.
    """
    return TickDecisionDict(
        command=primary,
        secondary_command=secondary,
        behavior=BehaviorScoreDict(
            mode="HUNT",
            score=100,
            target_x=0,
            target_y=0,
            target_id=-1,
            reason_kind="manual_hold",
            reason_context={},
        ),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )


# ---------------------------------------------------------------------
# Positive: a freshly-decided scenario satisfies every invariant.
# ---------------------------------------------------------------------


def test_decided_scenario_satisfies_every_universal_invariant() -> None:
    """A baseline scenario's first decision passes every invariant."""
    scenario = BotScenario()
    scenario.place_self(x=100, y=100, fuel=800)

    decision = scenario.decide()
    self_state = scenario.self_state
    if self_state is None:
        pytest.fail("place_self must populate self_state")

    violations = check_all_universal_invariants(decision, self_state, scenario.inventory)
    assert violations == []


# ---------------------------------------------------------------------
# check_target_on_map
# ---------------------------------------------------------------------


def test_check_target_on_map_rejects_off_map_x() -> None:
    """A target X above 255 fails."""
    decision = _decision_with_command(
        TeleportCommandDict(cmd_type="teleport", target_x=999, target_y=10)
    )
    violation = check_target_on_map(decision)
    if violation is None:
        pytest.fail("off-map X must produce a violation")
    assert violation["invariant"] == "decision_target_on_map"
    assert "off the 0..255 map" in violation["detail"]


def test_check_target_on_map_rejects_off_map_y_negative() -> None:
    """A negative target Y fails."""
    decision = _decision_with_command(
        ShootCommandDict(cmd_type="shoot", target_x=10, target_y=-1, target_id=5)
    )
    violation = check_target_on_map(decision)
    if violation is None:
        pytest.fail("negative Y must produce a violation")
    assert violation["invariant"] == "decision_target_on_map"


def test_check_target_on_map_passes_for_commands_without_coords() -> None:
    """Radar and map_open carry no coords; they pass unconditionally."""
    radar_dec = _decision_with_command(RadarCommandDict(cmd_type="radar"))
    assert check_target_on_map(radar_dec) is None
    map_dec = _decision_with_command(MapOpenCommandDict(cmd_type="map_open"))
    assert check_target_on_map(map_dec) is None


def test_check_target_on_map_passes_for_valid_target() -> None:
    """Valid in-map targets pass."""
    decision = _decision_with_command(
        TeleportCommandDict(cmd_type="teleport", target_x=128, target_y=64)
    )
    assert check_target_on_map(decision) is None


# ---------------------------------------------------------------------
# check_does_not_target_self
# ---------------------------------------------------------------------


def test_check_does_not_target_self_rejects_self_shot() -> None:
    """Shooting at the bot's own tile fails."""
    decision = _decision_with_command(
        ShootCommandDict(cmd_type="shoot", target_x=100, target_y=100, target_id=-1)
    )
    scenario = BotScenario()
    scenario.place_self(x=100, y=100)
    self_state = scenario.self_state
    if self_state is None:
        pytest.fail("place_self must populate self_state")

    violation = check_does_not_target_self(decision, self_state)
    if violation is None:
        pytest.fail("self-target must produce a violation")
    assert violation["invariant"] == "decision_does_not_target_self"


def test_check_does_not_target_self_passes_for_other_tile() -> None:
    """Shooting at a different tile passes."""
    decision = _decision_with_command(
        ShootCommandDict(cmd_type="shoot", target_x=99, target_y=100, target_id=5)
    )
    scenario = BotScenario()
    scenario.place_self(x=100, y=100)
    self_state = scenario.self_state
    if self_state is None:
        pytest.fail("place_self must populate self_state")

    assert check_does_not_target_self(decision, self_state) is None


# NOTE: ``check_does_not_radar_with_zero_inventory`` was removed
# 2026-06-21. The bot's ``radar`` command in the AI layer is the same
# command for both variants -- the server routes it to the unlimited
# built-in 5x5 scan when ``extra_radars=0`` and to the inventory-
# consuming extended scan otherwise. The foraging mode
# (``bot/ai/forage.py:147``) deliberately fires radar with zero
# extras so the bot can still reveal containers. The invariant was
# unsound.


# ---------------------------------------------------------------------
# check_does_not_teleport_to_origin_sentinel
# ---------------------------------------------------------------------


def test_check_does_not_teleport_to_origin_sentinel_rejects_zero_zero() -> None:
    """Teleport target (0, 0) fails."""
    decision = _decision_with_command(
        TeleportCommandDict(cmd_type="teleport", target_x=0, target_y=0)
    )
    violation = check_does_not_teleport_to_origin_sentinel(decision)
    if violation is None:
        pytest.fail("teleport to (0,0) must produce a violation")
    assert violation["invariant"] == "decision_does_not_teleport_to_origin_sentinel"


def test_check_does_not_teleport_to_origin_sentinel_passes_for_real_tile() -> None:
    """Teleport target (1, 0) passes; only the exact sentinel is rejected."""
    decision = _decision_with_command(
        TeleportCommandDict(cmd_type="teleport", target_x=1, target_y=0)
    )
    assert check_does_not_teleport_to_origin_sentinel(decision) is None


# ---------------------------------------------------------------------
# check_secondary_does_not_duplicate_primary
# ---------------------------------------------------------------------


def test_check_secondary_does_not_duplicate_primary_rejects_duplicate() -> None:
    """A secondary identical to the primary fails."""
    primary = RadarCommandDict(cmd_type="radar")
    secondary = RadarCommandDict(cmd_type="radar")
    decision = _decision_with_command(primary, secondary=secondary)

    violation = check_secondary_does_not_duplicate_primary(decision)
    if violation is None:
        pytest.fail("duplicated secondary must produce a violation")
    assert violation["invariant"] == "decision_secondary_does_not_duplicate_primary"


def test_check_secondary_does_not_duplicate_primary_passes_when_different() -> None:
    """Different secondary passes."""
    primary = MapOpenCommandDict(cmd_type="map_open")
    secondary = RadarCommandDict(cmd_type="radar")
    decision = _decision_with_command(primary, secondary=secondary)
    assert check_secondary_does_not_duplicate_primary(decision) is None


def test_check_secondary_does_not_duplicate_primary_passes_with_no_secondary() -> None:
    """Absent secondary passes."""
    decision = _decision_with_command(MapOpenCommandDict(cmd_type="map_open"), secondary=None)
    assert check_secondary_does_not_duplicate_primary(decision) is None


# ---------------------------------------------------------------------
# Suite-level helper: assert_no_violations
# ---------------------------------------------------------------------


def test_assert_no_violations_passes_on_clean_decision() -> None:
    """The pytest helper does not raise on a clean decision."""
    scenario = BotScenario()
    scenario.place_self(x=100, y=100, fuel=800)
    decision = scenario.decide()
    self_state = scenario.self_state
    if self_state is None:
        pytest.fail("place_self must populate self_state")
    assert_no_violations(decision, self_state, scenario.inventory)


def test_assert_no_violations_raises_on_violation() -> None:
    """The pytest helper raises an AssertionError naming the violation."""
    scenario = BotScenario()
    scenario.place_self(x=100, y=100, fuel=800)
    self_state = scenario.self_state
    if self_state is None:
        pytest.fail("place_self must populate self_state")

    bad_decision = _decision_with_command(
        TeleportCommandDict(cmd_type="teleport", target_x=0, target_y=0)
    )
    with pytest.raises(AssertionError, match="decision_does_not_teleport_to_origin_sentinel"):
        assert_no_violations(bad_decision, self_state, _full_inventory())
