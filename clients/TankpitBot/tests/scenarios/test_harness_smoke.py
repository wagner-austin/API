"""Smoke test for the BotScenario harness itself.

Proves the harness imports cleanly, constructs a minimal scenario,
ingests typed messages through the real dispatcher, and runs the
production ``decide()`` function without exception. Failures here
block the rest of the scenarios package from running, so we keep
this file deliberately tiny and the assertions strong (no weak
``is not None`` checks, no key-existence assertions).
"""

from __future__ import annotations

import pytest

from tests.scenarios._harness import (
    DEFAULT_SELF_FUEL,
    DEFAULT_SELF_TANK_ID,
    DEFAULT_SELF_TEAM,
    DEFAULT_START_TIMESTAMP_MS,
    BotScenario,
)


@pytest.fixture()
def scenario() -> BotScenario:
    """Yield a fresh scenario; world state is reset by construction.

    Returns:
        A :class:`BotScenario` whose dispatcher has been freshly
        initialised. The world has no tanks and no self_state.
    """
    return BotScenario()


def test_construction_clears_world_state(scenario: BotScenario) -> None:
    """A freshly-constructed scenario has an empty, untouched world."""
    if scenario.self_state is not None:
        pytest.fail("self_state must be None on a fresh scenario")
    assert scenario.world["tanks"] == {}
    assert scenario.world["containers"] == {}
    assert scenario.world["mines"] == {}
    assert scenario.timestamp_ms == DEFAULT_START_TIMESTAMP_MS


def test_place_self_establishes_self_state(scenario: BotScenario) -> None:
    """``place_self`` drives a real MovementResponse + FuelGain."""
    scenario.place_self(x=100, y=100, fuel=750)

    self_state = scenario.self_state
    if self_state is None:
        pytest.fail("place_self must populate self_state")
    assert self_state["x"] == 100
    assert self_state["y"] == 100
    assert self_state["fuel"] == 750
    assert self_state["tank_id"] == DEFAULT_SELF_TANK_ID
    assert self_state["team"] == DEFAULT_SELF_TEAM


def test_place_self_defaults_fuel_to_well_above_critical(
    scenario: BotScenario,
) -> None:
    """The default fuel keeps the bot out of COLLECT on the first tick."""
    scenario.place_self(x=10, y=10)
    self_state = scenario.self_state
    if self_state is None:
        pytest.fail("place_self must populate self_state")
    assert self_state["fuel"] == DEFAULT_SELF_FUEL


def test_place_enemy_registers_threat_candidate(scenario: BotScenario) -> None:
    """``place_enemy`` makes the enemy visible to ``analyze_threats``."""
    from tankpit_bot.bot.ai.threats import analyze_threats

    scenario.place_self(x=100, y=100)
    scenario.place_enemy(tank_id=5, x=99, y=100, name="orange-3")

    self_state = scenario.self_state
    if self_state is None:
        pytest.fail("place_self must populate self_state")
    threats = analyze_threats(scenario.world, self_state, scenario.timestamp_ms)
    threat_ids = [threat["tank_id"] for threat in threats]
    assert threat_ids == [5]


def test_decide_returns_a_production_tick_decision(
    scenario: BotScenario,
) -> None:
    """``decide()`` invokes the production pipeline and returns a typed result.

    The harness must not crash on the simplest possible input (own
    tank placed, no enemies, no resources). The decision must carry
    a valid behavior mode -- we check the value, not just existence.
    """
    scenario.place_self(x=100, y=100)

    decision = scenario.decide()

    behavior_mode = decision["behavior"]["mode"]
    assert behavior_mode in {
        "HUNT",
        "COLLECT",
        "PATROL",
        "IDLE",
    }
    command_type = decision["command"]["cmd_type"]
    assert command_type in {
        "radar",
        "shoot",
        "teleport",
        "move",
        "map_open",
        "none",
    }
    assert decision["updated_ai_state"]["mode"] in {
        "HUNT",
        "COLLECT",
        "UNSET",
    }


def test_advance_clock_increments_timestamp(scenario: BotScenario) -> None:
    """``advance_clock`` bumps the scenario clock by the given delta."""
    start = scenario.timestamp_ms
    scenario.advance_clock(delta_ms=2500)
    assert scenario.timestamp_ms == start + 2500


def test_decide_raises_without_self_state(scenario: BotScenario) -> None:
    """Calling decide() without place_self() is a clear error, not a crash."""
    with pytest.raises(RuntimeError, match="self_state"):
        scenario.decide()


def test_decide_many_runs_n_ticks_and_advances_clock(
    scenario: BotScenario,
) -> None:
    """``decide_many`` returns one decision per tick and advances the clock."""
    scenario.place_self(x=100, y=100)
    start = scenario.timestamp_ms

    decisions = scenario.decide_many(ticks=3)

    assert len(decisions) == 3
    # Clock advanced by exactly 3 * default tick delta.
    assert scenario.timestamp_ms == start + 3 * 1000
