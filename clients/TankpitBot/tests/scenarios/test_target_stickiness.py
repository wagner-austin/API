"""Failure-mode regression: bot must stay on a target after engaging.

Encodes the user-stated rule "don't keep hopping enemies after one
hit". The bot picks a target, fires, the target is still alive next
tick -- the bot MUST continue engaging the same target. Hopping to a
different (possibly closer) enemy after a single hit is the failure
mode this scenario locks down.

The scenarios use real wire-message dispatch and the production
``decide()`` function; the only test-specific code constructs input
messages and asserts on the returned :class:`TickDecisionDict`.
"""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.threats import WIRE_PRESENCE_TTL_MS, analyze_threats
from tests.scenarios._harness import BotScenario
from tests.scenarios._wire_builders import movement_response

#: Tank ids used by the scenarios. Disjoint values so violation
#: messages name the offender unambiguously.
TARGET_TANK_ID: int = 511
RIVAL_TANK_ID: int = 612


def test_bot_acquires_the_closer_of_two_visible_enemies() -> None:
    """Baseline: with two enemies in view, HUNT locks the closer one.

    The durable lock lives on ``updated_ai_state.combat_target_id``;
    the ``behavior.target_id`` is the current tick's engagement
    scoring target and is intentionally 0 during teleport-close
    phases. Establishes the precondition for the stickiness tests --
    if the bot doesn't lock the closer target on tick 0, the
    stickiness tests below would be measuring the wrong thing.
    """
    scenario = BotScenario()
    scenario.place_self(x=100, y=100, fuel=800)
    # Closer enemy 1 tile west of self.
    scenario.place_enemy(tank_id=TARGET_TANK_ID, x=99, y=100, name="closer")
    # Rival enemy 3 tiles south-east of self.
    scenario.place_enemy(tank_id=RIVAL_TANK_ID, x=103, y=103, name="farther")

    decision = scenario.decide()

    assert decision["behavior"]["mode"] == "HUNT"
    assert decision["updated_ai_state"]["combat_target_id"] == TARGET_TANK_ID


def test_bot_stays_on_target_after_one_shot_when_target_is_still_alive() -> None:
    """The bot does NOT switch targets between ticks while the current one lives.

    Two enemies are visible; the bot acquires the closer one on tick
    0. On tick 1 the bot is still on the same target -- the
    ``combat_target_id`` field of ``ai_state`` carries the lock, and
    the next decision MUST stay on it. If the rival becomes "closer"
    by virtue of our tank moving, that does NOT justify dropping a
    live engaged target.
    """
    scenario = BotScenario()
    scenario.place_self(x=100, y=100, fuel=800)
    scenario.place_enemy(tank_id=TARGET_TANK_ID, x=99, y=100, name="locked")
    scenario.place_enemy(tank_id=RIVAL_TANK_ID, x=103, y=103, name="rival")

    tick0 = scenario.decide()
    assert tick0["updated_ai_state"]["combat_target_id"] == TARGET_TANK_ID

    # Advance one tick without any state change: target is still alive,
    # still at the same tile. The bot must not abandon it.
    scenario.advance_clock()
    tick1 = scenario.decide()

    assert tick1["behavior"]["mode"] == "HUNT"
    assert tick1["updated_ai_state"]["combat_target_id"] == TARGET_TANK_ID, (
        "bot dropped the engaged target with no provocation; this is the "
        "'hop after one hit' failure mode the test guards against"
    )


def test_bot_releases_lock_when_target_drops_off_threat_list() -> None:
    """The lock is released the moment the threat list loses the target.

    Live-run 2026-06-21 tracking probe proved the pre-fix
    world-state fallback was the second source of the "fires one
    shot then hops" failure loop: it kept locks alive on tanks the
    JS client itself no longer listed in ``activeGame.P.j``,
    sending the bot teleporting after phantoms. The fix removed
    the fallback. This test locks the new behaviour in: when the
    viewport-presence gate fires on the locked target,
    ``get_locked_target`` returns ``None``, ``_decide_hunt_engage``
    enters ``CONFIRM_KILL``, and the bot reacquires from fresh
    intel instead of chasing a stale registry position.

    Setup: two enemies at tick 0; bot acquires the closer one.
    Advance the clock past ``VIEWPORT_PRESENCE_TTL_MS`` without
    re-confirming the locked target so it leaves the threat list.
    Rival gets a fresh MovementResponse so it stays in the threat
    list. Next decision MUST not be locked to the disappeared
    target.
    """
    scenario = BotScenario()
    scenario.place_self(x=100, y=100, fuel=800)
    scenario.place_enemy(tank_id=TARGET_TANK_ID, x=99, y=100, name="locked")
    scenario.place_enemy(tank_id=RIVAL_TANK_ID, x=103, y=103, name="rival")

    tick0 = scenario.decide()
    assert tick0["updated_ai_state"]["combat_target_id"] == TARGET_TANK_ID

    # Advance past the wire-presence TTL so the locked tank's threat
    # entry expires; rival remains because we refresh its wire timestamp.
    scenario.advance_clock(delta_ms=WIRE_PRESENCE_TTL_MS + 1000)
    scenario.ingest(movement_response(tank_id=RIVAL_TANK_ID, x=103, y=103, team=1, rank=1))

    self_state = scenario.self_state
    if self_state is None:
        pytest.fail("place_self must populate self_state")
    threats = analyze_threats(scenario.world, self_state, scenario.timestamp_ms)
    threat_ids = sorted(threat["tank_id"] for threat in threats)
    assert threat_ids == [RIVAL_TANK_ID], (
        "precondition: locked target should have aged out of the threat list"
    )

    tick1 = scenario.decide()
    assert tick1["updated_ai_state"]["combat_target_id"] != TARGET_TANK_ID, (
        "bot is still locked to a phantom whose viewport-presence gate fired; "
        "the world-fallback that masked this divergence was removed 2026-06-21"
    )
