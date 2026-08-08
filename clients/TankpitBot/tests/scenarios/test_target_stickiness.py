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

from tankpit_bot.bot.ai.threat_primitives import WIRE_PRESENCE_TTL_MS
from tankpit_bot.bot.ai.threats import analyze_threats
from tankpit_bot.sniffer.world_service import WorldService
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
    scenario.place_self(x=100, y=100, fuel=1200)
    # Closer enemy 1 tile west of self.
    scenario.place_enemy(tank_id=TARGET_TANK_ID, x=99, y=100, name="red-51")
    # Rival enemy 3 tiles south-east of self.
    scenario.place_enemy(tank_id=RIVAL_TANK_ID, x=103, y=103, name="red-52")

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
    scenario.place_self(x=100, y=100, fuel=1200)
    scenario.place_enemy(tank_id=TARGET_TANK_ID, x=99, y=100, name="red-51")
    scenario.place_enemy(tank_id=RIVAL_TANK_ID, x=103, y=103, name="red-52")

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


def test_bot_holds_lock_when_target_drops_off_threat_list() -> None:
    """The lock is preserved when the threat list loses the target.

    Behavior contract (user-confirmed, refined 2026-06-23): when a
    locked target ages out of the viewport-confirmed threat list,
    the bot does NOT release the lock or fall to CONFIRM_KILL. It
    treats the target as off-viewport (it teleported away or moved
    out of sight) and fires homing toward the last known wire
    position via ``_locked_target_pursuit``. Homing tracks, so the
    server picks an angle that can still land if the target is
    nearby. The lock holds until an authoritative deactivation
    signal (``liveness`` flips to ``deactivated`` or the tank lands
    in ``killed_tank_ids``).

    Pre-2026-06-23 there were two competing gates that broke this:
    the viewport-presence gate dropped the lock when the target
    aged out, and the wire-presence gate in ``engage_target``
    blocked pursuit shots after the wire TTL elapsed. Both removed
    -- the test was flipped 2026-06-23 to assert lock preservation
    instead of release.

    Setup: two enemies at tick 0; bot acquires the closer one.
    Advance the clock past ``WIRE_PRESENCE_TTL_MS`` without
    re-confirming the locked target so it leaves the threat list.
    Rival gets a fresh MovementResponse so it stays in the threat
    list. Next decision MUST still be locked to the original
    target, firing pursuit homing.
    """
    scenario = BotScenario()
    scenario.place_self(x=100, y=100, fuel=1200)
    scenario.place_enemy(tank_id=TARGET_TANK_ID, x=99, y=100, name="red-51")
    scenario.place_enemy(tank_id=RIVAL_TANK_ID, x=103, y=103, name="red-52")

    tick0 = scenario.decide()
    assert tick0["updated_ai_state"]["combat_target_id"] == TARGET_TANK_ID

    # Advance past the wire-presence TTL so the locked tank's threat
    # entry expires; rival remains because we refresh its wire timestamp.
    scenario.advance_clock(delta_ms=WIRE_PRESENCE_TTL_MS + 1000)
    scenario.ingest(movement_response(tank_id=RIVAL_TANK_ID, x=103, y=103, team=1, rank=1))

    self_state = scenario.self_state
    if self_state is None:
        pytest.fail("place_self must populate self_state")
    threats = analyze_threats(WorldService(), scenario.world, self_state, scenario.timestamp_ms)
    threat_ids = sorted(threat["tank_id"] for threat in threats)
    assert threat_ids == [RIVAL_TANK_ID], (
        "precondition: locked target should have aged out of the threat list"
    )

    tick1 = scenario.decide()
    assert tick1["updated_ai_state"]["combat_target_id"] == TARGET_TANK_ID, (
        "lock preservation contract: locked target stays locked through pursuit even "
        "when it ages out of the viewport-confirmed threat list; the bot fires homing "
        "toward last wire position rather than releasing"
    )
